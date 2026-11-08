#!/usr/bin/env python
import os
import sys
import numpy as np
import torch
import hydra
from ase.db import connect
from ase.io import read, write
from argparse import Namespace
from torch.utils.data import Subset
from pyscf import gto
from pyscf.data.elements import ELEMENTS
import time
from models import QHNet
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_ema import ExponentialMovingAverage
from transformers import get_polynomial_decay_schedule_with_warmup
from datetime import datetime
from tqdm import tqdm
from rotation import StandardOrientation, RotateOrbital
from nap_dataset_dzp_pyscf import NapDZPOfficialDBDataset, EvaluateDataset
from ase import Atoms
from overlap_matrix import OverlapMatrix
import itertools as it
from mlff.data.dataloader import AseDataLoader
from mlff.properties import md17_property_keys as prop_keys
from mlff.data import DataTuple, DataSet
from mlff.training import Coach
import mlff.properties.property_names as pn
from mlff.nn.stacknet import StackNet
from mlff.io import read_json, load_params_from_ckpt_dir
from mlff.nn import So3krates
from mlff.nn.stacknet import get_obs_and_force_fn, get_observable_fn, get_energy_force_stress_fn
import jax

orbital_length = {'C': 15, 'H': 2}

convention_dict = {
    "pyscf_dzp": Namespace(
        atom_to_orbitals_map={
            1: "ss",
            6: "ssssppd",
            7: "sssspp",
            8: "sssspp",
            9: "sssspp",
        },
        orbital_idx_map={"s": [0], "p": [1, 2, 0], "d": [0, 1, 2, 3, 4]},
        orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1],                 # ss
            6: [0, 1, 2, 3, 4, 5, 6], # ssss pp d
            7: [0, 1, 2, 3, 4, 5],    # ssss pp
            8: [0, 1, 2, 3, 4, 5],
            9: [0, 1, 2, 3, 4, 5],
        },
    ),
    "back2pyscf_dzp": Namespace(
        atom_to_orbitals_map={
            1: "ss",
            6: "ssssppd",
            7: "sssspp",
            8: "sssspp",
            9: "sssspp",
        },
        orbital_idx_map={"s": [0], "p": [2, 0, 1], "d": [0, 1, 2, 3, 4]},
        orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1],
            6: [0, 1, 2, 3, 4, 5, 6],
            7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5],
            9: [0, 1, 2, 3, 4, 5],
        },
    ),
}

def cast_batch_float(batch, dtype=torch.float32):
    for key in batch.keys():
        val = batch[key]
        if torch.is_tensor(val) and torch.is_floating_point(val):
            batch[key] = val.to(dtype)
    return batch

def cal_orbital_and_energies(overlap_matrix, full_hamiltonian):
    """
    overlap_matrix: [B, nAO, nAO]
    full_hamiltonian: [B, nAO, nAO]
    """
    eigvals, eigvecs = torch.vmap(torch.linalg.eigh)(overlap_matrix)
    eigvals = torch.where(eigvals > 1e-8, eigvals, 1e-8 * torch.ones_like(eigvals))

    frac_overlap = eigvecs / torch.sqrt(eigvals).unsqueeze(-2)
    Fs = torch.bmm(
        torch.bmm(frac_overlap.transpose(-1, -2), full_hamiltonian),
        frac_overlap
    )
    orbital_energies, orbital_coefficients = torch.vmap(torch.linalg.eigh)(Fs)
    orbital_coefficients = torch.bmm(frac_overlap, orbital_coefficients)
    return orbital_energies, orbital_coefficients

def matrix_transform(matrices: torch.Tensor, atoms_np: np.ndarray, convention: str = "back2pyscf_dzp"):
    """
    Reorder/sign-fix AO matrices to match target convention.

    matrices: (..., nAO, nAO) torch Tensor
    atoms_np: (n_atoms,) numpy array of Z, WITHOUT padding zeros
    """
    conv = convention_dict[convention]

    orbitals = ""
    orbitals_order = []
    for a in atoms_np:
        a = int(a)
        if a <= 0:
            raise ValueError(f"[matrix_transform] non-positive Z in atoms: {a}")
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[a]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a]]

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        map_sign = conv.orbital_sign_map[orb]
        transform_indices.append(np.array(map_idx) + offset)
        transform_signs.append(np.array(map_sign))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = np.concatenate(transform_indices).astype(np.int32)
    transform_signs = np.concatenate(transform_signs)

    transform_indices = torch.from_numpy(transform_indices).to(matrices.device).long()
    transform_signs = torch.from_numpy(transform_signs).to(matrices.device, dtype=matrices.dtype)

    M = matrices[..., transform_indices, :]
    M = M[..., :, transform_indices]
    M = M * transform_signs[:, None]
    M = M * transform_signs[None, :]
    return M

# ============================================================
# Evaluation + HOMO coeff collection (RAW AO only)
# ============================================================
@torch.no_grad()
def collect_homo_coefficients_dzp(dataloader, model, torch_overlap_matrix, device, basis_dat_path):
    model.eval()

    all_homo_ml = []
    for it, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        start = time.time()
        batch = cast_batch_float(batch, torch.float32)
        batch = batch.to(device, non_blocking=(device.type == "cuda"))

        outputs = model(batch)
        # H from model/dataset (internal ordering)
        #H_ml = model.build_final_matrix(
        #    batch,
        #    outputs["hamiltonian_diagonal_blocks"],
        #    outputs["hamiltonian_non_diagonal_blocks"],
        #).to(torch.float64)

        H_ml = model.build_final_matrix_from_index(
            batch,
            outputs["hamiltonian_diagonal_blocks"],
            outputs["hamiltonian_non_diagonal_blocks"],
        ).to(torch.float64)

        atoms = batch.atoms.view(-1)
        atoms = atoms[atoms > 0]  # remove padding zeros
        atoms_np = atoms.detach().cpu().numpy()

        # Build PySCF overlap (PySCF ordering) + mol for nelectron
        #mol, atoms_np, overlap = build_mol_and_overlap_from_batch_dzp(
        #    batch, basis_dat_path=basis_dat_path, device=device
        #)
        inputs_overlap = {}
        num_batch = torch.unique(batch.batch).shape[0]
        inputs_overlap["R"] = batch.pos.reshape((num_batch, -1, 3))
        inputs_overlap["cell"] = torch.zeros((num_batch, 3, 3)).to("cuda")
        inputs_overlap["offsets"] = torch.zeros((num_batch, inputs_overlap["R"].shape[1],
            inputs_overlap["R"].shape[1], 3)).to("cuda")
        overlap = torch_overlap_matrix(inputs_overlap)
        atoms_np = atoms_np[:inputs_overlap["R"].shape[1]]
        H_ml = matrix_transform(H_ml, atoms_np, convention="back2pyscf_dzp")
        _, C_ml = cal_orbital_and_energies(overlap.double(), H_ml)
        all_homo_ml.append(C_ml.detach().cpu().numpy())
    all_homo_ml = np.concatenate(all_homo_ml, axis=0)
    return all_homo_ml

def get_overlaps(orb1, 
        orb2,
        index,
        dataset,
        iteration,
        batch_size,
        mass,
        so,
        rotate_orbital,
        ZA,
        positions_opt,
        torch_overlap_matrix_dimer,
        homo_opt,
        symbols_mon1,
        symbols):
    #system depend
    orb1 = orb1[:, :, index]
    orb2 = orb2[:, :, index]
    
    data = dataset[np.arange(iteration, iteration+batch_size)]
    positions1 = []
    for m1 in range(0, batch_size, 2):
        positions1.append(data[m1]["pos"][None])
    positions1 = torch.cat(positions1, dim=0)

    positions2 = []
    for m2 in range(1, batch_size, 2):
        positions2.append(data[m2]["pos"][None])
    positions2 = torch.cat(positions2, dim=0)

    positions = torch.cat((positions1, positions2), dim=1).to("cuda")

    inputs = {}
    inputs["monA"] = torch.arange(0, positions1.shape[1])
    inputs["monB"] = torch.arange(positions1.shape[1], positions1.shape[1]+positions2.shape[1])
    inputs["mass"] = mass.to("cuda")
    inputs["R"] = positions.to("cuda")
    output = so.forward(inputs)

    orb1_rotate = rotate_orbital(orb1, output["frag1_rmatrix"].transpose(2,1), torch.from_numpy(ZA)[None])
    orb2_rotate = rotate_orbital(orb2, output["frag2_rmatrix"].transpose(2,1), torch.from_numpy(ZA)[None])

    inputs_overlap = {}
    positions_opt_t = torch.tile(positions_opt[None], (output["frag1_pos"].shape[0], 1, 1))
    inputs_overlap["R"] = torch.cat((positions_opt_t, output["frag1_pos"]), dim=1)
    inputs_overlap["cell"] = torch.zeros((positions_opt_t.shape[0], 3, 3)).to("cuda")
    inputs_overlap["offsets"] = torch.zeros((positions_opt_t.shape[0], inputs_overlap["R"].shape[1],
        inputs_overlap["R"].shape[1], 3)).to("cuda")
    S = torch_overlap_matrix_dimer(inputs_overlap)

    homo_opt_p = torch.tile(homo_opt[None], (orb1_rotate.shape[0], 1, 1))
    orb1_rotate_p = orb1_rotate[..., None]
    orb_append = torch.zeros_like(homo_opt_p)
    wDD = torch.cat((homo_opt_p, orb_append), dim=1)
    wAA = torch.cat((orb_append, orb1_rotate_p), dim=1)
    wDA = torch.bmm(wDD, wAA.transpose(1,2))

    s = torch.vmap(torch.trace)(torch.bmm(wDA.float(), S.float()))
    orb1_rotate *= torch.sign(s)[:, None]
    orb1 = rotate_orbital(orb1_rotate, output["frag1_rmatrix"], torch.from_numpy(ZA)[None])

    inputs_overlap = {}
    positions_opt_t = torch.tile(positions_opt[None], (output["frag2_pos"].shape[0], 1, 1))
    inputs_overlap["R"] = torch.cat((positions_opt_t, output["frag2_pos"]), dim=1)
    inputs_overlap["cell"] = torch.zeros((positions_opt_t.shape[0], 3, 3)).to("cuda")
    inputs_overlap["offsets"] = torch.zeros((positions_opt_t.shape[0], inputs_overlap["R"].shape[1],
        inputs_overlap["R"].shape[1], 3)).to("cuda")
    S = torch_overlap_matrix_dimer(inputs_overlap)

    homo_opt_p = torch.tile(homo_opt[None], (orb2_rotate.shape[0], 1, 1))
    orb2_rotate_p = orb2_rotate[..., None]
    orb_append = torch.zeros_like(homo_opt_p)

    wDD = torch.cat((homo_opt_p, orb_append), dim=1)
    wAA = torch.cat((orb_append, orb2_rotate_p), dim=1)
    wDA = torch.bmm(wDD, wAA.transpose(1,2))

    s = torch.vmap(torch.trace)(torch.bmm(wDA.float(), S.float()))
    orb2_rotate *= torch.sign(s)[:, None]
    orb2 = rotate_orbital(orb2_rotate, output["frag2_rmatrix"], torch.from_numpy(ZA)[None])

    inputs_overlap = {}
    inputs_overlap["R"] = positions
    inputs_overlap["cell"] = torch.zeros((positions_opt_t.shape[0], 3, 3)).to("cuda")
    inputs_overlap["offsets"] = torch.zeros((positions_opt_t.shape[0], inputs_overlap["R"].shape[1],
        inputs_overlap["R"].shape[1], 3)).to("cuda")
    S = torch_overlap_matrix_dimer(inputs_overlap)

    orb1 = orb1[..., None]
    orb2 = orb2[..., None]
    orb_append = torch.zeros_like(orb1)
    wDD = torch.cat((orb1, orb_append), dim=1)
    wAA = torch.cat((orb_append, orb2), dim=1)
    wDA = torch.bmm(wDD, wAA.transpose(1,2))

    s = torch.vmap(torch.trace)(torch.bmm(wDA.float(), S.float()))
    #Orbital indices for first molecule
    indices_mon1 = []
    start_id = 0
    for s in symbols_mon1:
        num_orbitals = orbital_length[s]
        indices_mon1.append(np.arange(start_id, start_id+num_orbitals))
        start_id += num_orbitals

    #Orbital indices for second molecule
    indices_mon2 = []
    for s in symbols_mon1:
        num_orbitals = orbital_length[s]
        indices_mon2.append(np.arange(start_id, start_id+num_orbitals))
        start_id += num_orbitals

    monomer_1_pairs = list(it.product(indices_mon1, indices_mon2))
    #Overlap between sets of basis functions centered on a given atom
    atom_overlaps = []
    for idx1, idx2 in monomer_1_pairs:
        coef_slice = wDA[:, idx1, :][:, :, idx2]
        S_slice = S[:, idx1, :][:, :, idx2]
        val = torch.vmap(torch.mul)(coef_slice, S_slice)
        val = val.sum(dim=(1,2))
        atom_overlaps.append(val.detach().cpu().numpy())

    monomer_2_pairs = list(it.product(indices_mon2, indices_mon1))
    for idx1, idx2 in monomer_2_pairs:
        coef_slice = wDA[:, idx2, :][:, :, idx1]
        S_slice = S[:, idx2, :][:, :, idx1]
        val = torch.vmap(torch.mul)(coef_slice, S_slice)
        val = val.sum(dim=(1,2))
        atom_overlaps.append(val.detach().cpu().numpy())

    atom_overlaps = np.asarray(atom_overlaps).T
    atom_overlaps = atom_overlaps.reshape(atom_overlaps.shape[0], positions.shape[1], -1)
    return atom_overlaps.sum(axis=-1)

# ============================================================
# Main
# ============================================================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[INFO] device =", device)

    #Reference geometry
    ref = read("naphthalene_monomer.xyz")

    #Naphthalene MO coefficients for the above geometry
    homo_opt = np.load("mo_coef_napthalene_opt.npy")
    #Naphthalene HOMO is index 33
    homo_opt = homo_opt[:, 33]
    homo_opt = torch.from_numpy(homo_opt).to("cuda")
    #Batch size is how many dimers to evaluate at once
    batch_size = 100
    #Moves to origin and rotates
    so = StandardOrientation()
    rotate_orbital = RotateOrbital()

    #Contains naphthalenen dimers
    db_path = "naphthalene_overlap_100.db"
    db_path_coupling = "naphthalene_overlap_pairs.db"
    ckpt_path = "output_models/naphthalene_homo.pt"
    basis_dat_path = "QHNet_train/dzp_q-chem.dat"

    database = connect(db_path)
    traj = database.get(id=1)
    traj = traj.toatoms()
    traj_dimer = traj+database.get(id=2).toatoms()
    
    #PyTorch calculation for AO overlap matrix
    torch_overlap_matrix = OverlapMatrix(ref, False, coord='spherical', order='qchem', basis=basis_dat_path, device='cuda')
    torch_overlap_matrix_dimer = OverlapMatrix(traj_dimer, False, coord='spherical', order='qchem', basis=basis_dat_path, device='cuda')

    # -------- Dataset --------
    dataset = EvaluateDataset(
        db_path=db_path,
        convention="pyscf_dzp",
        max_items=len(database),
        seed=42,
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        include_overlap=True,
    )

    data_index = np.arange(0, len(database), 1)
    #data_index = np.arange(0, 100, 1)
    test_dataset = dataset[data_index]
    one_example = dataset[np.arange(2)]

    one_shot_loader = DataLoader(
        one_example,
        batch_size=1,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )

    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )

    # -------- Model --------
    model = QHNet(
        in_node_features=1,
        sh_lmax=4,
        hidden_size=128,
        bottle_hidden_size=32,
        num_gnn_layers=5,
        max_radius=15,
        num_nodes=10,
        radius_embed_dim=16,
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.set(device)

    #Gets shapes of homo for later work
    for iteri, batch in tqdm(enumerate(one_shot_loader), total=len(one_shot_loader)):
        batch = cast_batch_float(batch, torch.float32)
        batch = batch.to(device, non_blocking=(device.type == "cuda"))

        outputs = model(batch)
        # H from model/dataset (internal ordering)
        H_ml = model.build_final_matrix_indices(
            batch,
            outputs["hamiltonian_diagonal_blocks"],
            outputs["hamiltonian_non_diagonal_blocks"],
        ).to(torch.float64)
        break

    homo_ml = collect_homo_coefficients_dzp(
        loader, model, torch_overlap_matrix, device, basis_dat_path=basis_dat_path
    )

    mass = database.get(1).toatoms().get_masses()
    mass = np.append(mass, mass, axis=0)
    mass = mass[None]
    mass = torch.from_numpy(mass)
    ZA = database.get(1).toatoms().numbers

    symbols_mon1 = database.get(1).toatoms().get_chemical_symbols()
    symbols = []
    for s in symbols_mon1: symbols.append(s)
    for s in database.get(2).toatoms().get_chemical_symbols(): symbols.append(s)

    positions_opt = torch.from_numpy(ref.get_positions()).to("cuda")
    homo_opt = homo_opt[:, None]
    atom_overlaps_data = []
    #Loop through orbitals and get atom_overlaps for each batch with the
    #get_overlaps function. Evaluates using QHNet
    for i in range(0, len(homo_ml), batch_size):
        print(f'it {i}')

        orb1 = torch.from_numpy(homo_ml[i:i+batch_size:2]).to("cuda")
        orb2 = torch.from_numpy(homo_ml[i+1:i+batch_size:2]).to("cuda")
        atom_overlaps = get_overlaps(orb1,
                orb2,
                33,
                dataset,
                i,
                batch_size,
                mass,
                so,
                rotate_orbital,
                ZA,
                positions_opt,
                torch_overlap_matrix_dimer,
                homo_opt,
                symbols_mon1,
                symbols)
        
        atom_overlaps_data.append(atom_overlaps)

    atom_overlaps_data = np.concatenate(atom_overlaps_data, axis=0)

    #The rest of this code is for running S03krates for getting V
    data = AseDataLoader(
        db_path_coupling,
        load_energy_and_forces=False,
        load_overlaps=False
        )
    data = data.load_all(8.0)
    data["overlaps"] = atom_overlaps_data
    prop_keys["overlaps"] = "overlaps"
    data_set = DataSet(data=data, prop_keys=prop_keys)
    data_set.index_split(data_idx_train=np.arange(50),
                     data_idx_valid=np.arange(1),
                     data_idx_test=np.arange(1),
                     training=False,
                     r_cut=8.0,
                     mic=False,
                    )
    d = data_set.get_data_split()

    ckpt_dir = './ckpt_dir'
    coach = Coach(inputs=[pn.atomic_position, pn.atomic_type, pn.idx_i, pn.idx_j, pn.node_mask, "overlaps"],
                  targets=[pn.energy],
                  epochs=1000,
                  training_batch_size=100,
                  validation_batch_size=100,
                  loss_weights={pn.energy: 1.0},
                  ckpt_dir=ckpt_dir,
                  data_path=db_path_coupling,
                  net_seed=0,
                  training_seed=0)
    
    old_ckpt = "output_models/naphthalene_V_model" 
    old_net = StackNet.create_from_ckpt_dir(old_ckpt)
    old_params = load_params_from_ckpt_dir(old_ckpt)

    obs_fn = get_observable_fn(old_net)
    obs_fn = jax.vmap(obs_fn, in_axes=(None, 0))
    
    data_tuple = DataTuple(inputs=coach.inputs,
                           targets=coach.targets,
                           prop_keys=prop_keys)
    
    train_ds, train_obs = data_tuple(d['train'])
    
    n_train = len(train_ds['R'])
    
    n_batches = n_train // 50
    idxs = np.arange(n_train)
    idxs = idxs[:n_batches * 50]  # skip incomplete batch
    idxs = idxs.reshape((n_batches, 50))
    
    coupling = []
    for i, idx in enumerate(idxs):
        input_batch = jax.tree.map(lambda y: y[idx, ...], train_ds)
        obs_pred = obs_fn(old_params, input_batch)
        coupling.append(obs_pred["E"].flatten())
    
    np.save(f"couplings_nap.npy", np.asarray(coupling).flatten())

if __name__ == "__main__":
    main()


