#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch_geometric.loader import DataLoader
from models import QHNet
from nap_dataset_dzp_pyscf import NapDZPOfficialDBDataset

from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from argparse import Namespace
from pyscf import gto
from pyscf.data.elements import ELEMENTS

# ============================================================
# Convention dict (DZP)
#   H: ss (2)
#   C: ssssppd (4s + 2p + 1d -> 4 + 6 + 5 = 15)
# ============================================================
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
# Helper: safely cast PyG batch
# ============================================================
def cast_batch_float(batch, dtype=torch.float32):
    for key in batch.keys:
        val = batch[key]
        if torch.is_tensor(val) and torch.is_floating_point(val):
            batch[key] = val.to(dtype)
    return batch


# ============================================================
# QM orbital solver (double precision)
# ============================================================
def cal_orbital_and_energies(overlap_matrix, full_hamiltonian):
    """
    overlap_matrix: [B, nAO, nAO]
    full_hamiltonian: [B, nAO, nAO]
    """
    eigvals, eigvecs = torch.linalg.eigh(overlap_matrix)
    eigvals = torch.where(eigvals > 1e-8, eigvals, 1e-8 * torch.ones_like(eigvals))

    frac_overlap = eigvecs / torch.sqrt(eigvals).unsqueeze(-2)
    Fs = torch.bmm(
        torch.bmm(frac_overlap.transpose(-1, -2), full_hamiltonian),
        frac_overlap
    )
    orbital_energies, orbital_coefficients = torch.linalg.eigh(Fs)
    orbital_coefficients = torch.bmm(frac_overlap, orbital_coefficients)
    return orbital_energies, orbital_coefficients


# ============================================================
# Sign alignment in raw AO basis
# ============================================================
def align_homo_sign(c_qm, c_ml, overlap):
    s = torch.einsum("bi,bij,bj->b", c_qm, overlap, c_ml)
    sign = torch.where(s >= 0, 1.0, -1.0).to(c_ml.dtype)
    return c_ml * sign[:, None]


# ============================================================
# Build PySCF mol + overlap (Q-Chem DZP .dat), NO padding
# ============================================================
def build_mol_and_overlap_from_batch_dzp(batch, basis_dat_path: str, device):
    atoms = batch.atoms.view(-1)
    atoms = atoms[atoms > 0]  # remove padding zeros
    n_atoms = int(atoms.numel())

    pos = batch.pos[:n_atoms].detach().cpu().numpy()

    atom_list = []
    for i in range(n_atoms):
        z = int(atoms[i].cpu().item())
        sym = ELEMENTS[z]
        atom_list.append([sym, pos[i].tolist()])

    gto_basis = {
        "H": gto.load(basis_dat_path, "H"),
        "C": gto.load(basis_dat_path, "C"),
        "N": gto.load(basis_dat_path, "N"),
        "O": gto.load(basis_dat_path, "O"),
        "F": gto.load(basis_dat_path, "F"),
    }

    mol = gto.Mole()
    mol.atom = atom_list
    mol.basis = gto_basis
    mol.unit = "ang"
    mol.cart = False
    mol.charge = 0
    mol.spin = 0
    mol.build(verbose=0)

    # PySCF AO ordering
    S = mol.intor("int1e_ovlp")
    S = torch.from_numpy(S).unsqueeze(0).to(device=device, dtype=torch.float64)

    return mol, atoms.detach().cpu().numpy(), S


# ============================================================
# Evaluation + HOMO coeff collection (RAW AO only)
# ============================================================
@torch.no_grad()
def collect_homo_coefficients_dzp(dataloader, model, device, basis_dat_path):
    model.eval()

    all_homo_qm = []
    all_homo_ml = []
    all_homo_mae_raw = []
    all_h_mae = []

    for it, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = cast_batch_float(batch, torch.float32)
        batch = batch.to(device, non_blocking=(device.type == "cuda"))
        print(batch)
        sys.exit()
        outputs = model(batch)

        # H from model/dataset (internal ordering)
        H_ml = model.build_final_matrix(
            batch,
            outputs["hamiltonian_diagonal_blocks"],
            outputs["hamiltonian_non_diagonal_blocks"],
        ).to(torch.float64)

        H_qm = model.build_final_matrix(
            batch,
            batch[0].diagonal_hamiltonian,
            batch[0].non_diagonal_hamiltonian,
        ).to(torch.float64)

        # Build PySCF overlap (PySCF ordering) + mol for nelectron
        mol, atoms_np, overlap = build_mol_and_overlap_from_batch_dzp(
            batch, basis_dat_path=basis_dat_path, device=device
        )

        # IMPORTANT: transform H -> PySCF ordering so that it matches overlap
        H_ml = matrix_transform(H_ml, atoms_np, convention="back2pyscf_dzp")
        H_qm = matrix_transform(H_qm, atoms_np, convention="back2pyscf_dzp")
        # overlap is already PySCF ordering -> DO NOT transform overlap

        # dense H MAE
        h_mae = float(torch.mean(torch.abs(H_ml - H_qm)).item())
        all_h_mae.append(h_mae)

        # Solve orbitals
        _, C_ml = cal_orbital_and_energies(overlap, H_ml)
        _, C_qm = cal_orbital_and_energies(overlap, H_qm)

        # HOMO index from mol.nelectron
        homo_idx = mol.nelectron // 2 - 1

        homo_qm = C_qm[:, :, homo_idx]   # [1, nAO]
        homo_ml = C_ml[:, :, homo_idx]   # [1, nAO]

        # sign alignment in raw AO basis
        homo_ml = align_homo_sign(homo_qm, homo_ml, overlap)

        # raw coeff MAE
        mae_raw = torch.mean(torch.abs(homo_ml - homo_qm), dim=1)  # [1]

        all_homo_qm.append(homo_qm.cpu().reshape(-1))
        all_homo_ml.append(homo_ml.cpu().reshape(-1))
        all_homo_mae_raw.append(mae_raw.cpu())

        if it < 20:
            print(
                f"[dbg] i={it} homo_idx={homo_idx} | "
                f"mae_raw={float(mae_raw.item()):.3e} | H_mae={h_mae:.3e}"
            )

    all_homo_qm = torch.cat(all_homo_qm).numpy()
    all_homo_ml = torch.cat(all_homo_ml).numpy()
    all_homo_mae_raw = torch.cat(all_homo_mae_raw).numpy()
    all_h_mae = np.array(all_h_mae, dtype=np.float64)

    return all_homo_qm, all_homo_ml, all_homo_mae_raw, all_h_mae


# ============================================================
# Main
# ============================================================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[INFO] device =", device)

    # ====== TODO: change these paths if needed ======
    db_path = "/Data2/jason/paper/0_monomer_dataset_for_qhnet/nap_mono_10000.db"
    ckpt_path = "/Data2/jason/paper/1_ML1_training/nap_1k/weights/results_best_homo.pt"
    basis_dat_path = "/Data2/jason/qhnet/AIRS/OpenDFT/QHBench_dzp/QH9/dzp_q-chem.dat"

    # -------- Dataset --------
    dataset = NapDZPOfficialDBDataset(
        db_path=db_path,
        convention="pyscf_dzp",
        max_items=1000,
        seed=42,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        include_overlap=True,
    )

    test_dataset = dataset[dataset.test_mask[:]]
    loader = DataLoader(
        test_dataset,
        batch_size=1,
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

    homo_qm, homo_ml, homo_mae_raw, h_mae = collect_homo_coefficients_dzp(
        loader, model, device, basis_dat_path=basis_dat_path
    )

    print("\nHOMO coefficient MAE (RAW AO):")
    print(f"  mean = {homo_mae_raw.mean():.6e}")
    print(f"  std  = {homo_mae_raw.std():.6e}")
    print(f"  max  = {homo_mae_raw.max():.6e}")

    print("\nH dense MAE (PySCF AO order):")
    print(f"  mean = {h_mae.mean():.6e}")
    print(f"  std  = {h_mae.std():.6e}")
    print(f"  max  = {h_mae.max():.6e}")

    # ---------------- Plot (raw AO scatter) ----------------
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.scatter(homo_qm, homo_ml, marker="x", color="red", s=8, alpha=0.6)

    all_vals = np.concatenate([homo_qm, homo_ml])
    vmax = np.max(np.abs(all_vals))
    vmax = np.ceil(vmax * 10) / 10

    ax.set_xlim(-vmax, vmax)
    ax.set_ylim(-vmax, vmax)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    l = np.linspace(-vmax, vmax, 200)
    ax.plot(l, l, color="black", linewidth=1)

    mae = homo_mae_raw.mean()
    mae_str = f"{mae:.3e}"
    n, e = mae_str.split("e")
    ax.annotate(
        fr"MAE = {n} $\times$ $10^{{{int(e)}}}$",
        xy=(0.95, 0.05),
        xycoords="axes fraction",
        fontsize=16,
        ha="right",
        va="bottom",
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
    )

    ax.set_xlabel("QM HOMO Coefficients", fontsize=18)
    ax.set_ylabel("ML HOMO Coefficients", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_aspect("equal", adjustable="box")

    plt.savefig("HOMO_coeff_QM_vs_ML_raw_dzp.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
