import os
import sys
import numpy as np
import torch
import hydra

from argparse import Namespace
from torch.utils.data import Subset
from pyscf import gto
from pyscf.data.elements import ELEMENTS

from models import QHNet
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_ema import ExponentialMovingAverage
from transformers import get_polynomial_decay_schedule_with_warmup
from datetime import datetime

from nap_dataset_dzp_pyscf import NapDZPOfficialDBDataset

# ============================================================
# Logging
# ============================================================
def log_info(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    sys.stdout.write(f"[{ts}][root][INFO] - {msg}\n")
    sys.stdout.flush()


# ============================================================
# HOMO helper functions
# ============================================================
convention_dict = {
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
    M = M * transform_signs.view(*([1] * (M.dim() - 2)), -1, 1)
    M = M * transform_signs.view(*([1] * (M.dim() - 2)), 1, -1)
    return M


def cast_batch_float(batch, dtype=torch.float32):
    for key in batch.keys:
        val = batch[key]
        if torch.is_tensor(val) and torch.is_floating_point(val):
            batch[key] = val.to(dtype)
    return batch


def post_processing(batch, default_type):
    for key in batch.keys():
        if torch.is_tensor(batch[key]) and torch.is_floating_point(batch[key]):
            batch[key] = batch[key].type(default_type)
    return batch


def cal_orbital_and_energies(overlap_matrix, full_hamiltonian):
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


def align_homo_sign(c_qm, c_ml, overlap):
    s = torch.einsum("bi,bij,bj->b", c_qm, overlap, c_ml)
    sign = torch.where(s >= 0, 1.0, -1.0).to(c_ml.dtype)
    return c_ml * sign[:, None]


def build_mol_and_overlap_from_batch_dzp(batch, basis_dat_path: str, device):
    atoms = batch.atoms.view(-1)
    atoms = atoms[atoms > 0]
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

    S = mol.intor("int1e_ovlp")
    S = torch.from_numpy(S).unsqueeze(0).to(device=device, dtype=torch.float64)

    return mol, atoms.detach().cpu().numpy(), S


def keep_only_first_graph(batch):
    first_data = batch.get_example(0)
    return Batch.from_data_list([first_data])


def compute_homo_mae_loss(batch, outputs, model, basis_dat_path, device):
    H_ml = model.build_final_matrix(
        batch,
        outputs["hamiltonian_diagonal_blocks"],
        outputs["hamiltonian_non_diagonal_blocks"],
    ).to(torch.float64)

    H_qm = model.build_final_matrix(
        batch,
        batch.diagonal_hamiltonian,
        batch.non_diagonal_hamiltonian,
    ).to(torch.float64)

    mol, atoms_np, overlap = build_mol_and_overlap_from_batch_dzp(
        batch, basis_dat_path=basis_dat_path, device=device
    )

    H_ml = matrix_transform(H_ml, atoms_np, convention="back2pyscf_dzp")
    H_qm = matrix_transform(H_qm, atoms_np, convention="back2pyscf_dzp")

    _, C_ml = cal_orbital_and_energies(overlap, H_ml)
    _, C_qm = cal_orbital_and_energies(overlap, H_qm)

    homo_idx = mol.nelectron // 2 - 1
    homo_qm = C_qm[:, :, homo_idx]
    homo_ml = C_ml[:, :, homo_idx]

    homo_ml = align_homo_sign(homo_qm, homo_ml, overlap)

    homo_mae = torch.mean(torch.abs(homo_qm - homo_ml))
    return homo_mae


@torch.no_grad()
def validation_homo_mae_subset(valid_homo_loader, model, device, basis_dat_path, default_type):
    model.eval()
    total_homo_mae = 0.0
    total_items = 0

    for _, batch in enumerate(valid_homo_loader):
        batch = post_processing(batch, default_type)
        batch = batch.to(device, non_blocking=(device.type == "cuda"))

        outputs = model(batch)
        homo_mae = compute_homo_mae_loss(batch, outputs, model, basis_dat_path, device)

        total_homo_mae += homo_mae.item()
        total_items += 1

    return total_homo_mae / max(total_items, 1)


# ============================================================
# Common utility
# ============================================================
def strip_prefix_if_present(state_dict, prefix):
    if len(state_dict) == 0:
        return state_dict
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def drop_keys_containing(state, substrings):
    out = {}
    for k, v in state.items():
        if any(s in k for s in substrings):
            continue
        out[k] = v
    return out


def tensor_dict_to_python(d):
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            if v.numel() == 1:
                out[k] = v.item()
            else:
                out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out


def save_checkpoint(path, model, optimizer, scheduler, ema, epoch, batch_idx,
                    best_homo_val_result, final_train_result, final_val_homo_result, errors):
    ckpt = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "epoch": epoch,
        "batch_idx": batch_idx,
        "best_homo_val_result": best_homo_val_result,
        "final_train_result": final_train_result,
        "final_val_homo_result": final_val_homo_result,
        "eval": tensor_dict_to_python(errors),
    }
    torch.save(ckpt, path)


def train_one_batch(conf, batch, model, optimizer, basis_dat_path, device, default_type):
    batch = keep_only_first_graph(batch)
    batch = post_processing(batch, default_type)
    batch = batch.to(device)

    outputs = model(batch)
    homo_mae = compute_homo_mae_loss(batch, outputs, model, basis_dat_path, device)

    optimizer.zero_grad()
    homo_mae.backward()

    if conf.datasets.use_gradient_clipping:
        torch.nn.utils.clip_grad_norm_(model.parameters(), conf.datasets.clip_norm)

    optimizer.step()

    return {
        "loss": homo_mae,
        "homo_mae": homo_mae.detach(),
    }


# ============================================================
# Main
# ============================================================
@hydra.main(version_base=None, config_path='config', config_name='config')
def main(conf):
    log_info(conf)

    default_type = torch.float32
    torch.set_default_dtype(default_type)

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        gpu_id = conf.device
        if gpu_id < 0 or gpu_id >= n_gpu:
            raise ValueError(
                f"Invalid conf.device={gpu_id}, but only {n_gpu} CUDA device(s) are visible. "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device('cpu')

    include_overlap = getattr(conf.datasets, "include_overlap", True)
    dataset = NapDZPOfficialDBDataset(
        db_path=conf.datasets.db_path,
        convention=conf.datasets.convention,
        max_items=conf.datasets.max_items,
        seed=conf.datasets.seed,
        train_ratio=conf.datasets.train_ratio,
        val_ratio=conf.datasets.val_ratio,
        test_ratio=conf.datasets.test_ratio,
        include_overlap=include_overlap,
    )

    train_dataset = dataset[dataset.train_mask]
    valid_dataset = dataset[dataset.val_mask]

    homo_eval_size = getattr(conf.datasets, "homo_eval_size", 5)
    if homo_eval_size < 0:
        homo_subset_size = len(valid_dataset)
    else:
        homo_subset_size = min(homo_eval_size, len(valid_dataset))

    valid_homo_subset = Subset(valid_dataset, list(range(homo_subset_size)))

    g = torch.Generator()
    g.manual_seed(0)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=conf.datasets.train_batch_size,
        shuffle=True,
        num_workers=conf.datasets.num_workers,
        pin_memory=conf.datasets.pin_memory,
        generator=g,
    )

    val_homo_loader = DataLoader(
        valid_homo_subset,
        batch_size=1,
        shuffle=False,
        num_workers=conf.datasets.num_workers,
        pin_memory=conf.datasets.pin_memory,
    )

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
    model.set(device)

    num_params = sum(p.numel() for p in model.parameters())
    log_info(model)
    log_info(f"the number of parameters in this model is {num_params}.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=conf.datasets.learning_rate,
        betas=(0.99, 0.999),
        amsgrad=False,
    )

    ema = ExponentialMovingAverage(model.parameters(), decay=0.99)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=conf.datasets.warmup_steps,
        num_training_steps=conf.datasets.total_steps,
        lr_end=conf.datasets.lr_end,
        power=1.0,
        last_epoch=-1,
    )

    basis_dat_path = getattr(conf.datasets, "basis_dat_path", None)
    if basis_dat_path is None:
        raise ValueError("Please set conf.datasets.basis_dat_path for HOMO training/evaluation.")

    resume_ckpt = getattr(conf, "resume_ckpt", None)
    use_pretrained = getattr(conf, "use_pretrained", False)

    if isinstance(resume_ckpt, str) and resume_ckpt.strip() == "":
        resume_ckpt = None

    if resume_ckpt is not None and use_pretrained:
        raise ValueError("Please use either resume_ckpt or use_pretrained, not both.")

    start_epoch = 0
    start_batch_idx = 0

    best_homo_val_result = 1.0e9
    final_train_result = 1.0e9
    final_val_homo_result = 1.0e9

    if resume_ckpt is not None:
        if not os.path.isfile(resume_ckpt):
            raise FileNotFoundError(f"resume checkpoint not found: {resume_ckpt}")

        log_info(f"[RESUME] Loading checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location="cpu")

        state = ckpt.get("state_dict", ckpt)
        state = strip_prefix_if_present(state, "module.")
        state = strip_prefix_if_present(state, "model.")

        missing, unexpected = model.load_state_dict(state, strict=False)
        log_info(f"[RESUME] missing_keys={len(missing)}")
        log_info(f"[RESUME] unexpected_keys={len(unexpected)}")
        if len(missing) > 0:
            log_info(f"[RESUME] example missing: {missing[:10]}")
        if len(unexpected) > 0:
            log_info(f"[RESUME] example unexpected: {unexpected[:10]}")

        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            log_info("[RESUME] optimizer loaded")

        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
            log_info("[RESUME] scheduler loaded")

        if ckpt.get("ema") is not None:
            ema.load_state_dict(ckpt["ema"])
            log_info("[RESUME] ema loaded")

        start_epoch = ckpt.get("epoch", 0)
        start_batch_idx = ckpt.get("batch_idx", -1) + 1

        best_homo_val_result = ckpt.get("best_homo_val_result", best_homo_val_result)
        final_train_result = ckpt.get("final_train_result", final_train_result)
        final_val_homo_result = ckpt.get("final_val_homo_result", final_val_homo_result)

        log_info("[RESUME] start_epoch=%s" % start_epoch)
        log_info("[RESUME] start_batch_idx=%s" % start_batch_idx)
        log_info(f"[RESUME] best_homo_val_result={best_homo_val_result:.8f}")

    elif use_pretrained:
        ckpt_path = getattr(conf, "pretrained_model_parameter_dir", None)
        if ckpt_path is None:
            raise ValueError("use_pretrained=True but pretrained_model_parameter_dir is not set")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"pretrained checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        state = strip_prefix_if_present(state, "module.")
        state = strip_prefix_if_present(state, "model.")

        try:
            missing, unexpected = model.load_state_dict(state, strict=False)
            log_info(f"[PRETRAIN] loaded FULL: {ckpt_path}")
            log_info(f"[PRETRAIN] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
        except RuntimeError as e:
            log_info("[PRETRAIN] FULL load failed. Falling back to backbone-only.")
            log_info(str(e).split("\n")[0])

            state2 = drop_keys_containing(state, [
                "expand_ii", "expand_ij",
                "fc_ii", "fc_ij",
                "fc_ii_bias", "fc_ij_bias",
                "output_ii", "output_ij",
            ])

            missing, unexpected = model.load_state_dict(state2, strict=False)
            log_info(f"[PRETRAIN] loaded BACKBONE only: {ckpt_path}")
            log_info(f"[PRETRAIN] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")

    p = next(model.parameters()).detach().float().cpu()
    log_info(f"[CHECK] first_param_abs_sum={p.abs().sum().item():.10f}")

    train_iterator = iter(train_data_loader)
    epoch = start_epoch
    model.train()

    for batch_idx in range(start_batch_idx, conf.datasets.run_steps):
        try:
            batch = next(train_iterator)
        except StopIteration:
            epoch += 1
            train_iterator = iter(train_data_loader)
            batch = next(train_iterator)

        errors = train_one_batch(
            conf, batch, model, optimizer, basis_dat_path, device, default_type
        )

        scheduler.step()

        if conf.ema_start_epoch > -1 and epoch > conf.ema_start_epoch:
            ema.update()

        if batch_idx % conf.datasets.train_batch_interval == 0:
            log_info(f"Train: Epoch {epoch} {batch_idx} HOMO MAE: {errors['homo_mae']:.8f}.")
            log_info(f"lr: {optimizer.param_groups[0]['lr']}.")

        if batch_idx % conf.datasets.validation_batch_interval == 0:
            log_info(f"Evaluating on epoch {epoch}")
            log_info("[VAL] computing HOMO MAE ...")

            if conf.ema_start_epoch > -1 and epoch > conf.ema_start_epoch:
                log_info("with ema")
                with ema.average_parameters():
                    val_homo_mae = validation_homo_mae_subset(
                        val_homo_loader, model, device, basis_dat_path, default_type
                    )
            else:
                val_homo_mae = validation_homo_mae_subset(
                    val_homo_loader, model, device, basis_dat_path, default_type
                )

            log_info(f"[VAL] subset HOMO MAE = {val_homo_mae:.8f}")

            if val_homo_mae < best_homo_val_result:
                best_homo_val_result = val_homo_mae
                final_val_homo_result = val_homo_mae
                final_train_result = errors["homo_mae"].item()

                save_checkpoint(
                    "results_best_homo.pt",
                    model,
                    optimizer,
                    scheduler,
                    ema,
                    epoch,
                    batch_idx,
                    best_homo_val_result,
                    final_train_result,
                    final_val_homo_result,
                    errors,
                )

            if batch_idx > 0 and batch_idx % 10000 == 0:
                save_checkpoint(
                    f"results_{batch_idx}.pt",
                    model,
                    optimizer,
                    scheduler,
                    ema,
                    epoch,
                    batch_idx,
                    best_homo_val_result,
                    final_train_result,
                    final_val_homo_result,
                    errors,
                )

            log_info(f"train HOMO MAE: {errors['homo_mae']:.8f}")
            log_info(f"val subset HOMO MAE: {val_homo_mae:.8f}")
            log_info("-------------------------")
            log_info(f"best val subset HOMO MAE so far: {best_homo_val_result:.8f}.")
            log_info(
                f"final train HOMO MAE so far: {final_train_result:.8f}, "
                f"final val HOMO MAE so far: {final_val_homo_result:.8f}."
            )
            log_info("=========================")


if __name__ == '__main__':
    main()
