#!/usr/bin/env python3
import numpy as np
import torch
from ase.db import connect
from torch_geometric.data import Data
from torch.utils.data import Dataset
from argparse import Namespace

# ----------------------------
# DZP conventions
# ----------------------------
convention_dict = {
    "pyscf_dzp": Namespace(
        # shell strings:
        #   s -> 1 AO
        #   p -> 3 AO (with orbital_idx_map ordering)
        #   d -> 5 AO (spherical)
        #
        # Here:
        #   H: ss            -> 2 AO
        #   C: ssssppd       -> 4*1 + 2*3 + 5 = 15 AO
        #   N/O/F: sssspp    -> 4*1 + 2*3     = 10 AO   (your [4s,2p])
        atom_to_orbitals_map={
            1: "ss",
            6: "ssssppd",
            7: "sssspp",
            8: "sssspp",
            9: "sssspp",
        },
        orbital_idx_map={"s": [0], "p": [1, 2, 0], "d": [0, 1, 2, 3, 4]},
        orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
        # shell order indices (shells, not AO functions)
        # For each atom: list indices over shells in its atom_to_orbitals_map string.
        # H: ss -> 2 shells -> [0,1]
        # C: ssssppd -> 7 shells -> [0..6]
        # N/O/F: sssspp -> 6 shells -> [0..5]
        orbital_order_map={
            1: [0, 1],
            6: [0, 1, 2, 3, 4, 5, 6],
            7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5],
            9: [0, 1, 2, 3, 4, 5],
        },
    )
}


def matrix_transform(matrices: np.ndarray, atoms: np.ndarray, convention: str = "pyscf_dzp"):
    """
    Reorder AO indices within each atom (shell order, p ordering, etc.) and apply sign flips.
    matrices: (..., nao, nao)
    atoms: (n_atoms,) atomic numbers
    """
    conv = convention_dict[convention]
    atoms = np.asarray(atoms, dtype=np.int64)

    orbitals = ""
    orbitals_order = []
    for a in atoms:
        z = int(a)
        if z not in conv.atom_to_orbitals_map:
            raise KeyError(f"Atom Z={z} not supported in convention {convention}")
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[z]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[z]]

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        map_sign = conv.orbital_sign_map[orb]
        transform_indices.append(np.array(map_idx, dtype=np.int32) + offset)
        transform_signs.append(np.array(map_sign, dtype=np.float64))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = np.concatenate(transform_indices).astype(np.int32)
    transform_signs = np.concatenate(transform_signs).astype(np.float64)

    out = matrices[..., transform_indices, :]
    out = out[..., :, transform_indices]
    out = out * transform_signs[:, None]
    out = out * transform_signs[None, :]
    return out

class EvaluateDataset(Dataset):
    """
    DZP-style dataset with per-element AO padding.
    Now supports H/C/N/O/F.

    IMPORTANT:
      - This code assumes C has 15 AO (ssssppd, spherical d).
      - N/O/F have 10 AO (sssspp) as you showed ([4s,2p]).
      - If your DB's stored H/S were produced with different basis ordering/count,
        the nao mismatch check will tell you immediately.
    """

    # pad size = max AO among supported elements
    full_orbitals = 15

    class _Subset(Dataset):
        def __init__(self, parent, indices):
            self.parent = parent
            self.indices = np.asarray(indices, dtype=np.int64)

        def __len__(self):
            return int(self.indices.shape[0])

        def __getitem__(self, i):
            return self.parent[int(self.indices[int(i)])]

    def __init__(
        self,
        db_path: str,
        convention: str = "pyscf_dzp",
        max_items: int = None,
        seed: int = 0,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        include_overlap: bool = True,
    ):
        self.db_path = db_path
        self.db = connect(db_path)
        self.convention = convention
        self.include_overlap = include_overlap

        ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64)
        if np.any(ratios < 0):
            raise ValueError(f"ratios must be >= 0, got {ratios}")
        s = float(ratios.sum())
        if not np.isclose(s, 1.0):
            raise ValueError(f"train/val/test ratios must sum to 1.0, got {s}")

        self.ids = [int(row.id) for row in self.db.select()]
        if len(self.ids) == 0:
            raise RuntimeError(f"No rows found in DB: {db_path}")

        rng = np.random.RandomState(seed)
        if max_items is not None:
            max_items = int(max_items)
            if max_items <= 0:
                raise ValueError(f"max_items must be positive, got {max_items}")
            if max_items < len(self.ids):
                self.ids = list(rng.choice(self.ids, size=max_items, replace=False))

        # orbital masks into padded full_orbitals x full_orbitals
        self.orbital_mask = {
            1: torch.arange(2, dtype=torch.long),                  # H: 2
            6: torch.arange(15, dtype=torch.long),                 # C: 15
            7: torch.arange(10, dtype=torch.long),                 # N: 10
            8: torch.arange(10, dtype=torch.long),                 # O: 10
            9: torch.arange(10, dtype=torch.long),                 # F: 10
        }

        # split (pure random permutation on ids list)
        N = len(self.ids)
        perm = rng.permutation(N)
        n_train = int(np.floor(N * train_ratio))
        n_val = int(np.floor(N * val_ratio))
        self.train_mask = perm[:n_train].astype(np.int64)
        self.val_mask = perm[n_train:n_train + n_val].astype(np.int64)
        self.test_mask = perm[n_train + n_val:].astype(np.int64)

        first = self.db.get(id=int(self.ids[0]))
        Z0 = first.toatoms().numbers.astype(np.int64)
        self._cache = {
            "n_atoms": int(len(Z0)),
            "edge_index_full": self._edge_index_full(int(len(Z0))),
        }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple, np.ndarray, torch.Tensor)):
            if isinstance(idx, torch.Tensor):
                idx = idx.detach().cpu().numpy()
            return EvaluateDataset._Subset(self, idx)

        idx = int(idx)
        row = self.db.get(id=int(self.ids[idx]))
        atoms_ase = row.toatoms()
        Z = atoms_ase.numbers.astype(np.int64)
        pos = atoms_ase.positions.astype(np.float64)  # Ang

        def ao_count(z: int) -> int:
            if z == 1:
                return 2
            if z == 6:
                return 15
            if z in (7, 8, 9):
                return 10
            raise KeyError(f"Unsupported atom Z={z} for DZP dataset. Add to ao_count/orbital_mask.")

        expected_nao = int(sum(ao_count(int(z)) for z in Z))

        if self._cache["n_atoms"] == len(Z):
            edge_index_full = self._cache["edge_index_full"]

        return Data(
                pos=torch.tensor(pos, dtype=torch.float64),
                atoms=torch.tensor(Z, dtype=torch.int64).view(-1, 1),
                edge_index_full=edge_index_full,
            )

    @staticmethod
    def _edge_index_full(n_atoms: int) -> torch.Tensor:
        edges = []
        for src in range(n_atoms):
            for dst in range(n_atoms):
                if src != dst:
                    edges.append([dst, src])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

class NapDZPOfficialDBDataset(Dataset):
    """
    DZP-style dataset with per-element AO padding.
    Now supports H/C/N/O/F.

    IMPORTANT:
      - This code assumes C has 15 AO (ssssppd, spherical d).
      - N/O/F have 10 AO (sssspp) as you showed ([4s,2p]).
      - If your DB's stored H/S were produced with different basis ordering/count,
        the nao mismatch check will tell you immediately.
    """

    # pad size = max AO among supported elements
    full_orbitals = 15

    class _Subset(Dataset):
        def __init__(self, parent, indices):
            self.parent = parent
            self.indices = np.asarray(indices, dtype=np.int64)

        def __len__(self):
            return int(self.indices.shape[0])

        def __getitem__(self, i):
            return self.parent[int(self.indices[int(i)])]

    def __init__(
        self,
        db_path: str,
        convention: str = "pyscf_dzp",
        max_items: int = None,
        seed: int = 0,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        include_overlap: bool = True,
    ):
        self.db_path = db_path
        self.db = connect(db_path)
        self.convention = convention
        self.include_overlap = include_overlap

        ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64)
        if np.any(ratios < 0):
            raise ValueError(f"ratios must be >= 0, got {ratios}")
        s = float(ratios.sum())
        if not np.isclose(s, 1.0):
            raise ValueError(f"train/val/test ratios must sum to 1.0, got {s}")

        self.ids = [int(row.id) for row in self.db.select()]
        if len(self.ids) == 0:
            raise RuntimeError(f"No rows found in DB: {db_path}")

        rng = np.random.RandomState(seed)

        if max_items is not None:
            max_items = int(max_items)
            if max_items <= 0:
                raise ValueError(f"max_items must be positive, got {max_items}")
            if max_items < len(self.ids):
                self.ids = list(rng.choice(self.ids, size=max_items, replace=False))

        # orbital masks into padded full_orbitals x full_orbitals
        self.orbital_mask = {
            1: torch.arange(2, dtype=torch.long),                  # H: 2
            6: torch.arange(15, dtype=torch.long),                 # C: 15
            7: torch.arange(10, dtype=torch.long),                 # N: 10
            8: torch.arange(10, dtype=torch.long),                 # O: 10
            9: torch.arange(10, dtype=torch.long),                 # F: 10
        }

        # split (pure random permutation on ids list)
        N = len(self.ids)
        perm = rng.permutation(N)
        n_train = int(np.floor(N * train_ratio))
        n_val = int(np.floor(N * val_ratio))
        self.train_mask = perm[:n_train].astype(np.int64)
        self.val_mask = perm[n_train:n_train + n_val].astype(np.int64)
        self.test_mask = perm[n_train + n_val:].astype(np.int64)

        first = self.db.get(id=int(self.ids[0]))
        Z0 = first.toatoms().numbers.astype(np.int64)
        self._cache = {
            "n_atoms": int(len(Z0)),
            "edge_index_full": self._edge_index_full(int(len(Z0))),
        }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple, np.ndarray, torch.Tensor)):
            if isinstance(idx, torch.Tensor):
                idx = idx.detach().cpu().numpy()
            return NapDZPOfficialDBDataset._Subset(self, idx)

        idx = int(idx)
        row = self.db.get(id=int(self.ids[idx]))
        atoms_ase = row.toatoms()
        Z = atoms_ase.numbers.astype(np.int64)
        pos = atoms_ase.positions.astype(np.float64)  # Ang

        H = np.asarray(row.data["hamiltonian"])
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise RuntimeError(f"Bad H shape at idx={idx}, id={self.ids[idx]}: {H.shape}")
        nao = int(H.shape[0])

        S = None
        if self.include_overlap:
            S = np.asarray(row.data["overlap"])
            if S.ndim != 2 or S.shape[0] != S.shape[1]:
                raise RuntimeError(f"Bad S shape at idx={idx}, id={self.ids[idx]}: {S.shape}")
            if S.shape[0] != H.shape[0]:
                raise RuntimeError(f"H/S nao mismatch at idx={idx}: H={H.shape}, S={S.shape}")

        def ao_count(z: int) -> int:
            if z == 1:
                return 2
            if z == 6:
                return 15
            if z in (7, 8, 9):
                return 10
            raise KeyError(f"Unsupported atom Z={z} for DZP dataset. Add to ao_count/orbital_mask.")

        expected_nao = int(sum(ao_count(int(z)) for z in Z))
        if nao != expected_nao:
            # print unique Z to help
            uniq = {int(x): int((Z == x).sum()) for x in np.unique(Z)}
            raise RuntimeError(
                f"[nao mismatch vs DZP convention]\n"
                f"  nao in DB      = {nao}\n"
                f"  expected_nao   = {expected_nao}\n"
                f"  Z counts       = {uniq}\n"
                f"  convention     = H(2) C(15) N/O/F(10)\n"
                f"  => Your stored H/S are not in this AO-count layout.\n"
                f"     Most likely: your C basis count differs (10 vs 15), or d is cartesian (6) not spherical (5),\n"
                f"     or the DB is not built with this dzp_q-chem.dat.\n"
            )

        # matrix_transform to convention
        H = matrix_transform(H, Z, convention=self.convention)
        H = torch.tensor(H, dtype=torch.float64)
        dH, ndH, dH_mask, ndH_mask, edge_index_full = self.cut_matrix(H, Z)

        if self._cache["n_atoms"] == len(Z):
            edge_index_full = self._cache["edge_index_full"]

        if not self.include_overlap:
            return Data(
                pos=torch.tensor(pos, dtype=torch.float64),
                atoms=torch.tensor(Z, dtype=torch.int64).view(-1, 1),
                diagonal_hamiltonian=dH,
                non_diagonal_hamiltonian=ndH,
                diagonal_hamiltonian_mask=dH_mask,
                non_diagonal_hamiltonian_mask=ndH_mask,
                edge_index_full=edge_index_full,
            )

        S = matrix_transform(S, Z, convention=self.convention)
        S = torch.tensor(S, dtype=torch.float64)
        dS, ndS, dS_mask, ndS_mask, _ = self.cut_matrix(S, Z)

        return Data(
            pos=torch.tensor(pos, dtype=torch.float64),
            atoms=torch.tensor(Z, dtype=torch.int64).view(-1, 1),

            diagonal_hamiltonian=dH,
            non_diagonal_hamiltonian=ndH,
            diagonal_hamiltonian_mask=dH_mask,
            non_diagonal_hamiltonian_mask=ndH_mask,

            diagonal_overlap=dS,
            non_diagonal_overlap=ndS,
            diagonal_overlap_mask=dS_mask,
            non_diagonal_overlap_mask=ndS_mask,

            edge_index_full=edge_index_full,
        )

    @staticmethod
    def _edge_index_full(n_atoms: int) -> torch.Tensor:
        edges = []
        for src in range(n_atoms):
            for dst in range(n_atoms):
                if src != dst:
                    edges.append([dst, src])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def cut_matrix(self, matrix: torch.Tensor, atoms: np.ndarray):
        """
        Cut full (nao,nao) matrix into padded blocks (full_orbitals x full_orbitals).
        """
        atoms = torch.tensor(atoms, dtype=torch.long)

        all_diag, all_off = [], []
        all_diag_mask, all_off_mask = [], []
        col_idx = 0
        edge_index_full = []

        for idx_i, atom_i in enumerate(atoms):  # src
            row_idx = 0
            zi = int(atom_i.item())
            if zi not in self.orbital_mask:
                raise KeyError(f"orbital_mask missing for Z={zi}")
            mask_i = self.orbital_mask[zi]

            for idx_j, atom_j in enumerate(atoms):  # dst
                if idx_i != idx_j:
                    edge_index_full.append([idx_j, idx_i])

                zj = int(atom_j.item())
                if zj not in self.orbital_mask:
                    raise KeyError(f"orbital_mask missing for Z={zj}")
                mask_j = self.orbital_mask[zj]

                block = torch.zeros(self.full_orbitals, self.full_orbitals, dtype=torch.float64)
                block_mask = torch.zeros(self.full_orbitals, self.full_orbitals, dtype=torch.float64)

                extracted = matrix[row_idx: row_idx + len(mask_j), col_idx: col_idx + len(mask_i)]

                tmp = block[mask_j]
                tmp[:, mask_i] = extracted
                block[mask_j] = tmp

                tmpm = block_mask[mask_j]
                tmpm[:, mask_i] = 1.0
                block_mask[mask_j] = tmpm

                if idx_i == idx_j:
                    all_diag.append(block)
                    all_diag_mask.append(block_mask)
                else:
                    all_off.append(block)
                    all_off_mask.append(block_mask)

                row_idx += len(mask_j)

            col_idx += len(mask_i)

        return (
            torch.stack(all_diag, dim=0),
            torch.stack(all_off, dim=0),
            torch.stack(all_diag_mask, dim=0),
            torch.stack(all_off_mask, dim=0),
            torch.tensor(edge_index_full, dtype=torch.long).t().contiguous(),
        )

