#!/usr/bin/env python
import numpy as np
from ase.io import read, write
from ase import Atoms
#import psi4
import torch
from torch import nn
from torch.autograd import grad
from pyscf import gto
from pyscf.gto.basis import parse_gaussian
import sys

class OverlapMatrix(nn.Module):
    """
    Calculates the overlap matrix for the inputted molecular system.
    This currently only works for one type of system, so this will
    have to be modified if you want to input multiple systems
    """
    def __init__(self, init_atoms, requires_derivative, coord='cartesian', order='psi4', basis=None, device='cuda'):
        """
        Parameters
        -----------
        init_atoms: ASE Atoms object
            Input a representative system geometry. This is used to initialize the basis.
        requires_derivative: bool
            Deprecated, but you can uncomment the code in forward if you want the Overlap gradient
        coord: str
            Whether the Overlap should be cartesian or spherical output. 
        order: str
            Different quantum chemistry codes output the overlap matrix following different orbital
            ordering conventions. You can have the code output in psi4, qchem or pyscf format
        device: str
            Whether the calculation should be run with PyTorch cpu or cuda
        """
        super(OverlapMatrix, self).__init__()
       
        self.basis_path = basis
        self.basis_num_func = {0.0: 1,
                1.0: 3,
                2.0: 6}

        self.device = torch.device(device)
        #Assemble the basis for the given system
        self._init_basis(init_atoms)

        #Determines whether output respect to positions is needed
        #Currently doesn't do anything, but 
        self.requires_derivative = requires_derivative
        
        self.pi = torch.tensor(np.pi)

        #Decides the ordering for a certain subset of integrals
        #For example, px, py and pz could be ordered differently depending on the QC code
        #These lists hard-code the ordering because it is an inherently arbitary ordering
        #If you need f functions (or higher order) you have to add the corresponding
        #lists to at least one selected dictionary defined below
        self.reorder_psi4 = {'ss_one_electron': None, 
                'sp_one_electron': [1, 0, 2], 
                'pp_one_electron': [4, 1, 7, 3, 0, 6, 5, 2, 8],
                'sd_one_electron': [2, 1, 4, 0, 3, 5], 
                'pd_one_electron': [8, 7, 10, 6, 9, 11, 2, 1, 4, 0, 3, 5, 14, 13, 16, 12, 15, 17],
                'dd_one_electron': [14, 13, 16, 12, 15, 17, 8, 7, 10, 6, 9, 11, 26, 25, 28, 24, 27, 29, 2, 1, 4, 0, 3, 5, 20, 19, 22, 18, 21, 23, 32, 31, 34, 30, 33, 35]}
        self.reorder_pyscf = {'ss_one_electron': None, 
                'sp_one_electron': [1, 0, 2], 
                'pp_one_electron': [4, 1, 5, 3, 0, 2, 7, 6, 8], 
                'sd_one_electron': [2, 1, 4, 0, 3, 5], 
                'pd_one_electron': [8, 7, 10, 6, 9, 11, 2, 1, 4, 0, 3, 5, 14, 13, 16, 12, 15, 17],
                'dd_one_electron': [14, 13, 16, 12, 15, 17, 8, 7, 10, 6, 9, 11, 26, 25, 28, 24, 27, 29, 2, 1, 4, 0, 3, 5, 20, 19, 22, 18, 21, 23, 32, 31, 34, 30, 33, 35]}
        self.reorder_qchem = {'ss_one_electron': None,
                'sp_one_electron': [1, 0, 2],
                'pp_one_electron': [4, 1, 5, 3, 0, 2, 7, 6, 8],
                'sd_one_electron': [2, 1, 4, 0, 3, 5],
                'pd_one_electron': [8, 7, 10, 6, 9, 11, 2, 1, 4, 0, 3, 5, 14, 13, 16, 12, 15, 17],
                'dd_one_electron': [14, 13, 16, 12, 15, 17, 8, 7, 10, 6, 9, 11, 26, 25, 28, 24, 27, 29, 2, 1, 4, 0, 3, 5, 20, 19, 22, 18, 21, 23, 32, 31, 34, 30, 33, 35]}

        self.coord = coord
        self.order = order
        if self.order == 'psi4':
            self.reorder = self.reorder_psi4
        elif self.order == 'pyscf': 
            self.reorder = self.reorder_pyscf
        elif self.order == 'qchem': 
            self.reorder = self.reorder_qchem
        else:
            print(f"WARNING: Ordering for {self.order} is not implemented")
            sys.exit()

    def _primitive_norm(self, l, e):
        tmp = l + 1.5
        g = (2.0 * e)**tmp
        return np.sqrt((2.0**l * g) / (np.pi * np.sqrt(np.pi) * self.df[2*int(l)]))
    
    def _init_basis(self, atoms):
        """
        Parameters
        -----------
        atoms: ASE atoms object
            Used for initial information given to Psi4
        """
        symbols = atoms.get_chemical_symbols()
        numbers = atoms.numbers
        pos = atoms.get_positions()
        mol = gto.Mole()
        mol.atom = zip(symbols, pos)
        gto_basis = {'C': gto.load(self.basis_path, 'C'), 'H': gto.load(self.basis_path, 'H')}
        mol.basis = gto_basis
        mol.build()

        self.df = np.zeros((500))
        self.df[0] = 1.0
        self.df[1] = 1.0
        self.df[2] = 1.0
        #If ever need more than l = 2 uncomment this and run in python
        for i in range(3, 100):
            self.df[i] = (i-1) * self.df[i-2]

        shell_index = []
        #Number of basis functions the shell corresponds to
        func_per_shell = []
        primitives = []
        for i in range(mol.nbas):
            shell_index.append(i)
            num_func = 2*mol._bas[i][1]+1
            if mol._bas[i][1] == 2: 
                num_func += 1
            primitives.append(mol._bas[i][2])
            func_per_shell.append(num_func)

        func_per_shell = np.asarray(func_per_shell)
        max_func_per_shell = np.max(func_per_shell)

        shell_index = np.asarray(shell_index)
        shell_to_func = -np.ones((len(shell_index), max_func_per_shell))
        #Max number of primitives for a given shell
        max_prim = np.asarray(primitives).max()
        #Primitive coefficients for each shell (which have been normalized by Psi4)
        self.coeffs = np.zeros((len(shell_index), max_prim))
        #Primitive exponent for each shell
        self.exps = np.zeros((len(shell_index), max_prim))
        #Angular momentum of each shell
        am_val = np.zeros((len(shell_index)))
        #Atomic center for a given basis function
        center_ids = np.zeros((len(shell_index)))
        func_number = 0
       
        og_basis = []
        for atom in range(mol.natm):
            symbol = mol.atom_symbol(atom)
            basis = mol._basis[symbol] 
            for b in basis:
                og_basis.append(b[1:])
        
        for i in range(mol.nbas):
            center_ids[i] = mol._bas[i][0]

            am_val[i] = mol._bas[i][1]

            exp = mol.bas_exp(i)
            coefs = og_basis[i]
            coefs_unnorm = []
            for f in coefs:
                coefs_unnorm.append(f[1])
            
            for e in range(len(exp)):
                self.exps[i, e] = exp[e]
                self.coeffs[i, e] = self._primitive_norm(am_val[i], exp[e]) * coefs_unnorm[e] 
            
            num_func = self.basis_num_func[am_val[i]]
            for f in range(num_func):
                shell_to_func[i, f] = func_number
                func_number += 1
        
        self.shell_func_indices = torch.from_numpy(shell_to_func.astype(int)).to(self.device)
        self.max_func_per_shell = np.max(np.asarray(func_per_shell))
        #Get pairs of shell indices for which an integral needs to be computed 
        shell_pairs = np.asarray(np.meshgrid(shell_index, shell_index)).T.reshape(-1, 2)
        shell_pairs.sort(axis=1)
        self.shell_pairs = torch.from_numpy(np.unique(shell_pairs, axis=0))

        #Max number of primitive functions spanning all shells
        prim_func_index = np.arange(0, max_prim)
        prim_pairs = np.asarray(np.meshgrid(prim_func_index, prim_func_index)).T.reshape(-1, 2)
        #For the second listed primitive function index, add max_prim. A little hard to explain 
        #why, but basically the second index for each pair in prim_pairs is the index for the shell
        #and we need it to be the index that corresponds to the global coeffs and exps
        prim_pairs[:, 1] += max_prim
        self.prim_pairs = torch.from_numpy(prim_pairs).to(self.device)
        self.center_ids = torch.from_numpy(center_ids).to(self.device)
        self.func_number = func_number
        #angular momentum value for shell
        self.am = torch.from_numpy(am_val).to(self.device)
        #Coefficients for basis set 
        self.coeffs = torch.from_numpy(self.coeffs).to(self.device)
        #Zeta exponents for basis set exponentials
        self.exps = torch.from_numpy(self.exps).to(self.device)

    def _setup_integral(self, inputs):
        """
        Parameters
        -----------
        inputs: dict
            Dictionary containing the system tensors
        """
        #Positions, convert Angstrom to Bohr
        self.R = inputs["R"] * 1.88973
        #If mask is not already defined
        if not hasattr(self, "mask"):
            #Due to the array indexing approach, we will be computing integrals between
            #basis functions with themselves. This will result in 0 distances, which
            #autograd can't handle. We mask these situations with the below tensor
            self.mask = torch.where(self.shell_pairs[:, 0] != self.shell_pairs[:, 1])[0]
            
            #Basis function indices associated with each shell
            self.coeffs = self.coeffs[self.shell_pairs[:, :], :]
            self.exps = self.exps[self.shell_pairs[:, :], :]
            self.center_ids = self.center_ids[self.shell_pairs[:, :]].long()
            #Pairs of coordinates between basis functions
            self.R_pairs = self.R[:, self.center_ids[:, :], :]
            #Offsets for computing the minimum image
            self.offsets = inputs["offsets"][:, self.center_ids[:, 0], self.center_ids[:, 1], :]
            #Cell vectors
            self.cell = inputs["cell"] * 1.88973
            #Needs to be modified for periodic boundaries, actually R**2 and not distances
            self.R2 = self.compute_distances()
            #Pairs of angular momentum values between shells
            self.am_pairs = self.am[self.shell_pairs[:, :]]
            #Basis functions that each shell corresponds to 
            self.func_indices = self.shell_func_indices[self.shell_pairs[:, :]]

            #Get coefficients for each pair of primitive basis functions
            self.coeffs = self.coeffs.reshape(self.coeffs.shape[0], self.coeffs.shape[1]*self.coeffs.shape[2])
            self.coeffs = self.coeffs[:, self.prim_pairs]
            #Get exponents for each pair of primitive basis functions
            self.exps = self.exps.reshape(self.exps.shape[0], self.exps.shape[1]*self.exps.shape[2])
            self.exps = self.exps[:, self.prim_pairs]
            #Repeat R and R2 so that they are the same shape as self.exps
            self.R_pairs = self.R_pairs[:, :, None]
            self.R_pairs = torch.repeat_interleave(self.R_pairs, self.exps.shape[1], axis=2)
            self.R2 = self.R2[:, :, None]
            self.R2 = torch.repeat_interleave(self.R2, self.exps.shape[1], axis=2)

            #Determine the required integrals for the highest angular momentum
            self.max_am = torch.max(self.am)
            self.integral_list = []
            self.integral_index = []
            if self.max_am == 0:
                self.required_integrals = ['ss_one_electron']
            if self.max_am == 1:
                self.required_integrals = ['ss_one_electron', 'sp_one_electron', 'pp_one_electron']
            if self.max_am == 2:
                self.required_integrals = ['ss_one_electron', 'sp_one_electron', 'pp_one_electron',
                        'sd_one_electron', 'pd_one_electron', 'dd_one_electron']

            #Initialize Overlap matrix (n_batch, n_basis_functions, n_basis_functions)
            self.S = torch.zeros((self.R.shape[0], self.func_number, self.func_number), device=self.device).float()
            #Get cartesian to spherical transform matrix
            if self.coord == 'spherical':
                self.c2sph = self.get_spherical_transform()
        else:
            #Assemble the positions and coordinates pairs, since the coeffs and exps pairs are already defined
            self.R_pairs = self.R[:, self.center_ids[:, :], :]
            self.offsets = inputs["offsets"][:, self.center_ids[:, 0], self.center_ids[:, 1], :]
            #Cell vectors
            self.cell = inputs["cell"] * 1.88973
            self.R2 = self.compute_distances()
            self.R_pairs = self.R_pairs[:, :, None]
            self.R_pairs = torch.repeat_interleave(self.R_pairs, self.exps.shape[1], axis=2)
            self.R2 = self.R2[:, :, None]
            self.R2 = torch.repeat_interleave(self.R2, self.exps.shape[1], axis=2)
            self.S = torch.zeros((self.R.shape[0], self.func_number, self.func_number), device=self.device).float()

    def compute_distances(self):
        """
        Really computing r**2 here

        Returns
        ----------
        tmp_R2: tensor (n_batch, n_shell_pairs)
            r**2 between each pair of Gaussian shells
        """
        dist_vec = self.R_pairs[:, :, 0] - self.R_pairs[:, :, 1]
        offsets = self.offsets.bmm(self.cell)
        dist_vec = dist_vec - offsets
        R2 = torch.pow(dist_vec, 2).sum(axis=-1)
        tmp_R2 = torch.zeros_like(R2)
        #Mask out zero values due to computing r**2 between the same center
        tmp_R2[:, self.mask] = R2[:, self.mask]
        self.R_pairs[:, :, 1] += offsets
        return tmp_R2

    def get_spherical_transform(self):
        """
        Get cartesian to spherical transform for integral

        Returns
        ---------
        tensor
            Tensor defining the cartesian to spherical transform
        """
        if self.order == 'pyscf' or self.order == 'qchem':
            #S and p orbitals don't require converting
            c2sph = [torch.tensor([1.0])]
            
            p_conv = torch.tensor(
                    [[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]]
                    )
            c2sph.append(p_conv)

            #See https://en.wikipedia.org/wiki/Solid_harmonics for d cartesian to spherical transform
            d_conv = np.zeros((6, 5))
            d_conv[:, 0] = [0, np.sqrt(3), 0, 0, 0, 0]
            d_conv[:, 1] = [0, 0, 0, 0, np.sqrt(3), 0]
            d_conv[:, 2] = [-0.5, 0, 0, -0.5, 0, 1.0]
            d_conv[:, 3] = [0, 0, np.sqrt(3), 0, 0, 0]
            d_conv[:, 4] = [np.sqrt(3)/2, 0, 0, -np.sqrt(3)/2, 0, 0]
            c2sph.append(torch.from_numpy(d_conv))

            c2sph_matrix = []
            #Fill c2sph_matrix based on angular momentum orbital values
            for am in self.am:
                am_value = am.long().item()
                c2sph_matrix.append(c2sph[am_value])

        elif self.order == 'psi4':
            #Same process as above, although some things are ordered differently
            c2sph = [torch.tensor(1.0)]

            p_conv = torch.tensor(
                    [[0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0]]
                    )
            c2sph.append(p_conv)

            d_conv = np.zeros((6, 5))
            d_conv[:, 0] = [-0.5, 0, 0, -0.5, 0, 1.0]
            d_conv[:, 1] = [0, 0, np.sqrt(3), 0, 0, 0]
            d_conv[:, 2] = [0, 0, 0, 0, np.sqrt(3), 0]
            d_conv[:, 3] = [np.sqrt(3)/2, 0, 0, -np.sqrt(3)/2, 0, 0]
            d_conv[:, 4] = [0, np.sqrt(3), 0, 0, 0, 0]
            c2sph.append(torch.from_numpy(d_conv))

            c2sph_matrix = []
            for am in self.am:
                am_value = am.detach().cpu().numpy().astype(int)
                c2sph_matrix.append(c2sph[am_value])

        return torch.block_diag(*c2sph_matrix)

    def get_pair_values(self, inds):
        """
        Get necessary values for pair of Gaussian shells

        Parameters
        -----------
        inds: tensor
            indices corresponding to each pair

        Returns
        -----------
        coeffs: tensor
            Basis function coefficients for each pair
        exps: tensor
            Basis function exponents for each pair
        R: tensor
            position tensor for each pair
        R2: tensor
            r**2 value for each pair
        func_indices: tensor
            The basis function indices for each pair
        ams: tensor
            The angular momentum for each pair of Gaussian shells
        """
        coeffs = self.coeffs[inds]
        exps = self.exps[inds]
        R = self.R_pairs[:, inds]
        R2 = self.R2[:, inds]
        func_indices = self.func_indices[inds]
        ams = self.am_pairs[inds]
        return coeffs, exps, R, R2, func_indices, ams

    def get_add_index(self, func_indices):
        """
        Gets necessary indices for adding each integral to the full overlap matrix

        Parameters
        -----------
        func_indices: tensor
            basis function indices for each shell

        Returns
        -----------
        func_indices[inds_val]: tensor
            Each pair of valid basis function (where none of the basis function pairs are placeholder -1)
        """
        func_pairs_indices = torch.arange(0, self.max_func_per_shell)
        func_pairs = torch.meshgrid(func_pairs_indices, func_pairs_indices, indexing='ij')
        func_pairs = torch.stack(func_pairs)
        func_pairs = torch.transpose(func_pairs, 0, -1).reshape(-1, 2)
        func_pairs[:, 1] += self.max_func_per_shell

        func_indices = func_indices.reshape(func_indices.shape[0], func_indices.shape[1]*func_indices.shape[2])
        func_indices = func_indices[:, func_pairs]
        inds_val = torch.where(torch.logical_and(func_indices[:, :, 0] != -1, func_indices[:, :, 1] != -1))
        return func_indices[inds_val]

    def get_s00_integral(self, coeffs, exps, exp_sum_inv, R2):
        """
        Calculate standard 3D Gaussian integral

        Parameters
        -----------
        coeffs: tensor
            Gaussian coefficients
        exps: tensor
            Gaussian exponents
        exp_sum_inv: tensor
            Inverse sum of Gaussian exponents 
        R2: tensor
            r**2 value for the integral

        Returns
        -----------
        integ: tensor
            integral of each primitive Gaussian
        """
        coeff = torch.prod(coeffs, axis=-1)
        alpha = torch.prod(exps, axis=-1)

        pref = (self.pi * exp_sum_inv)**(3/2)
        integ = pref * torch.exp(-alpha * exp_sum_inv * R2)
        integ *= coeff
        return integ

    def obara_saika_recursion(self, PA, PB, exp_sum, am_i, am_j, cart_fac):
        """
        Performs Obara-Saika recursion for an arbitary integral 

        Parameters
        -----------
        PA: tensor
            P coordinate from A to P
        PB: tensor
            P coordinate from B to P
        exp_sum: tensor
            Same thing as exp_sum_inv
        am_i: tensor
            angular momentum value of the first shell
        am_j: tensor
            angular momentum value of the second shell
        cart_fac: tensor
            Prefactor for each shell pair that gets modified by the recursion algorithm

        Returns
        -----------
        tensor
            The modified cart_fac
        """
        if am_i <= 0.0 and am_j <= 0.0:
            return cart_fac

        if am_j > am_i:
            return (PB * self.obara_saika_recursion(PA, PB, exp_sum, am_i, am_j-1, cart_fac)) + \
                0.5 * exp_sum * (am_i * self.obara_saika_recursion(PA, PB, exp_sum, am_i-1, am_j-1, cart_fac) + \
                (am_j-1) * self.obara_saika_recursion(PA, PB, exp_sum, am_i, am_j-2, cart_fac))
        else:
            return (PA * self.obara_saika_recursion(PA, PB, exp_sum, am_i-1, am_j, cart_fac)) + \
                0.5 * exp_sum * ((am_i-1) * self.obara_saika_recursion(PA, PB, exp_sum, am_i-2, am_j, cart_fac) + \
                (am_j) * self.obara_saika_recursion(PA, PB, exp_sum, am_i-1, am_j-1, cart_fac))

    def run_obara_saika(self, am_ints, P_coords, exps, exp_sum_inv, s00_int):
        """ 
        Not actually used anymore
        """
        integral_list = []
        for integral in am_ints:
            fac = torch.ones_like(s00_int)
            for cart in range(3):
                a_vals = integral[:, cart]
                cart_fac = torch.ones_like(s00_int)
                cart_fac = self.obara_saika_recursion(P_coords[cart][0], P_coords[cart][1], exp_sum_inv, a_vals[0], a_vals[1], cart_fac)
                fac *= cart_fac
            integral_add = s00_int * fac
            integral_add = integral_add.sum(axis=-1)
            integral_list.append(integral_add)
        
        return [i.float() for i in integral_list]

    def get_P_coordinate(self, positions, exps, exp_sum_inv):
        """
        The P coordinates are essentially weighted coordinates between the centers of the two Gaussians
        The weighting is done by the exponent values

        Parameters
        -----------
        positions: tensor
            Positions for the two Gaussian shells
        exps: tensor
            Gaussian exponents
        exp_sum_inv: tensor
            1/(exp_1 + exp_2)

        Returns
        -----------
        list
            Contains PXA and PXB for each coordinate for each shell
        """
        exp_sum = exps.sum(axis=-1)
        exp_sum_inv = torch.zeros_like(exp_sum)
        exp_sum_inv[exp_sum != 0] = 1/exp_sum[exp_sum != 0]
       
        P_x = (exps[:, :, :, 0] * positions[:, :, :, 0, 0] + exps[:, :, :, 1] * positions[:, :, :, 1, 0]) * exp_sum_inv
        P_y = (exps[:, :, :, 0] * positions[:, :, :, 0, 1] + exps[:, :, :, 1] * positions[:, :, :, 1, 1]) * exp_sum_inv
        P_z = (exps[:, :, :, 0] * positions[:, :, :, 0, 2] + exps[:, :, :, 1] * positions[:, :, :, 1, 2]) * exp_sum_inv

        PXA = (P_x - positions[:, :, :, 0, 0])
        PXB = (P_x - positions[:, :, :, 1, 0])

        PYA = (P_y - positions[:, :, :, 0, 1])
        PYB = (P_y - positions[:, :, :, 1, 1])

        PZA = (P_z - positions[:, :, :, 0, 2])
        PZB = (P_z - positions[:, :, :, 1, 2])
        return [[PXA, PXB], [PYA, PYB], [PZA, PZB]]

    def ss_one_electron(self, reorder):
        """
        Parameters
        -----------
        reorder: None
            nothing to reorder in this case
        """
        #Find where each Gaussian shell has 0 angular momentum
        s_s = torch.where(torch.logical_and(self.am_pairs[:, 0] == 0, self.am_pairs[:, 1] == 0))[0]
        coeffs, exps, positions, R2, func_indices, am = self.get_pair_values(s_s)

        #Expand dimension on zeroth axis for broadcasting
        coeffs = coeffs[None]
        exps = exps[None]
        #Often need 1/(exp_1 + exp_2), so compute here
        exp_sum = exps.sum(axis=-1)
        exp_sum_inv = torch.zeros_like(exp_sum)
        exp_sum_inv[exp_sum != 0] = 1/exp_sum[exp_sum != 0]

        #3D Gaussian integral
        integ = self.get_s00_integral(coeffs, exps, exp_sum_inv, R2)
        #Sum along primitive integrals
        integ = integ.sum(axis=-1)
        integ = integ.float()
        #Get indices for adding integral values to overlap matrix
        add_S_index = self.get_add_index(func_indices)
        
        #Tricky: add integrals to full overlap matrix
        #arange corresponding to the batch shape
        index_batch = torch.arange(integ.shape[0], device=self.device)
        #repeat index_batch based on number of integral indices for one batch
        index_batch = torch.repeat_interleave(index_batch, add_S_index.shape[0], dim=0)
        #Repeat integral indices for one batch by the total number of batches
        add_S_index_1 = torch.tile(add_S_index[:, 0], (integ.shape[0],))
        add_S_index_2 = torch.tile(add_S_index[:, 1], (integ.shape[0],))
        integ = integ.flatten()
        
        #Form temporary overlap matrix
        S_tmp = torch.zeros_like(self.S)
        #index_put_ places the integral indices into S_tmp using the corresponding index set
        S_tmp.index_put_((index_batch, add_S_index_1, add_S_index_2), integ)

        #Add the symmetric off-diagonal values to self.S. Avoid double-counting diagonal values
        S_diag = S_tmp[:, torch.eye(S_tmp.shape[1]).bool()]
        S_diag = torch.diag_embed(S_diag)
        self.S += S_tmp + S_tmp.transpose(1,2) - S_diag

    def sp_one_electron(self, reorder):
        """
        Parameters
        -----------
        reorder: list
            reorder the integrals according to reorder
        """
        #Find where angular momentum pair is 0:1 and 1:0 (both directions)
        s_p = torch.where(torch.logical_or(
            torch.logical_and(self.am_pairs[:, 0] == 0, self.am_pairs[:, 1] == 1),
            torch.logical_and(self.am_pairs[:, 0] == 1, self.am_pairs[:, 1] == 0))
            )[0]
        
        #Get p_orbitals
        p_orbitals = self.shell_func_indices[self.am == 1]
        coeffs, exps, positions, R2, func_indices, am = self.get_pair_values(s_p)
        coeffs = coeffs[None]
        exps = exps[None]

        exp_sum = exps.sum(axis=-1)
        exp_sum_inv = torch.zeros_like(exp_sum)
        exp_sum_inv[exp_sum != 0] = 1/exp_sum[exp_sum != 0]

        #3D Gaussian integral
        integ = self.get_s00_integral(coeffs, exps, exp_sum_inv, R2)
        #See Helgaker Modern Electronic-Structure Theory. P_coords is needed for higher AM values than 0
        P_coords = self.get_P_coordinate(positions, exps, exp_sum_inv)
        
        am_pairs = am[:, None]
        am_pairs = torch.repeat_interleave(am_pairs, coeffs.shape[2], axis=1)
        am_pairs = am_pairs[None]
        am_pairs = torch.repeat_interleave(am_pairs, integ.shape[0], axis=0)
        
        #Sometimes the integral pairs are listed as AM 0:1 and sometimes they are 1:0. We can 
        #take care of both cases with this where statement
        PX = torch.where(am_pairs[:, :, :, 0] == 1, P_coords[0][0], P_coords[0][1])
        PY = torch.where(am_pairs[:, :, :, 0] == 1, P_coords[1][0], P_coords[1][1])
        PZ = torch.where(am_pairs[:, :, :, 0] == 1, P_coords[2][0], P_coords[2][1])

        #The _alt values take care of the case where the shell pair is 0:1
        PX_alt = torch.where(am_pairs[:, :, :, 0] == 0, P_coords[0][0], P_coords[0][1])
        PY_alt = torch.where(am_pairs[:, :, :, 0] == 0, P_coords[1][0], P_coords[1][1])
        PZ_alt = torch.where(am_pairs[:, :, :, 0] == 0, P_coords[2][0], P_coords[2][1])

        #Form a tensor list of each angular momentum pair that needs to be calculated
        a_max = torch.max(am_pairs)
        am_values = torch.arange(a_max+1)
        am_grid = torch.meshgrid(am_values, am_values, am_values, indexing='ij')
        am_grid = torch.stack(am_grid)
        am_grid = torch.transpose(am_grid, 0, 3).reshape(-1, 3)
        p_orbs = am_grid[torch.sum(am_grid, axis=1) == 1]
        s_orbs = am_grid[torch.sum(am_grid, axis=1) == 0]

        am_p = torch.tile(p_orbs, (s_orbs.shape[0], 1))
        am_s = torch.repeat_interleave(s_orbs, p_orbs.shape[0], axis=0)
        am_ints = torch.hstack((am_p, am_s))
        am_ints = am_ints.reshape(am_ints.shape[0], 2, -1)
       
        #Combine each cartesian component together
        P_1 = torch.cat((PY, PX, PZ), dim=1)
        P_2 = torch.cat((PY_alt, PX_alt, PZ_alt), dim=1)
        #The '3' tiles in the components below correspond to the 3 cartesian components
        exps = torch.tile(exps, (1,3,1,1))
        exp_sum_inv = torch.tile(exp_sum_inv, (1,3,1))
        exp_sum_inv = torch.repeat_interleave(exp_sum_inv, P_1.shape[0], dim=0)
        integ = torch.tile(integ, (1,3,1))
        
        #Prefactor for multiplying the 3D Gaussian integrals
        fac = torch.ones_like(integ)
        #Run OS algorithm with angular momentum pair 1 and 0
        am_1, am_2 = torch.tensor([1.0], device=self.device), torch.tensor([0.0], device=self.device)
        fac = self.obara_saika_recursion(P_1, P_2, exp_sum_inv, am_1, am_2, fac)
        #Multiply each integral with the corresponding prefactor
        integral_add = integ * fac
        integral_add = integral_add.sum(axis=-1)

        #Assemble combined list of the integrals. The list corresponds to sp_y, sp_x and sp_z
        num_int = int(integral_add.shape[1]/3)
        integral_list = [integral_add[:, 0:num_int].float(), integral_add[:, num_int:num_int*2].float(), integral_add[:, num_int*2:num_int*3].float()]
        integral_list = [integral_list[i] for i in reorder]
        add_S_index = self.get_add_index(func_indices)
        
        x_orbs_p = p_orbitals[:, 0]
        y_orbs_p = p_orbitals[:, 1]
        z_orbs_p = p_orbitals[:, 2]
        
        #Reorder add_S_index arbitarily
        add_S_ord = torch.zeros_like(add_S_index)
        orb_pairs = []
        for p_orb in [x_orbs_p, y_orbs_p, z_orbs_p]:
            p_loc = torch.isin(add_S_index, p_orb)
            orb_pair = torch.where(p_loc)[0]
            orb_pairs.append(orb_pair)

        start = 0
        #Fill add_S_ord according to orb_pairs
        for orb in orb_pairs:
            add_S_ord[start:start+len(orb), :] = add_S_index[orb]
            start += len(orb)

        #See above for the process to add the integrals across batches
        integral_list = torch.cat(integral_list, dim=1)
        index_batch = torch.arange(integral_list.shape[0], device=self.device)
        index_batch = torch.repeat_interleave(index_batch, add_S_ord.shape[0], dim=0)
        add_S_index_1 = torch.tile(add_S_ord[:, 0], (integral_list.shape[0],))
        add_S_index_2 = torch.tile(add_S_ord[:, 1], (integral_list.shape[0],))
        integral_list = integral_list.flatten()

        S_tmp = torch.zeros_like(self.S)
        S_tmp.index_put_((index_batch, add_S_index_1, add_S_index_2), integral_list)

        S_diag = S_tmp[:, torch.eye(S_tmp.shape[1]).bool()]
        S_diag = torch.diag_embed(S_diag)
        self.S += S_tmp + S_tmp.transpose(1,2) - S_diag

    def pp_one_electron(self, reorder):
        """
        Parameters
        -----------
        reorder: list
            reorder the integrals according to reorder
        """
        #Most of the code is a repeat from above, so I'll only comment in a few places
        p_p = torch.where(torch.logical_and(self.am_pairs[:, 0] == 1, self.am_pairs[:, 1] == 1))[0]
        
        p_orbitals = self.shell_func_indices[self.am == 1]
        coeffs, exps, positions, R2, func_indices, am = self.get_pair_values(p_p)
        coeffs = coeffs[None]
        exps = exps[None]

        exp_sum = exps.sum(axis=-1)
        exp_sum_inv = torch.zeros_like(exp_sum)
        exp_sum_inv[exp_sum != 0] = 1/exp_sum[exp_sum != 0]

        integ = self.get_s00_integral(coeffs, exps, exp_sum_inv, R2)
        P_coords = self.get_P_coordinate(positions, exps, exp_sum_inv)

        am_pairs = am[:, None]
        am_pairs = torch.repeat_interleave(am_pairs, coeffs.shape[1], axis=1)

        a_max = torch.max(am_pairs)
        am_values = torch.arange(a_max+1)
        am_grid = torch.meshgrid(am_values, am_values, am_values, indexing='ij')
        am_grid = torch.stack(am_grid)
        am_grid = torch.transpose(am_grid, 0, 3).reshape(-1, 3)
        
        p_orbs = am_grid[torch.sum(am_grid, axis=1) == 1]

        am_p1 = torch.tile(p_orbs, (p_orbs.shape[0], 1))
        am_p2 = torch.repeat_interleave(p_orbs, p_orbs.shape[0], axis=0)
        am_ints = torch.hstack((am_p1, am_p2))
        am_ints = am_ints.reshape(am_ints.shape[0], 2, -1)
      
        #Reassembling the integral combination gets more challenging with more integrals
        #Assemble a dictionary of the integrals for each component
        coord_dict = {0: 'x', 1: 'y', 2: 'z'}
        integral_dict = {'x': {}, 'y': {}, 'z': {}}
        P_1 = torch.cat((P_coords[1][0], P_coords[0][0], P_coords[2][0]), dim=1)
        P_2 = torch.cat((P_coords[1][1], P_coords[0][1], P_coords[2][1]), dim=1)
        exps = torch.tile(exps, (1,3,1,1))
        exp_sum_inv = torch.tile(exp_sum_inv, (1,3,1))
        exp_sum_inv = torch.repeat_interleave(exp_sum_inv, P_1.shape[0], dim=0)

        #Now we can have multiple types of integrals: 1:1, 1:0, 0:1 and 0:0. Handle all these conditions
        am_1, am_2 = torch.tensor([1.0], device=self.device), torch.tensor([1.0], device=self.device)
        fac = torch.ones_like(exp_sum_inv)
        fac = self.obara_saika_recursion(P_1, P_2, exp_sum_inv, am_1, am_2, fac)
        num_int = int(fac.shape[1]/3)
        key = (1.0, 1.0)
        integral_dict['y'][key] = fac[:, 0:num_int]
        integral_dict['x'][key] = fac[:, num_int:num_int*2]
        integral_dict['z'][key] = fac[:, num_int*2:num_int*3]

        am_1, am_2 = torch.tensor([1.0], device=self.device), torch.tensor([0.0], device=self.device)
        fac = torch.ones_like(exp_sum_inv)
        fac = self.obara_saika_recursion(P_1, P_2, exp_sum_inv, am_1, am_2, fac)
        key = (1.0, 0.0)
        integral_dict['y'][key] = fac[:, 0:num_int]
        integral_dict['x'][key] = fac[:, num_int:num_int*2]
        integral_dict['z'][key] = fac[:, num_int*2:num_int*3]

        am_1, am_2 = torch.tensor([0.0], device=self.device), torch.tensor([1.0], device=self.device)
        fac = torch.ones_like(exp_sum_inv)
        fac = self.obara_saika_recursion(P_1, P_2, exp_sum_inv, am_1, am_2, fac)
        key = (0.0, 1.0)
        integral_dict['y'][key] = fac[:, 0:num_int]
        integral_dict['x'][key] = fac[:, num_int:num_int*2]
        integral_dict['z'][key] = fac[:, num_int*2:num_int*3]

        key = (0.0, 0.0)
        integral_dict['y'][key] = torch.ones_like(fac[:, 0:num_int])
        integral_dict['x'][key] = torch.ones_like(fac[:, 0:num_int])
        integral_dict['z'][key] = torch.ones_like(fac[:, 0:num_int])

        #Add each integral to integral_list
        integral_list = []
        for i in am_ints:
            fac = torch.ones_like(integ)
            for cart in range(3):
                d = coord_dict[cart]
                key = i[:, cart]
                key = (key[0].item(), key[1].item())
                fac = fac * integral_dict[d][key]
            integral_add = fac * integ
            integral_add = integral_add.sum(axis=-1)
            integral_list.append(integral_add)

        integral_list = [integral_list[i].float() for i in reorder]
        
        add_S_index = self.get_add_index(func_indices)

        x_orbs_p = p_orbitals[:, 0]
        y_orbs_p = p_orbitals[:, 1]
        z_orbs_p = p_orbitals[:, 2]

        add_S_ord = torch.zeros_like(add_S_index)
        orb_pairs = []
        for p_orb1 in [x_orbs_p, y_orbs_p, z_orbs_p]:
            for p_orb2 in [x_orbs_p, y_orbs_p, z_orbs_p]:
                p_loc1 = torch.isin(add_S_index, p_orb1)
                p_loc2 = torch.isin(add_S_index, p_orb2)
                orb_pair = torch.where(
                    torch.logical_and(p_loc1[:,0], p_loc2[:,1]),
                    )[0]
                orb_pairs.append(orb_pair)

        start = 0
        for orb in orb_pairs:
            add_S_ord[start:start+len(orb), :] = add_S_index[orb]
            start += len(orb)

        add_S_ord, sort_index = torch.sort(add_S_ord, dim=1)
        add_S_ord, unique_inds, counts = torch.unique(add_S_ord, dim=0, return_inverse=True, return_counts=True)

        t, ind_sorted = torch.sort(unique_inds, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0], device=self.device), cum_sum[:-1]))
        unique_inds = ind_sorted[cum_sum]
        add_S_ord = add_S_ord[torch.argsort(unique_inds)]
        unique_inds, sort_inds = torch.sort(unique_inds, dim=0)

        integral_list = torch.cat(integral_list, dim=1)
        integral_list = integral_list[:, unique_inds]
      
        index_batch = torch.arange(integral_list.shape[0], device=self.device)
        index_batch = torch.repeat_interleave(index_batch, add_S_ord.shape[0], dim=0)
        add_S_index_1 = torch.tile(add_S_ord[:, 0], (integral_list.shape[0],))
        add_S_index_2 = torch.tile(add_S_ord[:, 1], (integral_list.shape[0],))
        integral_list = integral_list.flatten()

        S_tmp = torch.zeros_like(self.S)
        S_tmp.index_put_((index_batch, add_S_index_1, add_S_index_2), integral_list)

        S_diag = S_tmp[:, torch.eye(S_tmp.shape[1]).bool()]
        S_diag = torch.diag_embed(S_diag)
        self.S += S_tmp + S_tmp.transpose(1,2) - S_diag

    def sd_one_electron(self, reorder):
        """
        Parameters
        -----------
        reorder: list
            reorder the integrals according to reorder
        """
        s_d = torch.where(torch.logical_or(
            torch.logical_and(self.am_pairs[:, 0] == 0, self.am_pairs[:, 1] == 2),
            torch.logical_and(self.am_pairs[:, 0] == 2, self.am_pairs[:, 1] == 0))
            )[0]

        d_orbitals = self.shell_func_indices[self.am == 2]
        coeffs, exps, positions, R2, func_indices, am = self.get_pair_values(s_d)
        coeffs = coeffs[None]
        exps = exps[None]
        
        exp_sum = exps.sum(axis=-1)
        exp_sum_inv = torch.zeros_like(exp_sum)
        exp_sum_inv[exp_sum != 0] = 1/exp_sum[exp_sum != 0]

        integ = self.get_s00_integral(coeffs, exps, exp_sum_inv, R2)
        P_coords = self.get_P_coordinate(positions, exps, exp_sum_inv)
        
        am_pairs = am[:, None]
        am_pairs = torch.repeat_interleave(am_pairs, coeffs.shape[2], axis=1)
        am_pairs = am_pairs[None]
        am_pairs = torch.repeat_interleave(am_pairs, integ.shape[0], axis=0)

        PX = torch.where(am_pairs[:, :, :, 0] == 2, P_coords[0][0], P_coords[0][1])
        PY = torch.where(am_pairs[:, :, :, 0] == 2, P_coords[1][0], P_coords[1][1])
        PZ = torch.where(am_pairs[:, :, :, 0] == 2, P_coords[2][0], P_coords[2][1])

        PX_alt = torch.where(am_pairs[:, :, :, 0] == 0, P_coords[0][0], P_coords[0][1])
        PY_alt = torch.where(am_pairs[:, :, :, 0] == 0, P_coords[1][0], P_coords[1][1])
        PZ_alt = torch.where(am_pairs[:, :, :, 0] == 0, P_coords[2][0], P_coords[2][1])

        a_max = torch.max(am_pairs)
        am_values = torch.arange(a_max+1)
        am_grid = torch.meshgrid(am_values, am_values, am_values, indexing='ij')
        am_grid = torch.stack(am_grid)
        am_grid = torch.transpose(am_grid, 0, -1).reshape(-1, 3)

        d_orbs = am_grid[torch.sum(am_grid, axis=1) == 2]
        s_orbs = am_grid[torch.sum(am_grid, axis=1) == 0]

        am_d = torch.tile(d_orbs, (s_orbs.shape[0], 1))
        am_s = torch.repeat_interleave(s_orbs, d_orbs.shape[0], axis=0)
        am_ints = torch.hstack((am_d, am_s))
        am_ints = am_ints.reshape(am_ints.shape[0], 2, -1)
        
        coord_dict = {0: 'x', 1: 'y', 2: 'z'}
        integral_dict = {'x': {}, 'y': {}, 'z': {}}
        P_1 = torch.cat((PY, PX, PZ), dim=1)
        P_2 = torch.cat((PY_alt, PX_alt, PZ_alt), dim=1)
        exps = torch.tile(exps, (1,3,1,1))
        exp_sum_inv = torch.tile(exp_sum_inv, (1,3,1))
        exp_sum_inv = torch.repeat_interleave(exp_sum_inv, P_1.shape[0], dim=0)

        #Now can have 2:0, 1:0 and 0:0. We don't need to do 0:2, etc. here as we can handle that with the above where statement
        fac = torch.ones_like(exp_sum_inv)
        am_1, am_2 = torch.tensor([2.0], device=self.device), torch.tensor([0.0], device=self.device)
        fac = self.obara_saika_recursion(P_1, P_2, exp_sum_inv, am_1, am_2, fac)
        num_int = int(fac.shape[1]/3)
        integral_dict['y'][(2.0, 0.0)] = fac[:, 0:num_int]
        integral_dict['x'][(2.0, 0.0)] = fac[:, num_int:num_int*2]
        integral_dict['z'][(2.0, 0.0)] = fac[:, num_int*2:num_int*3]

        am_1, am_2 = torch.tensor([1.0], device=self.device), torch.tensor([0.0], device=self.device)
        fac = torch.ones_like(exp_sum_inv)
        fac = self.obara_saika_recursion(P_1, P_2, exp_sum_inv, am_1, am_2, fac)
        integral_dict['y'][(1.0, 0.0)] = fac[:, 0:num_int]
        integral_dict['x'][(1.0, 0.0)] = fac[:, num_int:num_int*2]
        integral_dict['z'][(1.0, 0.0)] = fac[:, num_int*2:num_int*3]

        integral_dict['y'][(0.0, 0.0)] = torch.ones_like(fac[:, 0:num_int])
        integral_dict['x'][(0.0, 0.0)] = torch.ones_like(fac[:, 0:num_int])
        integral_dict['z'][(0.0, 0.0)] = torch.ones_like(fac[:, 0:num_int])

        integral_list = []
        for i in am_ints:
            fac = torch.ones_like(integ)
            for cart in range(3):
                d = coord_dict[cart]
                sort_id, index = i[:, cart].sort(descending=True)
                key = (sort_id[0].item(), sort_id[1].item())
                fac = fac * integral_dict[d][key]
            integral_add = fac * integ
            integral_add = integral_add.sum(axis=-1)
            integral_list.append(integral_add)

        integral_list = [integral_list[i].float() for i in reorder]

        add_S_index = self.get_add_index(func_indices)

        xx_orbs_d = d_orbitals[:,0]
        xy_orbs_d = d_orbitals[:,1]
        xz_orbs_d = d_orbitals[:,2]
        yy_orbs_d = d_orbitals[:,3]
        yz_orbs_d = d_orbitals[:,4]
        zz_orbs_d = d_orbitals[:,5]
        
        add_S_ord = torch.zeros_like(add_S_index)
        orb_pairs = []
        for d_orb in [xx_orbs_d, xy_orbs_d, xz_orbs_d, yy_orbs_d, yz_orbs_d, zz_orbs_d]:
            d_loc = torch.isin(add_S_index, d_orb)
            orb_pair = torch.where(d_loc)[0]
            orb_pairs.append(orb_pair)

        start = 0
        for orb in orb_pairs:
            add_S_ord[start:start+len(orb), :] = add_S_index[orb]
            start += len(orb)

        integral_list = torch.cat(integral_list, dim=1)
        index_batch = torch.arange(integ.shape[0], device=self.device)
        index_batch = torch.repeat_interleave(index_batch, add_S_ord.shape[0], dim=0)
        add_S_index_1 = torch.tile(add_S_ord[:, 0], (integral_list.shape[0],))
        add_S_index_2 = torch.tile(add_S_ord[:, 1], (integral_list.shape[0],))
        integral_list = integral_list.flatten()

        S_tmp = torch.zeros_like(self.S)
        S_tmp.index_put_((index_batch, add_S_index_1, add_S_index_2), integral_list)

        S_diag = S_tmp[:, torch.eye(S_tmp.shape[1]).bool()]
        S_diag = torch.diag_embed(S_diag)
        self.S += S_tmp + S_tmp.transpose(1,2) - S_diag

    def pd_one_electron(self, reorder):
        """
        Parameters
        -----------
        reorder: list
            reorder the integrals according to reorder
        """
        p_d = torch.where(torch.logical_or(
            torch.logical_and(self.am_pairs[:, 0] == 1, self.am_pairs[:, 1] == 2),
            torch.logical_and(self.am_pairs[:, 0] == 2, self.am_pairs[:, 1] == 1))
            )[0]    

        d_orbitals = self.shell_func_indices[self.am == 2]
        p_orbitals = self.shell_func_indices[self.am == 1]

        coeffs, exps, positions, R2, func_indices, am = self.get_pair_values(p_d)
        coeffs = coeffs[None]
        exps = exps[None]

        exp_sum = exps.sum(axis=-1)
        exp_sum_inv = torch.zeros_like(exp_sum)
        exp_sum_inv[exp_sum != 0] = 1/exp_sum[exp_sum != 0]

        integ = self.get_s00_integral(coeffs, exps, exp_sum_inv, R2)
        P_coords = self.get_P_coordinate(positions, exps, exp_sum_inv)

        am_pairs = am[:, None]
        am_pairs = torch.repeat_interleave(am_pairs, coeffs.shape[2], axis=1)
        am_pairs = am_pairs[None]
        am_pairs = torch.repeat_interleave(am_pairs, integ.shape[0], axis=0)

        PX = torch.where(am_pairs[:, :, :, 0] == 2, P_coords[0][0], P_coords[0][1])
        PY = torch.where(am_pairs[:, :, :, 0] == 2, P_coords[1][0], P_coords[1][1])
        PZ = torch.where(am_pairs[:, :, :, 0] == 2, P_coords[2][0], P_coords[2][1])

        PX_alt = torch.where(am_pairs[:, :, :, 0] == 1, P_coords[0][0], P_coords[0][1])
        PY_alt = torch.where(am_pairs[:, :, :, 0] == 1, P_coords[1][0], P_coords[1][1])
        PZ_alt = torch.where(am_pairs[:, :, :, 0] == 1, P_coords[2][0], P_coords[2][1])

        a_max = torch.max(am_pairs)
        am_values = torch.arange(a_max+1)
        am_grid = torch.meshgrid(am_values, am_values, am_values, indexing='ij')
        am_grid = torch.stack(am_grid)
        am_grid = torch.transpose(am_grid, 0, -1).reshape(-1, 3)

        d_orbs = am_grid[torch.sum(am_grid, axis=1) == 2]
        p_orbs = am_grid[torch.sum(am_grid, axis=1) == 1]

        am_d = torch.tile(d_orbs, (p_orbs.shape[0], 1))
        am_p = torch.repeat_interleave(p_orbs, d_orbs.shape[0], axis=0)
        am_ints = torch.hstack((am_d, am_p))
        am_ints = am_ints.reshape(am_ints.shape[0], 2, -1)

        coord_dict = {0: 'x', 1: 'y', 2: 'z'}
        integral_dict = {'x': {}, 'y': {}, 'z': {}}
        P_1 = torch.cat((PY, PX, PZ), dim=1)
        P_2 = torch.cat((PY_alt, PX_alt, PZ_alt), dim=1)
        exps = torch.tile(exps, (1,3,1,1))
        exp_sum_inv = torch.tile(exp_sum_inv, (1,3,1))
        exp_sum_inv = torch.repeat_interleave(exp_sum_inv, P_1.shape[0], dim=0) 

        #Now we need 2:1, 1:2, 2:0, 0:2, 1:1, 1:0, 0:1 and 0:0
        #At least I think so
        fac = torch.ones_like(exp_sum_inv)
        am_1, am_2 = torch.tensor([2.0], device=self.device), torch.tensor([1.0], device=self.device)
        fac = self.obara_saika_recursion(P_1, P_2, exp_sum_inv, am_1, am_2, fac)
        num_int = int(fac.shape[1]/3)
        integral_dict['y'][(2.0, 1.0)] = fac[:, 0:num_int]
        integral_dict['x'][(2.0, 1.0)] = fac[:, num_int:num_int*2]
        integral_dict['z'][(2.0, 1.0)] = fac[:, num_int*2:num_int*3]

        am_1, am_2 = torch.tensor([1.0], device=self.device), torch.tensor([2.0], device=self.device)
        fac = torch.ones_like(exp_sum_inv)
        fac = self.obara_saika_recursion(P_1, P_2, exp_sum_inv, am_1, am_2, fac)
        integral_dict['y'][(1.0, 2.0)] = fac[:, 0:num_int]
        integral_dict['x'][(1.0, 2.0)] = fac[:, num_int:num_int*2]
        integral_dict['z'][(1.0, 2.0)] = fac[:, num_int*2:num_int*3]

        am_1, am_2 = torch.tensor([2.0], device=self.device), torch.tensor([0.0], device=self.device)
        fac = torch.ones_like(exp_sum_inv)
        fac = self.obara_saika_recursion(P_1, P_2, exp_sum_inv, am_1, am_2, fac)
        integral_dict['y'][(2.0, 0.0)] = fac[:, 0:num_int]
        integral_dict['x'][(2.0, 0.0)] = fac[:, num_int:num_int*2]
        integral_dict['z'][(2.0, 0.0)] = fac[:, num_int*2:num_int*3]

        am_1, am_2 = torch.tensor([0.0], device=self.device), torch.tensor([2.0], device=self.device)
        fac = torch.ones_like(exp_sum_inv)
        fac = self.obara_saika_recursion(P_1, P_2, exp_sum_inv, am_1, am_2, fac)
        integral_dict['y'][(0.0, 2.0)] = fac[:, 0:num_int]
        integral_dict['x'][(0.0, 2.0)] = fac[:, num_int:num_int*2]
        integral_dict['z'][(0.0, 2.0)] = fac[:, num_int*2:num_int*3]

        am_1, am_2 = torch.tensor([1.0], device=self.device), torch.tensor([1.0], device=self.device)
        fac = torch.ones_like(exp_sum_inv)
        fac = self.obara_saika_recursion(P_1, P_2, exp_sum_inv, am_1, am_2, fac)
        integral_dict['y'][(1.0, 1.0)] = fac[:, 0:num_int]
        integral_dict['x'][(1.0, 1.0)] = fac[:, num_int:num_int*2]
        integral_dict['z'][(1.0, 1.0)] = fac[:, num_int*2:num_int*3]

        am_1, am_2 = torch.tensor([1.0], device=self.device), torch.tensor([0.0], device=self.device)
        fac = torch.ones_like(exp_sum_inv)
        fac = self.obara_saika_recursion(P_1, P_2, exp_sum_inv, am_1, am_2, fac)
        integral_dict['y'][(1.0, 0.0)] = fac[:, 0:num_int]
        integral_dict['x'][(1.0, 0.0)] = fac[:, num_int:num_int*2]
        integral_dict['z'][(1.0, 0.0)] = fac[:, num_int*2:num_int*3]

        am_1, am_2 = torch.tensor([0.0], device=self.device), torch.tensor([1.0], device=self.device)
        fac = torch.ones_like(exp_sum_inv)
        fac = self.obara_saika_recursion(P_1, P_2, exp_sum_inv, am_1, am_2, fac)
        integral_dict['y'][(0.0, 1.0)] = fac[:, 0:num_int]
        integral_dict['x'][(0.0, 1.0)] = fac[:, num_int:num_int*2]
        integral_dict['z'][(0.0, 1.0)] = fac[:, num_int*2:num_int*3]

        integral_dict['y'][(0.0, 0.0)] = torch.ones_like(fac[:, 0:num_int])
        integral_dict['x'][(0.0, 0.0)] = torch.ones_like(fac[:, 0:num_int])
        integral_dict['z'][(0.0, 0.0)] = torch.ones_like(fac[:, 0:num_int])

        integral_list = []
        for i in am_ints:
            fac = torch.ones_like(integ)
            for cart in range(3):
                d = coord_dict[cart]
                key = (i[:, cart][0].item(), i[:, cart][1].item())
                fac = fac * integral_dict[d][key]
            integral_add = fac * integ
            integral_add = integral_add.sum(axis=-1)
            integral_list.append(integral_add)

        integral_list = [integral_list[i].float() for i in reorder]

        add_S_index = self.get_add_index(func_indices)

        x_orbs_p = p_orbitals[:, 0]
        y_orbs_p = p_orbitals[:, 1]
        z_orbs_p = p_orbitals[:, 2]
        xx_orbs_d = d_orbitals[:,0]
        xy_orbs_d = d_orbitals[:,1]
        xz_orbs_d = d_orbitals[:,2]
        yy_orbs_d = d_orbitals[:,3]
        yz_orbs_d = d_orbitals[:,4]
        zz_orbs_d = d_orbitals[:,5]

        add_S_ord = torch.zeros_like(add_S_index)
        orb_pairs = []
        for p_orb in [x_orbs_p, y_orbs_p, z_orbs_p]:
            for d_orb in [xx_orbs_d, xy_orbs_d, xz_orbs_d, yy_orbs_d, yz_orbs_d, zz_orbs_d]:
                p_loc = torch.isin(add_S_index, p_orb)
                d_loc = torch.isin(add_S_index, d_orb)
                orb_pair = torch.where(torch.logical_or(
                torch.logical_and(p_loc[:,0], d_loc[:,1]),
                torch.logical_and(p_loc[:,1], d_loc[:,0]),
                ))[0]
                orb_pairs.append(orb_pair)

        start = 0
        for orb in orb_pairs:
            add_S_ord[start:start+len(orb), :] = add_S_index[orb]
            start += len(orb)

        integral_list = torch.cat(integral_list, dim=1)
        index_batch = torch.arange(integral_list.shape[0], device=self.device)
        index_batch = torch.repeat_interleave(index_batch, add_S_ord.shape[0], dim=0)
        add_S_index_1 = torch.tile(add_S_ord[:, 0], (integral_list.shape[0],))
        add_S_index_2 = torch.tile(add_S_ord[:, 1], (integral_list.shape[0],))
        integral_list = integral_list.flatten()

        S_tmp = torch.zeros_like(self.S)
        S_tmp.index_put_((index_batch, add_S_index_1, add_S_index_2), integral_list)

        S_diag = S_tmp[:, torch.eye(S_tmp.shape[1]).bool()]
        S_diag = torch.diag_embed(S_diag)
        self.S += S_tmp + S_tmp.transpose(1,2) - S_diag

    def dd_one_electron(self, reorder):
        """
        Parameters
        -----------
        reorder: list
            reorder the integrals according to reorder
        """

        d_d = torch.where(torch.logical_and(self.am_pairs[:, 0] == 2, self.am_pairs[:, 1] == 2))[0]

        d_orbitals = self.shell_func_indices[self.am == 2]
        coeffs, exps, positions, R2, func_indices, am = self.get_pair_values(d_d)
        coeffs = coeffs[None]
        exps = exps[None]

        exp_sum = exps.sum(axis=-1)
        exp_sum_inv = torch.zeros_like(exp_sum)
        exp_sum_inv[exp_sum != 0] = 1/exp_sum[exp_sum != 0]

        integ = self.get_s00_integral(coeffs, exps, exp_sum_inv, R2)
        P_coords = self.get_P_coordinate(positions, exps, exp_sum_inv)

        am_pairs = am[:, None]
        am_pairs = torch.repeat_interleave(am_pairs, coeffs.shape[1], axis=1)
        a_max = torch.max(am_pairs)
        am_values = torch.arange(a_max+1)
        am_grid = torch.meshgrid(am_values, am_values, am_values, indexing='ij')
        am_grid = torch.stack(am_grid)
        am_grid = torch.transpose(am_grid, 0, 3).reshape(-1, 3)

        d_orbs = am_grid[torch.sum(am_grid, axis=1) == 2]

        am_d1 = torch.tile(d_orbs, (d_orbs.shape[0], 1))
        am_d2 = torch.repeat_interleave(d_orbs, d_orbs.shape[0], axis=0)
        am_ints = torch.hstack((am_d1, am_d2))
        am_ints = am_ints.reshape(am_ints.shape[0], 2, -1)

        coord_dict = {0: 'x', 1: 'y', 2: 'z'}
        integral_dict = {'x': {}, 'y': {}, 'z': {}}
        P_1 = torch.cat((P_coords[1][0], P_coords[0][0], P_coords[2][0]), dim=1)
        P_2 = torch.cat((P_coords[1][1], P_coords[0][1], P_coords[2][1]), dim=1)
        exps = torch.tile(exps, (1,3,1,1))
        exp_sum_inv = torch.tile(exp_sum_inv, (1,3,1))
        exp_sum_inv = torch.repeat_interleave(exp_sum_inv, P_1.shape[0], dim=0)

        #Do 2:2, 2:1, 1:2, 1:1, 1:0, 0:1 and 0:0 integrals
        am_1, am_2 = torch.tensor([2.0], device=self.device), torch.tensor([2.0], device=self.device)
        fac = torch.ones_like(exp_sum_inv)
        fac = self.obara_saika_recursion(P_1, P_2, exp_sum_inv, am_1, am_2, fac)
        num_int = int(fac.shape[1]/3)
        integral_dict['y'][(2.0, 2.0)] = fac[:, 0:num_int]
        integral_dict['x'][(2.0, 2.0)] = fac[:, num_int:num_int*2]
        integral_dict['z'][(2.0, 2.0)] = fac[:, num_int*2:num_int*3]
        
        am_1, am_2 = torch.tensor([2.0], device=self.device), torch.tensor([1.0], device=self.device)
        fac = torch.ones_like(exp_sum_inv)
        fac = self.obara_saika_recursion(P_1, P_2, exp_sum_inv, am_1, am_2, fac)
        integral_dict['y'][(2.0, 1.0)] = fac[:, 0:num_int]
        integral_dict['x'][(2.0, 1.0)] = fac[:, num_int:num_int*2]
        integral_dict['z'][(2.0, 1.0)] = fac[:, num_int*2:num_int*3]

        am_1, am_2 = torch.tensor([1.0], device=self.device), torch.tensor([2.0], device=self.device)
        fac = torch.ones_like(exp_sum_inv)
        fac = self.obara_saika_recursion(P_1, P_2, exp_sum_inv, am_1, am_2, fac)
        integral_dict['y'][(1.0, 2.0)] = fac[:, 0:num_int]
        integral_dict['x'][(1.0, 2.0)] = fac[:, num_int:num_int*2]
        integral_dict['z'][(1.0, 2.0)] = fac[:, num_int*2:num_int*3]

        am_1, am_2 = torch.tensor([2.0], device=self.device), torch.tensor([0.0], device=self.device)
        fac = torch.ones_like(exp_sum_inv)
        fac = self.obara_saika_recursion(P_1, P_2, exp_sum_inv, am_1, am_2, fac)
        integral_dict['y'][(2.0, 0.0)] = fac[:, 0:num_int]
        integral_dict['x'][(2.0, 0.0)] = fac[:, num_int:num_int*2]
        integral_dict['z'][(2.0, 0.0)] = fac[:, num_int*2:num_int*3]

        am_1, am_2 = torch.tensor([0.0], device=self.device), torch.tensor([2.0], device=self.device)
        fac = torch.ones_like(exp_sum_inv)
        fac = self.obara_saika_recursion(P_1, P_2, exp_sum_inv, am_1, am_2, fac)
        integral_dict['y'][(0.0, 2.0)] = fac[:, 0:num_int]
        integral_dict['x'][(0.0, 2.0)] = fac[:, num_int:num_int*2]
        integral_dict['z'][(0.0, 2.0)] = fac[:, num_int*2:num_int*3]

        am_1, am_2 = torch.tensor([1.0], device=self.device), torch.tensor([1.0], device=self.device)
        fac = torch.ones_like(exp_sum_inv)
        fac = self.obara_saika_recursion(P_1, P_2, exp_sum_inv, am_1, am_2, fac)
        integral_dict['y'][(1.0, 1.0)] = fac[:, 0:num_int]
        integral_dict['x'][(1.0, 1.0)] = fac[:, num_int:num_int*2]
        integral_dict['z'][(1.0, 1.0)] = fac[:, num_int*2:num_int*3]

        am_1, am_2 = torch.tensor([1.0], device=self.device), torch.tensor([0.0], device=self.device)
        fac = torch.ones_like(exp_sum_inv)
        fac = self.obara_saika_recursion(P_1, P_2, exp_sum_inv, am_1, am_2, fac)
        integral_dict['y'][(1.0, 0.0)] = fac[:, 0:num_int]
        integral_dict['x'][(1.0, 0.0)] = fac[:, num_int:num_int*2]
        integral_dict['z'][(1.0, 0.0)] = fac[:, num_int*2:num_int*3]

        am_1, am_2 = torch.tensor([0.0], device=self.device), torch.tensor([1.0], device=self.device)
        fac = torch.ones_like(exp_sum_inv)
        fac = self.obara_saika_recursion(P_1, P_2, exp_sum_inv, am_1, am_2, fac)
        integral_dict['y'][(0.0, 1.0)] = fac[:, 0:num_int]
        integral_dict['x'][(0.0, 1.0)] = fac[:, num_int:num_int*2]
        integral_dict['z'][(0.0, 1.0)] = fac[:, num_int*2:num_int*3]

        integral_dict['y'][(0.0, 0.0)] = torch.ones_like(fac[:, 0:num_int])
        integral_dict['x'][(0.0, 0.0)] = torch.ones_like(fac[:, 0:num_int])
        integral_dict['z'][(0.0, 0.0)] = torch.ones_like(fac[:, 0:num_int])

        integral_list = []
        for i in am_ints:
            fac = torch.ones_like(integ)
            for cart in range(3):
                d = coord_dict[cart]
                key = (i[:, cart][0].item(), i[:, cart][1].item())
                fac = fac * integral_dict[d][key]
            integral_add = fac * integ
            integral_add = integral_add.sum(axis=-1)
            integral_list.append(integral_add)
        integral_list = [integral_list[i].float() for i in reorder]

        add_S_index = self.get_add_index(func_indices)

        xx_orbs_d = d_orbitals[:,0]
        xy_orbs_d = d_orbitals[:,1]
        xz_orbs_d = d_orbitals[:,2]
        yy_orbs_d = d_orbitals[:,3]
        yz_orbs_d = d_orbitals[:,4]
        zz_orbs_d = d_orbitals[:,5]

        add_S_ord = torch.zeros_like(add_S_index)
        orb_pairs = []
        for d_orb1 in [xx_orbs_d, xy_orbs_d, xz_orbs_d, yy_orbs_d, yz_orbs_d, zz_orbs_d]:
            for d_orb2 in [xx_orbs_d, xy_orbs_d, xz_orbs_d, yy_orbs_d, yz_orbs_d, zz_orbs_d]:
                d_loc1 = torch.isin(add_S_index, d_orb1)
                d_loc2 = torch.isin(add_S_index, d_orb2)
                orb_pair = torch.where(
                    torch.logical_and(d_loc1[:,0], d_loc2[:,1]),
                    )[0]
                orb_pairs.append(orb_pair)

        start = 0   
        for orb in orb_pairs:
            add_S_ord[start:start+len(orb), :] = add_S_index[orb] 
            start += len(orb) 

        add_S_ord, sort_index = torch.sort(add_S_ord, dim=1)
        add_S_ord, unique_inds, counts = torch.unique(add_S_ord, dim=0, return_inverse=True, return_counts=True)

        t, ind_sorted = torch.sort(unique_inds, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0], device=self.device), cum_sum[:-1]))
        unique_inds = ind_sorted[cum_sum]
        add_S_ord = add_S_ord[torch.argsort(unique_inds)]
        unique_inds, sort_inds = torch.sort(unique_inds, dim=0)

        integral_list = torch.cat(integral_list, dim=1)
        integral_list = integral_list[:, unique_inds]

        index_batch = torch.arange(integral_list.shape[0], device=self.device)
        index_batch = torch.repeat_interleave(index_batch, add_S_ord.shape[0], dim=0)
        add_S_index_1 = torch.tile(add_S_ord[:, 0], (integral_list.shape[0],))
        add_S_index_2 = torch.tile(add_S_ord[:, 1], (integral_list.shape[0],))
        integral_list = integral_list.flatten()

        S_tmp = torch.zeros_like(self.S)
        S_tmp.index_put_((index_batch, add_S_index_1, add_S_index_2), integral_list)

        S_diag = S_tmp[:, torch.eye(S_tmp.shape[1]).bool()]
        S_diag = torch.diag_embed(S_diag)
        self.S += S_tmp + S_tmp.transpose(1,2) - S_diag

    def forward(self, inputs):
        """
        Forward pass through overlap matrix
        Parameters
        -----------
        inputs: dict
            Contains all tensors corresponding to the system data

        Returns
        -----------
        self.S: tensor shape (n_batch, n_basis_function, n_basis_function)
            overlap matrix for each input geometry in the batch
        """
        #Unpacks input dictionary and determines highest angular momentum needed
        self._setup_integral(inputs)
       
        #Loop through all required integral functions
        for int_name in self.required_integrals:
            fn = getattr(self, int_name)
            reorder = self.reorder[int_name]
            fn(reorder)
        
        #Convert S matrix from cartesian to spherical if necessary
        if self.coord == 'spherical':
            self.c2sph = self.c2sph.float().to(self.device)
            self.S = torch.matmul(self.c2sph.T, self.S)
            self.S = torch.matmul(self.S, self.c2sph)

        """
        dy = None
        if self.requires_derivative:
            dy = grad(
                self.S,
                self.R,
                grad_outputs=torch.ones_like(self.S),
                create_graph=True,
                retain_graph=True,
                )[0]
            dy = dy.detach().cpu().numpy()
        """
        return self.S
