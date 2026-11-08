#!/usr/bin/env python
import torch
from torch import nn

class StandardOrientation(nn.Module):
    """
    Shift monomer in dimer to Standard Orientation
    """
    def __init__(self, device="cuda"):
        """
        Parameters
        -----------
        device: str
            Whether to perform operations on cpu or cuda
        """
        super(StandardOrientation, self).__init__()
        self.device = torch.device(device)

    def rotate_monomer(self, frag_pos, frag_mass):
        """
        rotate monomer to standard orientation

        Parameters
        -----------
        frag_pos: tensor
            monomer position
        frag_mass: tensor
            masses of monomer

        Returns
        -----------
        frag_pos_p: tensor
            rotated monomer position
        rmatrix: tensor
            rotation matrix used to rotate the molecule
        """
        #Shift COM to origin
        frag_com = (frag_pos * frag_mass[:, :, None]).sum(axis=1)/frag_mass.sum(axis=1)[:, None]
        frag_pos -= frag_com[:, None]
        
        #Get inertia tensor
        I = torch.zeros((frag_pos.shape[0], 3, 3), device=self.device)
        I[:, 0, 0] = (frag_mass * (frag_pos[:, :, 1]**2 + frag_pos[:, :, 2]**2)).sum(axis=1)
        I[:, 1, 1] = (frag_mass * (frag_pos[:, :, 0]**2 + frag_pos[:, :, 2]**2)).sum(axis=1)
        I[:, 2, 2] = (frag_mass * (frag_pos[:, :, 0]**2 + frag_pos[:, :, 1]**2)).sum(axis=1)
        I[:, 0, 1] = -(frag_mass * frag_pos[:, :, 0] * frag_pos[:, :, 1]).sum(axis=1)
        I[:, 0, 2] = -(frag_mass * frag_pos[:, :, 0] * frag_pos[:, :, 2]).sum(axis=1)
        I[:, 1, 2] = -(frag_mass * frag_pos[:, :, 1] * frag_pos[:, :, 2]).sum(axis=1)

        I_diag = I[:, torch.eye(3,3).bool()]
        I_sub = torch.zeros_like(I)
        I_sub[:, torch.eye(3,3).bool()] += I_diag
        I = I + I.transpose(1, 2) - I_sub
        #eigenvector and eigenvalue of inertia
        eig, eigv = torch.linalg.eigh(I)
        eig_order, indices = torch.sort(eig, dim=1, descending=True)
       
        #Reorder eigenvectors to go in descending value by eigenvalue
        indices = indices.flatten()
        ind3 = torch.tile(indices, (eigv.shape[1],))
        
        ind1 = torch.arange(eigv.shape[0])
        ind1 = torch.repeat_interleave(ind1, eigv.shape[1]*eigv.shape[2])
        
        ind2 = torch.arange(eigv.shape[1])
        ind2 = torch.repeat_interleave(ind2, eigv.shape[1])
        ind2 = torch.tile(ind2, (eigv.shape[0],))

        eigv_order = eigv[(ind1, ind2, ind3)]
        eigv_order = eigv_order.reshape(eigv.shape[0], eigv.shape[1], eigv.shape[2])
        #May need to edit this if any molecule is linear or some other problem
        check_eig_equal = torch.where(eig_order[:, 0] == eig_order[:, 1])[0]

        if check_eig_equal.shape[0] > 9: 
            print("WARNING: EIGENVALUE MATCH FOUND. Could be linear molecule or \\"  
                    "another problem.")
            sys.exit()
       
        #Get component of x along the last eigenvector direction
        x_dir = (frag_pos[:, 0] * eigv_order[:, :, 2]).sum(axis=1)
        
        eig_sign = torch.sign(x_dir)
        eig_sign = eig_sign[:, None, None]
        eigv_order = eigv_order * eig_sign

        cartesian = torch.zeros_like(eigv_order)
        cartesian[:] = torch.eye(3)

        #Get rotation matrix along z direction
        rmatrixz = self.rotation_matrix(eigv_order[:, :, 0], cartesian[:, 2])
        tmp = torch.zeros_like(rmatrixz)
        rmatrixz = torch.where(rmatrixz.abs() > 1e-7, rmatrixz, tmp)
        
        #Get rmatrixz multipled by each eigenvector
        vec_z = (rmatrixz * eigv_order[:, :, 0][:, None]).sum(axis=-1)
        vec_y = (rmatrixz * eigv_order[:, :, 1][:, None]).sum(axis=-1)
        vec_x = (rmatrixz * eigv_order[:, :, 2][:, None]).sum(axis=-1)
        
        #Get rmatrixy rotation
        rmatrixy = self.rotation_matrix(vec_y, cartesian[:, 1])
        tmp = torch.zeros_like(rmatrixy)
        rmatrixy = torch.where(rmatrixy.abs() > 1e-7, rmatrixy, tmp)
        vec_z = (rmatrixy * eigv_order[:, :, 0][:, None]).sum(axis=-1)
        vec_y = (rmatrixy * eigv_order[:, :, 1][:, None]).sum(axis=-1)
        vec_x = (rmatrixy * eigv_order[:, :, 2][:, None]).sum(axis=-1)

        #Combine rmatrixy and rmatrixz
        rmatrix = torch.matmul(rmatrixy, rmatrixz)
        add_z = torch.Tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).to(self.device)
        add_y = torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).to(self.device)
        add_x = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).to(self.device)
        flip_z = torch.zeros_like(rmatrix)
        flip_y = torch.zeros_like(rmatrix)
        flip_x = torch.zeros_like(rmatrix)
        flip_z[:] += add_z
        flip_y[:] += add_y
        flip_x[:] += add_x
        I = torch.zeros_like(rmatrix)
        I[:] = torch.eye(3)

        #Figure out where to rotate the vectors 180 or not
        test_rot = torch.matmul(rmatrix, frag_pos.transpose(1,2)).transpose(1,2)
        test_rot_x = test_rot[:, 2, 0][:, None, None]
        test_rot_y = test_rot[:, 2, 1][:, None, None]
        flip_x = torch.where(torch.logical_and(test_rot_x > 0, test_rot_y > 0), flip_x, I)
        rmatrix = torch.matmul(flip_x, rmatrix)
        flip_y = torch.where(torch.logical_and(test_rot_x < 0, test_rot_y < 0), flip_y, I)
        rmatrix = torch.matmul(flip_y, rmatrix)
        flip_z = torch.where(torch.logical_and(test_rot_x < 0, test_rot_y > 0), flip_z, I)
        rmatrix = torch.matmul(flip_z, rmatrix)
        frag1_pos_p = torch.matmul(rmatrix, frag_pos.transpose(1,2)).transpose(1,2)
        return frag1_pos_p, rmatrix

    def rotation_matrix(self, vec1, vec2):
        """
        Construct rotation matrix based on 2 vectors

        Parameters
        -----------
        vec1: tensor
            vector along 1 component
        vec2: tensor
            vector along 2nd component

        Returns
        -----------
        rmatrix: tensor (n_batch, 3, 3)
            Rotation matrix corresponding to a certain direction
        """
        c = (vec1 * vec2).sum(axis=1)[:, None, None]
        v = torch.cross(vec1, vec2, dim=1)
        s = torch.norm(v, dim=1)[:, None, None]
        kmat = torch.zeros((vec1.shape[0], 3, 3), device=self.device)
        kmat[:, 0, 1] -= v[:, 2]
        kmat[:, 0, 2] += v[:, 1]
        kmat[:, 1, 0] += v[:, 2]
        kmat[:, 1, 2] -= v[:, 0]
        kmat[:, 2, 0] -= v[:, 1]
        kmat[:, 2, 1] += v[:, 0]

        kmul = torch.matmul(kmat, kmat)
        r1 = torch.zeros_like(kmat)
        r1[:] += torch.eye(3, device=self.device)
        rmatrix = r1 + kmat + kmul * ((1 - c) / (s**2))
        return rmatrix

    def forward(self, inputs):
        """
        Perform rotate monomer for both monomers

        Parameters
        -----------
        inputs: dict
            Dictionary containing the system information

        Returns
        -----------
        inputs: dict
            Same dict but with added monomer positions and rotation matrices in standard orientation
        """
        monA = inputs["monA"]
        monB = inputs["monB"]
        positions = inputs["R"]
        frag1_pos = positions[:, monA]
        frag2_pos = positions[:, monB]
        mass = inputs["mass"]

        frag1_mass = mass[:, monA]
        frag1_pos_p, frag1_rmatrix = self.rotate_monomer(frag1_pos, frag1_mass)

        frag2_mass = mass[:, monB]
        frag2_pos_p, frag2_rmatrix = self.rotate_monomer(frag2_pos, frag2_mass)

        inputs["frag1_pos"] = frag1_pos_p
        inputs["frag1_rmatrix"] = frag1_rmatrix
        inputs["frag2_pos"] = frag2_pos_p
        inputs["frag2_rmatrix"] = frag2_rmatrix
        return inputs

class RotateOrbital(nn.Module):
    """
    Rotate orbital back to initial position
    """
    def __init__(self, device="cuda"):
        """
        Parameters
        -----------
        device: str
            Whether to perform operations on cpu or cuda
        """
        super(RotateOrbital, self).__init__()
        self.device = torch.device(device)
        #For the DZ* basis, you will have this number of coefficients for carbon
        #Totally hard-coded in for now
        self.num_coeffs_carbon = 15
        self.s_orb = 4
        self.p2_orb = 7
        self.p3_orb = 10

    def rotation_matrix_D(self, rmatrix):
        #Perform necessary rotations for D matrix
        rmatrix = torch.reshape(rmatrix, (rmatrix.shape[0], -1))
        #Individual components of the standard rotation matrix
        a = rmatrix[:, 0, None]
        b = rmatrix[:, 1, None]
        c = rmatrix[:, 2, None]
        d = rmatrix[:, 3, None]
        e = rmatrix[:, 4, None]
        f = rmatrix[:, 5, None]
        g = rmatrix[:, 6, None]
        h = rmatrix[:, 7, None]
        i = rmatrix[:, 8, None]
        #These rotations are all taken from some other code in the group
        d11 = (torch.pow(a, 2) - torch.pow(d, 2) - torch.pow(b, 2) + torch.pow(e, 2)) / 2
        d12 = (a * g) - (b * h)
        d13 = (2 * torch.pow(g, 2) - torch.pow(d, 2) - torch.pow(a, 2) - 2 * torch.pow(h, 2) + \
                torch.pow(e, 2) + torch.pow(b, 2)) / 12**(1/2)
        d14 = (d * g) - (e * h)
        d15 = (a * d) - (b * e)

        d21 = (a * c) - (d * f)
        d22 = (a * i) + (c * g)
        d23 = ((2 * g * i) - (d * f) - (a * c)) / 3**(1/2)
        d24 = (d * i) + (f * g)
        d25 = (a * f) + (c * d)

        d31 = (2 * torch.pow(c, 2) - torch.pow(b, 2) - torch.pow(a, 2) - 2 * torch.pow(f, 2) + torch.pow(e, 2) + \
                torch.pow(d, 2)) / 12**(1/2)
        d32 = ((2 * c * i) - (b * h) - (a * g))/ 3**(1/2)
        d33 = (4 * torch.pow(i, 2) - 2 * (torch.pow(h, 2) + torch.pow(g, 2) + torch.pow(f, 2) + torch.pow(c, 2)) + torch.pow(e, 2) + \
                torch.pow(d, 2) + torch.pow(b, 2) + torch.pow(a, 2)) / 6
        d34 = ((2 * i * f) - (e * h) - (d * g)) / 3**(1/2)
        d35 = ((2 * c * f) - (e * b) - (a * d)) / 3**(1/2)

        d41 = (b * c) - (e * f)
        d42 = (b * i) + (c * h)
        d43 = ((2 * h * i) - (e * f) - (c * b)) / 3**(1/2)
        d44 = (e * i) + (f * h)
        d45 = (b * f) + (c * e)

        d51 = (a * b) - (d * e)
        d52 = (a * h) + (b * g)
        d53 = ((2 * g * h) - (e * d) - (a * b)) / 3**(1/2)
        d54 = (d * h) + (e * g)
        d55 = (a * e) + (b * d)

        #Combine all the individual elements to one matrix
        d1 = torch.cat((d55, d45, d35, d25, d15), dim=1)
        d2 = torch.cat((d54, d44, d34, d24, d14), dim=1)
        d3 = torch.cat((d53, d43, d33, d23, d13), dim=1)
        d4 = torch.cat((d52, d42, d32, d22, d12), dim=1)
        d5 = torch.cat((d51, d41, d31, d21, d11), dim=1)
        rmatrix_D = torch.cat((d1, d2, d3, d4, d5), dim=1)
        rmatrix_D = rmatrix_D.reshape((rmatrix.shape[0], 5, 5))
        return rmatrix_D

    def rotate_carbon(self, rmatrix, rmatrix_D, atom_orbital):
        """
        Rotate p orbitals and d orbitals with the rotation matrix

        Parameters
        -----------
        rmatrix: tensor
            obtained from the StandardOrientation class
        rmatrix_D: tensor
            for d orbitals, obtained above
        atom_orbital: tensor
            The MO orbital

        Returns
        -----------
        rot: tensor
            The rotated atom orbital
        """
        s_orbitals = atom_orbital[:, :self.s_orb]
        p2_orbitals = atom_orbital[:, self.s_orb:self.p2_orb]
        p3_orbitals = atom_orbital[:, self.p2_orb:self.p3_orb]
        d3_orbitals = atom_orbital[:, self.p3_orb:self.num_coeffs_carbon]
        
        #Rotate components
        p2_orbitals = p2_orbitals[:, :, None]
        p2_orbitals = torch.matmul(rmatrix, p2_orbitals)
        p2_orbitals = p2_orbitals.squeeze(-1)
        p3_orbitals = p3_orbitals[:, :, None]
        p3_orbitals = torch.matmul(rmatrix, p3_orbitals)
        p3_orbitals = p3_orbitals.squeeze(-1)
        d3_orbitals = d3_orbitals[:, :, None]
        d3_orbitals = torch.matmul(rmatrix_D, d3_orbitals)
        d3_orbitals = d3_orbitals.squeeze(-1)
        rot = torch.cat((s_orbitals, p2_orbitals, p3_orbitals, d3_orbitals), dim=1)
        return rot

    def forward(self, orbital, rmatrix, ZA):
        """
        rotate orbital. Can only handle carbon

        Parameters
        -----------
        orbital: tensor
            corresponds to MO coefficients
        rmatrix: tensor
            from StandardOrientation, was used to rotate the molecule
        ZA: tensor
            atom elements

        Returns
        -----------
        orbital_rotate: tensor
            rotated orbital
        """
        rmatrix = rmatrix.transpose(2,1)
        rmatrixD = self.rotation_matrix_D(rmatrix)
        numC = torch.where(ZA[0] == 6)[0].shape[0]
        orbital_rotate = torch.zeros_like(orbital)
        #Rotate the orbitals corresponding to the carbons in the molecule
        for i in range(numC):
            atom_orbital = orbital[:, self.num_coeffs_carbon*i:self.num_coeffs_carbon*(i+1)]
            rot = self.rotate_carbon(rmatrix, rmatrixD, atom_orbital)
            orbital_rotate[:, self.num_coeffs_carbon*i:self.num_coeffs_carbon*(i+1)] += rot
        
        #Place the remaining orbitals corresponding to hydrogen in the orbital tensor
        orbital_rotate[:, self.num_coeffs_carbon*(i+1):] += orbital[:, self.num_coeffs_carbon*(i+1):]
        return orbital_rotate
