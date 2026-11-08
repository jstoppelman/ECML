import time
import math
import torch
from torch import nn
from torch.nn import functional as F

from torch_cluster import radius_graph
from torch_scatter import scatter

from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Linear, TensorProduct

import numpy as np


# ============================================================
# Utils
# ============================================================
def prod(x):
    """Compute the product of a sequence."""
    out = 1
    for a in x:
        out *= a
    return out


def ShiftedSoftPlus(x):
    return torch.nn.functional.softplus(x) - math.log(2.0)


def softplus_inverse(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x + torch.log(-torch.expm1(-x))


def get_nonlinear(nonlinear: str):
    if nonlinear.lower() == 'ssp':
        return ShiftedSoftPlus
    elif nonlinear.lower() == 'silu':
        return F.silu
    elif nonlinear.lower() == 'tanh':
        return torch.tanh
    elif nonlinear.lower() == 'abs':
        return torch.abs
    else:
        raise NotImplementedError


def get_feasible_irrep(irrep_in1, irrep_in2, cutoff_irrep_out, tp_mode="uvu"):
    irrep_mid = []
    instructions = []

    for i, (_, ir_in) in enumerate(irrep_in1):
        for j, (_, ir_edge) in enumerate(irrep_in2):
            for ir_out in ir_in * ir_edge:
                if ir_out in cutoff_irrep_out:
                    if (cutoff_irrep_out.count(ir_out), ir_out) not in irrep_mid:
                        k = len(irrep_mid)
                        irrep_mid.append((cutoff_irrep_out.count(ir_out), ir_out))
                    else:
                        k = irrep_mid.index((cutoff_irrep_out.count(ir_out), ir_out))
                    instructions.append((i, j, k, tp_mode, True))

    irrep_mid = o3.Irreps(irrep_mid)
    normalization_coefficients = []
    for ins in instructions:
        ins_dict = {
            'uvw': (irrep_in1[ins[0]].mul * irrep_in2[ins[1]].mul),
            'uvu': irrep_in2[ins[1]].mul,
            'uvv': irrep_in1[ins[0]].mul,
            'uuw': irrep_in1[ins[0]].mul,
            'uuu': 1,
            'uvuv': 1,
            'uvu<v': 1,
            'u<vw': irrep_in1[ins[0]].mul * (irrep_in2[ins[1]].mul - 1) // 2,
        }
        alpha = irrep_mid[ins[2]].ir.dim
        x = sum([ins_dict[ins[3]] for ins in instructions])
        if x > 0.0:
            alpha /= x
        normalization_coefficients += [math.sqrt(alpha)]

    irrep_mid, p, _ = irrep_mid.sort()
    instructions = [
        (i_in1, i_in2, p[i_out], mode, train, alpha)
        for (i_in1, i_in2, i_out, mode, train), alpha
        in zip(instructions, normalization_coefficients)
    ]
    return irrep_mid, instructions


def cutoff_function(x, cutoff):
    zeros = torch.zeros_like(x)
    x_ = torch.where(x < cutoff, x, zeros)
    return torch.where(x < cutoff, torch.exp(-x_**2 / ((cutoff - x_) * (cutoff + x_))), zeros)


# ============================================================
# Radial basis
# ============================================================
class ExponentialBernsteinRadialBasisFunctions(nn.Module):
    def __init__(self, num_basis_functions, cutoff, ini_alpha=0.5):
        super().__init__()
        self.num_basis_functions = num_basis_functions
        self.ini_alpha = ini_alpha

        logfactorial = np.zeros((num_basis_functions))
        for i in range(2, num_basis_functions):
            logfactorial[i] = logfactorial[i - 1] + np.log(i)
        v = np.arange(0, num_basis_functions)
        n = (num_basis_functions - 1) - v
        logbinomial = logfactorial[-1] - logfactorial[v] - logfactorial[n]

        self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float32))
        self.register_buffer('logc', torch.tensor(logbinomial, dtype=torch.float32))
        self.register_buffer('n', torch.tensor(n, dtype=torch.float32))
        self.register_buffer('v', torch.tensor(v, dtype=torch.float32))
        self.register_parameter('_alpha', nn.Parameter(torch.tensor(1.0, dtype=torch.float32)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self._alpha, softplus_inverse(self.ini_alpha))

    def forward(self, r):
        alpha = F.softplus(self._alpha)
        x = -alpha * r
        x = self.logc + self.n * x + self.v * torch.log(-torch.expm1(x))
        rbf = cutoff_function(r, self.cutoff) * torch.exp(x)
        return rbf


# ============================================================
# Gating
# ============================================================
class NormGate(torch.nn.Module):
    def __init__(self, irrep):
        super().__init__()
        self.irrep = o3.Irreps(irrep)
        self.norm = o3.Norm(self.irrep)

        num_mul, num_mul_wo_0 = 0, 0
        for mul, ir in self.irrep:
            num_mul += mul
            if ir.l != 0:
                num_mul_wo_0 += mul

        self.mul = o3.ElementwiseTensorProduct(
            self.irrep[1:], o3.Irreps(f"{num_mul_wo_0}x0e")
        )
        self.fc = nn.Sequential(
            nn.Linear(num_mul, num_mul),
            nn.SiLU(),
            nn.Linear(num_mul, num_mul),
        )

    def forward(self, x):
        norm_x = self.norm(x)[:, self.irrep.slices()[0].stop:]
        f0 = torch.cat([x[:, self.irrep.slices()[0]], norm_x], dim=-1)
        gates = self.fc(f0)
        gated = self.mul(
            x[:, self.irrep.slices()[0].stop:],
            gates[:, self.irrep.slices()[0].stop:],
        )
        x = torch.cat([gates[:, self.irrep.slices()[0]], gated], dim=-1)
        return x


class InnerProduct(torch.nn.Module):
    def __init__(self, irrep_in):
        super().__init__()
        self.irrep_in = o3.Irreps(irrep_in).simplify()
        irrep_out = o3.Irreps([(mul, "0e") for mul, _ in self.irrep_in])
        instr = [(i, i, i, "uuu", False, 1 / ir.dim) for i, (mul, ir) in enumerate(self.irrep_in)]
        self.tp = o3.TensorProduct(
            self.irrep_in, self.irrep_in, irrep_out, instr, irrep_normalization="component"
        )
        self.irrep_out = irrep_out.simplify()

    def forward(self, features_1, features_2):
        return self.tp(features_1, features_2)


# ============================================================
# Convolution
# ============================================================
class ConvLayer(torch.nn.Module):
    def __init__(
        self,
        irrep_in_node,
        irrep_hidden,
        irrep_out,
        sh_irrep,
        edge_attr_dim,
        node_attr_dim,
        invariant_layers=1,
        invariant_neurons=32,
        avg_num_neighbors=None,
        nonlinear='ssp',
        use_norm_gate=True,
        edge_wise=False,
    ):
        super().__init__()
        self.avg_num_neighbors = avg_num_neighbors
        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.edge_wise = edge_wise

        self.irrep_in_node = irrep_in_node if isinstance(irrep_in_node, o3.Irreps) else o3.Irreps(irrep_in_node)
        self.irrep_hidden = irrep_hidden if isinstance(irrep_hidden, o3.Irreps) else o3.Irreps(irrep_hidden)
        self.irrep_out = irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)
        self.sh_irrep = sh_irrep if isinstance(sh_irrep, o3.Irreps) else o3.Irreps(sh_irrep)
        self.nonlinear_layer = get_nonlinear(nonlinear)

        self.irrep_tp_out_node, instruction_node = get_feasible_irrep(
            self.irrep_in_node, self.sh_irrep, self.irrep_hidden, tp_mode='uvu'
        )

        self.tp_node = TensorProduct(
            self.irrep_in_node,
            self.sh_irrep,
            self.irrep_tp_out_node,
            instruction_node,
            shared_weights=False,
            internal_weights=False,
        )

        self.fc_node = FullyConnectedNet(
            [self.edge_attr_dim] + invariant_layers * [invariant_neurons] + [self.tp_node.weight_numel],
            self.nonlinear_layer,
        )

        num_mul = 0
        for mul, ir in self.irrep_in_node:
            num_mul += mul

        self.layer_l0 = FullyConnectedNet(
            [num_mul + self.irrep_in_node[0][0]] + invariant_layers * [invariant_neurons] + [self.tp_node.weight_numel],
            self.nonlinear_layer,
        )

        self.linear_out = Linear(
            irreps_in=self.irrep_tp_out_node,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )

        self.use_norm_gate = use_norm_gate
        self.norm_gate = NormGate(self.irrep_in_node)
        self.irrep_linear_out, _ = get_feasible_irrep(self.irrep_in_node, o3.Irreps("0e"), self.irrep_in_node)

        self.linear_node = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_linear_out,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )
        self.linear_node_pre = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_linear_out,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )
        self.inner_product = InnerProduct(self.irrep_in_node)

    def forward(self, data, x):
        edge_dst, edge_src = data.edge_index[0], data.edge_index[1]

        if self.use_norm_gate:
            pre_x = self.linear_node_pre(x)
            s0 = self.inner_product(pre_x[edge_dst], pre_x[edge_src])[:, self.irrep_in_node.slices()[0].stop:]
            s0 = torch.cat(
                [
                    pre_x[edge_dst][:, self.irrep_in_node.slices()[0]],
                    pre_x[edge_dst][:, self.irrep_in_node.slices()[0]],
                    s0,
                ],
                dim=-1,
            )
            x = self.norm_gate(x)
            x = self.linear_node(x)
        else:
            s0 = self.inner_product(x[edge_dst], x[edge_src])[:, self.irrep_in_node.slices()[0].stop:]
            s0 = torch.cat(
                [
                    x[edge_dst][:, self.irrep_in_node.slices()[0]],
                    x[edge_dst][:, self.irrep_in_node.slices()[0]],
                    s0,
                ],
                dim=-1,
            )

        self_x = x

        edge_features = self.tp_node(
            x[edge_src],
            data.edge_sh,
            self.fc_node(data.edge_attr) * self.layer_l0(s0),
        )

        if self.edge_wise:
            out = edge_features
        else:
            out = scatter(edge_features, edge_dst, dim=0, dim_size=len(x))

        if self.irrep_in_node == self.irrep_out:
            out = out + self_x

        out = self.linear_out(out)
        return out


class ConvNetLayer(torch.nn.Module):
    def __init__(
        self,
        irrep_in_node,
        irrep_hidden,
        irrep_out,
        sh_irrep,
        edge_attr_dim,
        node_attr_dim,
        resnet: bool = True,
        use_norm_gate=True,
        edge_wise=False,
    ):
        super().__init__()
        self.irrep_in_node = irrep_in_node if isinstance(irrep_in_node, o3.Irreps) else o3.Irreps(irrep_in_node)
        self.irrep_hidden = irrep_hidden if isinstance(irrep_hidden, o3.Irreps) else o3.Irreps(irrep_hidden)
        self.irrep_out = irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)
        self.sh_irrep = sh_irrep if isinstance(sh_irrep, o3.Irreps) else o3.Irreps(sh_irrep)

        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.resnet = resnet and self.irrep_in_node == self.irrep_out

        self.conv = ConvLayer(
            irrep_in_node=self.irrep_in_node,
            irrep_hidden=self.irrep_hidden,
            sh_irrep=self.sh_irrep,
            irrep_out=self.irrep_out,
            edge_attr_dim=self.edge_attr_dim,
            node_attr_dim=self.node_attr_dim,
            invariant_layers=1,
            invariant_neurons=32,
            avg_num_neighbors=None,
            nonlinear='ssp',
            use_norm_gate=use_norm_gate,
            edge_wise=edge_wise,
        )

    def forward(self, data, x):
        old_x = x
        x = self.conv(data, x)
        if self.resnet and self.irrep_out == self.irrep_in_node:
            x = old_x + x
        return x


# ============================================================
# Pair/Self layers (unchanged from your version)
# ============================================================
class PairNetLayer(torch.nn.Module):
    def __init__(
        self,
        irrep_in_node,
        irrep_bottle_hidden,
        irrep_out,
        sh_irrep,
        edge_attr_dim,
        node_attr_dim,
        resnet: bool = True,
        invariant_layers=1,
        invariant_neurons=8,
        nonlinear='ssp'
    ):
        super().__init__()
        self.invariant_layers = invariant_layers
        self.invariant_neurons = invariant_neurons
        self.irrep_in_node = irrep_in_node if isinstance(irrep_in_node, o3.Irreps) else o3.Irreps(irrep_in_node)
        self.irrep_bottle_hidden = irrep_bottle_hidden if isinstance(irrep_bottle_hidden, o3.Irreps) else o3.Irreps(irrep_bottle_hidden)
        self.irrep_out = irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)
        self.sh_irrep = sh_irrep if isinstance(sh_irrep, o3.Irreps) else o3.Irreps(sh_irrep)

        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.nonlinear_layer = get_nonlinear(nonlinear)

        self.irrep_tp_in_node, _ = get_feasible_irrep(self.irrep_in_node, o3.Irreps("0e"), self.irrep_bottle_hidden)
        self.irrep_tp_out_node_pair, instruction_node_pair = get_feasible_irrep(
            self.irrep_tp_in_node, self.irrep_tp_in_node, self.irrep_bottle_hidden, tp_mode='uuu'
        )

        self.linear_node_pair_inner = Linear(self.irrep_in_node, self.irrep_in_node, internal_weights=True, shared_weights=True, biases=True)
        self.linear_node_pair_n = Linear(self.irrep_in_node, self.irrep_in_node, internal_weights=True, shared_weights=True, biases=True)

        self.tp_node_pair = TensorProduct(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_tp_out_node_pair,
            instruction_node_pair,
            shared_weights=False,
            internal_weights=False,
        )

        self.fc_node_pair = FullyConnectedNet(
            [self.edge_attr_dim] + invariant_layers * [invariant_neurons] + [self.tp_node_pair.weight_numel],
            self.nonlinear_layer,
        )

        self.linear_node_pair = Linear(
            irreps_in=self.irrep_tp_out_node_pair,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )

        self.resnet = (self.irrep_in_node == self.irrep_out and resnet)
        self.norm_gate_pre = NormGate(self.irrep_tp_out_node_pair)
        self.norm_gate = NormGate(self.irrep_tp_out_node_pair)

        self.inner_product = InnerProduct(self.irrep_in_node)

        num_mul = 0
        for mul, ir in self.irrep_in_node:
            num_mul += mul

        self.fc = nn.Sequential(
            nn.Linear(self.irrep_in_node[0][0] + num_mul, self.irrep_in_node[0][0]),
            nn.SiLU(),
            nn.Linear(self.irrep_in_node[0][0], self.tp_node_pair.weight_numel),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data, node_attr, node_pair_attr=None):
        dst, src = data.full_edge_index
        node_attr_0 = self.linear_node_pair_inner(node_attr)
        s0 = self.inner_product(node_attr_0[dst], node_attr_0[src])[:, self.irrep_in_node.slices()[0].stop:]
        s0 = torch.cat(
            [node_attr_0[dst][:, self.irrep_in_node.slices()[0]],
             node_attr_0[src][:, self.irrep_in_node.slices()[0]],
             s0],
            dim=-1
        )

        node_attr = self.norm_gate_pre(node_attr)
        node_attr = self.linear_node_pair_n(node_attr)

        node_pair = self.tp_node_pair(
            node_attr[src],
            node_attr[dst],
            self.fc_node_pair(data.full_edge_attr) * self.fc(s0),
        )

        node_pair = self.norm_gate(node_pair)
        node_pair = self.linear_node_pair(node_pair)

        if self.resnet and node_pair_attr is not None:
            node_pair = node_pair + node_pair_attr
        return node_pair


class SelfNetLayer(torch.nn.Module):
    def __init__(
        self,
        irrep_in_node,
        irrep_bottle_hidden,
        irrep_out,
        sh_irrep,
        edge_attr_dim,
        node_attr_dim,
        resnet: bool = True,
        nonlinear='ssp'
    ):
        super().__init__()
        self.sh_irrep = sh_irrep
        self.irrep_in_node = irrep_in_node if isinstance(irrep_in_node, o3.Irreps) else o3.Irreps(irrep_in_node)
        self.irrep_bottle_hidden = irrep_bottle_hidden if isinstance(irrep_bottle_hidden, o3.Irreps) else o3.Irreps(irrep_bottle_hidden)
        self.irrep_out = irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)

        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.resnet = resnet
        self.nonlinear_layer = get_nonlinear(nonlinear)

        self.irrep_tp_in_node, _ = get_feasible_irrep(self.irrep_in_node, o3.Irreps("0e"), self.irrep_bottle_hidden)
        self.irrep_tp_out_node, instruction_node = get_feasible_irrep(
            self.irrep_tp_in_node, self.irrep_tp_in_node, self.irrep_bottle_hidden, tp_mode='uuu'
        )

        self.linear_node_1 = Linear(self.irrep_in_node, self.irrep_in_node, internal_weights=True, shared_weights=True, biases=True)
        self.linear_node_2 = Linear(self.irrep_in_node, self.irrep_in_node, internal_weights=True, shared_weights=True, biases=True)

        self.tp = TensorProduct(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_tp_out_node,
            instruction_node,
            shared_weights=True,
            internal_weights=True,
        )
        self.norm_gate_1 = NormGate(self.irrep_in_node)
        self.norm_gate_2 = NormGate(self.irrep_in_node)
        self.norm_gate = NormGate(self.irrep_out)
        self.linear_node_3 = Linear(self.irrep_tp_out_node, self.irrep_out, internal_weights=True, shared_weights=True, biases=True)

    def forward(self, data, x, old_fii):
        old_x = x
        xl = self.norm_gate_1(x)
        xl = self.linear_node_1(xl)
        xr = self.norm_gate_2(x)
        xr = self.linear_node_2(xr)
        x = self.tp(xl, xr)
        if self.resnet:
            x = x + old_x
        x = self.norm_gate(x)
        x = self.linear_node_3(x)
        if self.resnet and old_fii is not None:
            x = old_fii + x
        return x

    @property
    def device(self):
        return next(self.parameters()).device


# ============================================================
# Expansion (unchanged)
# ============================================================
class Expansion(nn.Module):
    def __init__(self, irrep_in, irrep_out_1, irrep_out_2):
        super().__init__()
        self.irrep_in = o3.Irreps(irrep_in)
        self.irrep_out_1 = o3.Irreps(irrep_out_1)
        self.irrep_out_2 = o3.Irreps(irrep_out_2)
        self.instructions = self.get_expansion_path(self.irrep_in, self.irrep_out_1, self.irrep_out_2)
        self.num_path_weight = sum(prod(ins[-1]) for ins in self.instructions if ins[3])
        self.num_bias = sum([prod(ins[-1][1:]) for ins in self.instructions if ins[0] == 0])
        if self.num_path_weight > 0:
            self.weights = nn.Parameter(torch.rand(self.num_path_weight + self.num_bias))
        self.num_weights = self.num_path_weight + self.num_bias

    def forward(self, x_in, weights=None, bias_weights=None):
        batch_num = x_in.shape[0]
        if len(self.irrep_in) == 1:
            x_in_s = [x_in.reshape(batch_num, self.irrep_in[0].mul, self.irrep_in[0].ir.dim)]
        else:
            x_in_s = [
                x_in[:, i].reshape(batch_num, mul_ir.mul, mul_ir.ir.dim)
                for i, mul_ir in zip(self.irrep_in.slices(), self.irrep_in)
            ]

        outputs = {}
        flat_weight_index = 0
        bias_weight_index = 0
        for ins in self.instructions:
            mul_ir_in = self.irrep_in[ins[0]]
            x1 = x_in_s[ins[0]].reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim)

            w3j_matrix = o3.wigner_3j(ins[1], ins[2], ins[0]).to(self.device).type(x1.type())

            if ins[3] is True or weights is not None:
                if weights is None:
                    weight = self.weights[flat_weight_index:flat_weight_index + prod(ins[-1])].reshape(ins[-1])
                    result = torch.einsum("wuv, ijk, bwk-> buivj", weight, w3j_matrix, x1) / mul_ir_in.mul
                else:
                    weight = weights[:, flat_weight_index:flat_weight_index + prod(ins[-1])].reshape([-1] + ins[-1])
                    result = torch.einsum("bwuv, bwk-> buvk", weight, x1)
                    if ins[0] == 0 and bias_weights is not None:
                        bias_weight = bias_weights[:, bias_weight_index:bias_weight_index + prod(ins[-1][1:])].reshape(
                            [-1] + ins[-1][1:]
                        )
                        bias_weight_index += prod(ins[-1][1:])
                        result = result + bias_weight.unsqueeze(-1)
                    result = torch.einsum("ijk, buvk->buivj", w3j_matrix, result) / mul_ir_in.mul
                flat_weight_index += prod(ins[-1])
            else:
                result = torch.einsum(
                    "uvw, ijk, bwk-> buivj",
                    torch.ones(ins[-1]).type(x1.type()).to(self.device),
                    w3j_matrix,
                    x1,
                )

            result = result.reshape(batch_num, self.irrep_out_1[ins[1]].dim, self.irrep_out_2[ins[2]].dim)
            key = (ins[1], ins[2])
            outputs[key] = outputs.get(key, 0.0) + result

        rows = []
        for i in range(len(self.irrep_out_1)):
            blocks = []
            for j in range(len(self.irrep_out_2)):
                if (i, j) not in outputs:
                    blocks.append(
                        torch.zeros(
                            (x_in.shape[0], self.irrep_out_1[i].dim, self.irrep_out_2[j].dim),
                            device=x_in.device,
                            dtype=x_in.dtype,
                        )
                    )
                else:
                    blocks.append(outputs[(i, j)])
            rows.append(torch.cat(blocks, dim=-1))
        output = torch.cat(rows, dim=-2)
        return output

    def get_expansion_path(self, irrep_in, irrep_out_1, irrep_out_2):
        instructions = []
        for i, (num_in, ir_in) in enumerate(irrep_in):
            for j, (num_out1, ir_out1) in enumerate(irrep_out_1):
                for k, (num_out2, ir_out2) in enumerate(irrep_out_2):
                    if ir_in in ir_out1 * ir_out2:
                        instructions.append([i, j, k, True, 1.0, [num_in, num_out1, num_out2]])
        return instructions

    @property
    def device(self):
        return next(self.parameters()).device

    def __repr__(self):
        return (
            f"{self.irrep_in} -> {self.irrep_out_1}x{self.irrep_out_2} "
            f"and bias {self.num_bias} with parameters {self.num_path_weight}"
        )


# ============================================================
# QHNet (DZP-adapted for H/C; spherical d=5)
# ============================================================
class QHNet(nn.Module):
    """
    DZP version (H/C):
      - H: [2s]         -> Irreps("2x0e")   dim=2
      - C: [4s,2p,1d]   -> Irreps("4x0e + 2x1e + 1x2e") dim=15   (spherical d => 5)
    Implementation strategy:
      - Use MAX_AO = 15 for all atoms (pad), then slice with orbital_mask per element Z.
    """

    def __init__(
        self,
        in_node_features=1,
        sh_lmax=4,
        hidden_size=128,
        bottle_hidden_size=32,
        num_gnn_layers=5,
        max_radius=12,
        num_nodes=10,
        radius_embed_dim=32,
    ):
        super().__init__()

        # -------------------------
        # DZP AO layout (spherical)
        # -------------------------
        self.TARGET_IRREPS = o3.Irreps("4x0e + 2x1e + 1x2e")  # C DZP
        self.MAX_AO = self.TARGET_IRREPS.dim  # 15

        self.order = sh_lmax
        self.sh_irrep = o3.Irreps.spherical_harmonics(lmax=self.order)

        self.hs = hidden_size
        self.hbs = bottle_hidden_size
        self.radius_embed_dim = radius_embed_dim
        self.max_radius = max_radius
        self.num_gnn_layers = num_gnn_layers

        # Embedding (assumes data.atoms are integer categories in [0, num_nodes-1])
        self.node_embedding = nn.Embedding(num_nodes, self.hs)

        self.hidden_irrep = o3.Irreps(f"{self.hs}x0e + {self.hs}x1o + {self.hs}x2e + {self.hs}x3o + {self.hs}x4e")
        self.hidden_bottle_irrep = o3.Irreps(f"{self.hbs}x0e + {self.hbs}x1o + {self.hbs}x2e + {self.hbs}x3o + {self.hbs}x4e")

        self.hidden_irrep_base = o3.Irreps(f"{self.hs}x0e + {self.hs}x1e + {self.hs}x2e + {self.hs}x3e + {self.hs}x4e")
        self.hidden_bottle_irrep_base = o3.Irreps(f"{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e")

        self.input_irrep = o3.Irreps(f"{self.hs}x0e")
        self.distance_expansion = ExponentialBernsteinRadialBasisFunctions(self.radius_embed_dim, self.max_radius)

        self.e3_gnn_layer = nn.ModuleList()
        self.e3_gnn_node_pair_layer = nn.ModuleList()
        self.e3_gnn_node_layer = nn.ModuleList()

        self.start_layer = 2
        for i in range(self.num_gnn_layers):
            input_irrep = self.input_irrep if i == 0 else self.hidden_irrep
            self.e3_gnn_layer.append(
                ConvNetLayer(
                    irrep_in_node=input_irrep,
                    irrep_hidden=self.hidden_irrep,
                    irrep_out=self.hidden_irrep,
                    edge_attr_dim=self.radius_embed_dim,
                    node_attr_dim=self.hs,
                    sh_irrep=self.sh_irrep,
                    resnet=True,
                    use_norm_gate=True if i != 0 else False,
                )
            )

            if i > self.start_layer:
                self.e3_gnn_node_layer.append(
                    SelfNetLayer(
                        irrep_in_node=self.hidden_irrep_base,
                        irrep_bottle_hidden=self.hidden_irrep_base,
                        irrep_out=self.hidden_irrep_base,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        resnet=True,
                    )
                )

                self.e3_gnn_node_pair_layer.append(
                    PairNetLayer(
                        irrep_in_node=self.hidden_irrep_base,
                        irrep_bottle_hidden=self.hidden_irrep_base,
                        irrep_out=self.hidden_irrep_base,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        invariant_layers=1,
                        invariant_neurons=self.hs,
                        resnet=True,
                    )
                )

        # -------------------------
        # Expansion heads (use MAX basis = C basis = 15)
        # -------------------------
        self.expand_ii, self.expand_ij = nn.ModuleDict(), nn.ModuleDict()
        self.fc_ii, self.fc_ij = nn.ModuleDict(), nn.ModuleDict()
        self.fc_ii_bias, self.fc_ij_bias = nn.ModuleDict(), nn.ModuleDict()

        for name in {"hamiltonian"}:
            input_expand_ii = o3.Irreps(f"{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e")

            self.expand_ii[name] = Expansion(input_expand_ii, self.TARGET_IRREPS, self.TARGET_IRREPS)
            self.fc_ii[name] = nn.Sequential(nn.Linear(self.hs, self.hs), nn.SiLU(), nn.Linear(self.hs, self.expand_ii[name].num_path_weight))
            self.fc_ii_bias[name] = nn.Sequential(nn.Linear(self.hs, self.hs), nn.SiLU(), nn.Linear(self.hs, self.expand_ii[name].num_bias))

            self.expand_ij[name] = Expansion(
                o3.Irreps(f"{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e"),
                self.TARGET_IRREPS,
                self.TARGET_IRREPS,
            )
            
            self.fc_ij[name] = nn.Sequential(nn.Linear(self.hs * 2, self.hs), nn.SiLU(), nn.Linear(self.hs, self.expand_ij[name].num_path_weight))
            self.fc_ij_bias[name] = nn.Sequential(nn.Linear(self.hs * 2, self.hs), nn.SiLU(), nn.Linear(self.hs, self.expand_ij[name].num_bias))

        self.output_ii = Linear(self.hidden_irrep, self.hidden_bottle_irrep)
        self.output_ij = Linear(self.hidden_irrep, self.hidden_bottle_irrep)

        # orbital_mask set later in set(device)
        self.orbital_mask = None

    # -------------------------
    # DZP mask (H/C) - spherical d=5
    # -------------------------
    def get_orbital_mask(self):
        """
        IMPORTANT: `data.atoms` must contain atomic numbers Z (H=1, C=6, ...)
        If your dataset uses remapped indices (e.g. 0/1), you must adapt here accordingly.
        """
        orbital_mask = {}
        orbital_mask[1] = torch.arange(2, dtype=torch.long)    # H: 2s -> 2 AO
        orbital_mask[6] = torch.arange(15, dtype=torch.long)   # C: 4s2p1d -> 15 AO
        orbital_mask[7] = torch.arange(10, dtype=torch.long)     # N: 10 (4s2p)
        orbital_mask[8] = torch.arange(10, dtype=torch.long)     # O: 10
        orbital_mask[9] = torch.arange(10, dtype=torch.long)
        return orbital_mask

    def set(self, device):
        self.to(device)
        self.orbital_mask = self.get_orbital_mask()
        for k in list(self.orbital_mask.keys()):
            self.orbital_mask[k] = self.orbital_mask[k].to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ============================================================
    # Forward
    # ============================================================
    def forward(self, data, keep_blocks=True):
        node_attr, edge_index, rbf_new, edge_sh, _ = self.build_graph(data, self.max_radius)
        node_attr = self.node_embedding(node_attr)
        data.node_attr, data.edge_index, data.edge_attr, data.edge_sh = node_attr, edge_index, rbf_new, edge_sh

        _, full_edge_index, full_edge_attr, full_edge_sh, transpose_edge_index = self.build_graph(data, max_radius=10000)
        data.full_edge_index, data.full_edge_attr, data.full_edge_sh = full_edge_index, full_edge_attr, full_edge_sh

        full_dst, full_src = data.full_edge_index

        tic = time.time()
        fii = None
        fij = None

        for layer_idx, layer in enumerate(self.e3_gnn_layer):
            node_attr = layer(data, node_attr)
            if layer_idx > self.start_layer:
                fii = self.e3_gnn_node_layer[layer_idx - self.start_layer - 1](data, node_attr, fii)
                fij = self.e3_gnn_node_pair_layer[layer_idx - self.start_layer - 1](data, node_attr, fij)

        fii = self.output_ii(fii)
        fij = self.output_ij(fij)

        # Diagonal blocks: (N_atoms, MAX_AO, MAX_AO)
        hamiltonian_diagonal_matrix = self.expand_ii['hamiltonian'](
            fii,
            self.fc_ii['hamiltonian'](data.node_attr),
            self.fc_ii_bias['hamiltonian'](data.node_attr),
        )

        # Off-diagonal blocks: (N_full_edges, MAX_AO, MAX_AO)
        node_pair_embedding = torch.cat([data.node_attr[full_dst], data.node_attr[full_src]], dim=-1)
        hamiltonian_non_diagonal_matrix = self.expand_ij['hamiltonian'](
            fij,
            self.fc_ij['hamiltonian'](node_pair_embedding),
            self.fc_ij_bias['hamiltonian'](node_pair_embedding),
        )

        if keep_blocks is False:
            hamiltonian_matrix = self.build_final_matrix(data, hamiltonian_diagonal_matrix, hamiltonian_non_diagonal_matrix)
            hamiltonian_matrix = hamiltonian_matrix + hamiltonian_matrix.transpose(-1, -2)
            return {"hamiltonian": hamiltonian_matrix, "duration": torch.tensor([time.time() - tic], device=self.device)}
        else:
            ret_hamiltonian_diagonal_matrix = hamiltonian_diagonal_matrix + hamiltonian_diagonal_matrix.transpose(-1, -2)
            ret_hamiltonian_non_diagonal_matrix = hamiltonian_non_diagonal_matrix + \
                hamiltonian_non_diagonal_matrix[transpose_edge_index].transpose(-1, -2)
            return {
                "hamiltonian_diagonal_blocks": ret_hamiltonian_diagonal_matrix,
                "hamiltonian_non_diagonal_blocks": ret_hamiltonian_non_diagonal_matrix,
            }

    # ============================================================
    # Graph builder
    # ============================================================
    def build_graph(self, data, max_radius, edge_index=None):
        node_attr = data.atoms.squeeze()

        if edge_index is None:
            radius_edges = radius_graph(data.pos, max_radius, data.batch, max_num_neighbors=data.num_nodes)
        else:
            radius_edges = data.full_edge_index

        dst, src = radius_edges
        edge_vec = data.pos[dst.long()] - data.pos[src.long()]
        rbf = self.distance_expansion(edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(data.pos.type())

        edge_sh = o3.spherical_harmonics(
            self.sh_irrep,
            edge_vec[:, [1, 2, 0]],
            normalize=True,
            normalization='component',
        ).type(data.pos.type())

        # Transpose index mapping (assumes full graph edges in ordered (i,j) excluding i==j)
        start_edge_index = 0
        all_transpose_index = []
        for graph_idx in range(data.ptr.shape[0] - 1):
            num_nodes = data.ptr[graph_idx + 1] - data.ptr[graph_idx]
            graph_edge_index = radius_edges[:, start_edge_index:start_edge_index + num_nodes * (num_nodes - 1)]
            sub_graph_edge_index = graph_edge_index - data.ptr[graph_idx]
            bias = (sub_graph_edge_index[0] < sub_graph_edge_index[1]).type(torch.int)
            transpose_index = sub_graph_edge_index[0] * (num_nodes - 1) + sub_graph_edge_index[1] - bias
            transpose_index = transpose_index + start_edge_index
            all_transpose_index.append(transpose_index)
            start_edge_index = start_edge_index + num_nodes * (num_nodes - 1)

        return node_attr, radius_edges, rbf, edge_sh, torch.cat(all_transpose_index, dim=-1)

    # ============================================================
    # Assemble full matrix from blocks (variable AO per element via masks)
    # ============================================================
    def build_final_matrix_indices(self, data, diagonal_matrix, non_diagonal_matrix):
        """
        Assemble full Hamiltonian for each graph in batch.

        Requires:
          - self.orbital_mask[Z] gives indices inside [0..MAX_AO-1] for that element's AO subset.
          - data.atoms contains atomic numbers Z.
        """
        final_matrix = []
        final_matrix_index = []

        total_dim = 0
        for a in data.atoms:
            total_dim += self.orbital_mask[a.item()].shape[0]
        self.total_dim = total_dim

        diagonal_index = []
        offdiagonal_index = []
        offdiagonal_matrix_index = []
        start_index_row, start_index_col = 0, 0
        start_diag_index = 0
        start_offdiag_index = 0
        dst, src = data.full_edge_index
        for graph_idx in range(data.ptr.shape[0] - 1):
            matrix_block_col = []
            matrix_block_index = []
            for src_idx in range(data.ptr[graph_idx], data.ptr[graph_idx + 1]):
                matrix_col = []
                matrix_col_index = []
                for dst_idx in range(data.ptr[graph_idx], data.ptr[graph_idx + 1]):
                    z_dst = int(data.atoms[dst_idx].item())
                    z_src = int(data.atoms[src_idx].item())
                    mask_dst = self.orbital_mask[z_dst]
                    mask_src = self.orbital_mask[z_src]

                    if src_idx == dst_idx:
                        # diagonal block: pick AO subset for that element
                        matrix_col.append(
                            diagonal_matrix[src_idx]
                            .index_select(-2, mask_dst)
                            .index_select(-1, mask_src)
                        )
                        index_sel = torch.cumsum(torch.ones_like(diagonal_matrix[src_idx].flatten()), 0)
                        index_sel = index_sel.reshape(diagonal_matrix[src_idx].shape)
                        index_sel = index_sel.index_select(-2, mask_dst).index_select(-1, mask_src) - 1 + start_diag_index
                        diagonal_index.append(index_sel.flatten())
                        start_diag_index += diagonal_matrix[src_idx].flatten().shape[0]
                        matrix_col_index.append(
                            torch.ones_like(diagonal_matrix[src_idx]
                            .index_select(-2, mask_dst)
                            .index_select(-1, mask_src))
                        )
                        start_index_row += mask_dst.shape[0]
                    else:
                        # find the corresponding edge block
                        mask1 = (src == src_idx)
                        mask2 = (dst == dst_idx)
                        index = torch.where(mask1 & mask2)[0].item()

                        matrix_col.append(
                            non_diagonal_matrix[index]
                            .index_select(-2, mask_dst)
                            .index_select(-1, mask_src)
                        )
                        
                        index_sel = torch.cumsum(torch.ones_like(non_diagonal_matrix[index].flatten()), 0)
                        index_sel = index_sel.reshape(non_diagonal_matrix[index].shape)
                        index_sel = index_sel.index_select(-2, mask_dst).index_select(-1, mask_src) - 1 + start_offdiag_index
                        offdiagonal_index.append(index_sel.flatten())
                        start_offdiag_index += non_diagonal_matrix[index].flatten().shape[0]  

                        matrix_col_index.append(
                            -torch.ones_like(non_diagonal_matrix[index]
                            .index_select(-2, mask_dst)
                            .index_select(-1, mask_src))
                        )

                        row_index_tmp = start_index_row
                        col_index_tmp = start_index_col
                        for r in range(mask_dst.shape[0]):
                            start_d1 = row_index_tmp * total_dim + col_index_tmp
                            end_d1 = start_d1 + mask_src.shape[0]
                            offdiagonal_matrix_index.append(torch.arange(start_d1, end_d1).to("cuda"))
                            row_index_tmp += 1
                        
                        start_index_row += mask_dst.shape[0]
                
                matrix_block_col.append(torch.cat(matrix_col, dim=-2))
                matrix_block_index.append(torch.cat(matrix_col_index, dim=-2))
                start_index_row = 0
                start_index_col += matrix_block_col[-1].shape[-1]
            
            final_matrix.append(torch.cat(matrix_block_col, dim=-1))
            final_matrix_index.append(torch.cat(matrix_block_index, dim=-1))
        
        self.diagonal_index = torch.cat(diagonal_index, dim=0).to("cuda")
        self.offdiagonal_index = torch.cat(offdiagonal_index, dim=0).to("cuda")
        matrix_index = torch.stack(final_matrix_index, dim=0)

        diagonal_values = diagonal_matrix.flatten()[self.diagonal_index.flatten().long()]
        self.diagonal_matrix_indices = torch.where(matrix_index.flatten() == 1)[0].to("cuda")
        offdiagonal_values = non_diagonal_matrix.flatten()[self.offdiagonal_index.flatten().long()]

        self.offdiagonal_matrix_index = torch.cat(offdiagonal_matrix_index, dim=0).to("cuda")

        return torch.stack(final_matrix, dim=0)

    def build_final_matrix(self, data, diagonal_matrix, non_diagonal_matrix):
        """
        Assemble full Hamiltonian for each graph in batch.

        Requires:
          - self.orbital_mask[Z] gives indices inside [0..MAX_AO-1] for that element's AO subset.
          - data.atoms contains atomic numbers Z.
        """
        final_matrix = []
        dst, src = data.full_edge_index

        for graph_idx in range(data.ptr.shape[0] - 1):
            matrix_block_col = []
            for src_idx in range(data.ptr[graph_idx], data.ptr[graph_idx + 1]):
                matrix_col = []
                for dst_idx in range(data.ptr[graph_idx], data.ptr[graph_idx + 1]):
                    z_dst = int(data.atoms[dst_idx].item())
                    z_src = int(data.atoms[src_idx].item())
                    mask_dst = self.orbital_mask[z_dst]
                    mask_src = self.orbital_mask[z_src]

                    if src_idx == dst_idx:
                        # diagonal block: pick AO subset for that element
                        matrix_col.append(
                            diagonal_matrix[src_idx]
                            .index_select(-2, mask_dst)
                            .index_select(-1, mask_src)
                        )
                    else:
                        # find the corresponding edge block
                        mask1 = (src == src_idx)
                        mask2 = (dst == dst_idx)
                        index = torch.where(mask1 & mask2)[0].item()

                        matrix_col.append(
                            non_diagonal_matrix[index]
                            .index_select(-2, mask_dst)
                            .index_select(-1, mask_src)
                        )

                matrix_block_col.append(torch.cat(matrix_col, dim=-2))
            final_matrix.append(torch.cat(matrix_block_col, dim=-1))

        return torch.stack(final_matrix, dim=0)


    def build_final_matrix_from_index(self, data, diagonal_matrix, non_diagonal_matrix):
        num_batch = torch.unique(data.batch).shape[0]
        final_matrix = torch.zeros((num_batch, self.total_dim, self.total_dim)).to("cuda")

        diagonal_matrix = diagonal_matrix.reshape(num_batch, -1)
        diagonal_matrix_size = torch.tensor([diagonal_matrix.shape[-1]]).to("cuda")
        diagonal_matrix_size = torch.repeat_interleave(diagonal_matrix_size, num_batch)
        diagonal_matrix_size = torch.cumsum(diagonal_matrix_size, 0) - diagonal_matrix.shape[-1]
        diagonal_index = torch.tile(self.diagonal_index, (num_batch,)).long()
      
        diagonal_index = diagonal_index.reshape(num_batch, -1)
        diagonal_index += diagonal_matrix_size[:, None]
        diagonal_index = diagonal_index.flatten()
        diagonal_values = diagonal_matrix.flatten()[diagonal_index.long()]

        diagonal_matrix_indices = torch.tile(self.diagonal_matrix_indices, (num_batch,)).long()
        matrix_size = torch.tensor([self.total_dim*self.total_dim]).to("cuda")
        matrix_size = torch.repeat_interleave(matrix_size, num_batch)
        matrix_size = torch.cumsum(matrix_size, 0) - self.total_dim*self.total_dim
        diagonal_matrix_indices = diagonal_matrix_indices.reshape(num_batch, -1)
        diagonal_matrix_indices += matrix_size[:, None]
        diagonal_matrix_indices = diagonal_matrix_indices.flatten()
        
        final_matrix = final_matrix.flatten()
        final_matrix.index_add_(0, diagonal_matrix_indices, diagonal_values)

        non_diagonal_matrix = non_diagonal_matrix.reshape(num_batch, -1)
        non_diagonal_matrix_size = torch.tensor([non_diagonal_matrix.shape[-1]]).to("cuda")
        non_diagonal_matrix_size = torch.repeat_interleave(non_diagonal_matrix_size, num_batch)
        non_diagonal_matrix_size = torch.cumsum(non_diagonal_matrix_size, 0) - non_diagonal_matrix.shape[-1]
        non_diagonal_index = torch.tile(self.offdiagonal_index, (num_batch,)).long()

        non_diagonal_index = non_diagonal_index.reshape(num_batch, -1)
        non_diagonal_index += non_diagonal_matrix_size[:, None]
        non_diagonal_index = non_diagonal_index.flatten()
        non_diagonal_values = non_diagonal_matrix.flatten()[non_diagonal_index.long()]

        non_diagonal_matrix_indices = torch.tile(self.offdiagonal_matrix_index, (num_batch,)).long()
        matrix_size = torch.tensor([self.total_dim*self.total_dim]).to("cuda")
        matrix_size = torch.repeat_interleave(matrix_size, num_batch)
        matrix_size = torch.cumsum(matrix_size, 0) - self.total_dim*self.total_dim
        non_diagonal_matrix_indices = non_diagonal_matrix_indices.reshape(num_batch, -1)
        non_diagonal_matrix_indices += matrix_size[:, None]
        non_diagonal_matrix_indices = non_diagonal_matrix_indices.flatten()
        final_matrix.index_add_(0, non_diagonal_matrix_indices, non_diagonal_values)
        
        return final_matrix.reshape(num_batch, self.total_dim, self.total_dim)
    
    # ============================================================
    # Split full matrix into padded per-atom/per-edge blocks (MAX_AO=15)
    # ============================================================
    def split_matrix(self, data):
        """
        data.matrix: (batch, dim, dim) or flattened; will be reshaped to (B, dim, dim).
        This function pads into (N_atoms, MAX_AO, MAX_AO) and (E_edges, MAX_AO, MAX_AO) using masks.

        NOTE:
          - This assumes `data.matrix` uses the same AO ordering convention as your masks.
        """
        diagonal_matrix = torch.zeros(
            data.atoms.shape[0], self.MAX_AO, self.MAX_AO,
            device=self.device, dtype=data.pos.dtype
        )
        non_diagonal_matrix = torch.zeros(
            data.edge_index.shape[1], self.MAX_AO, self.MAX_AO,
            device=self.device, dtype=data.pos.dtype
        )

        data.matrix = data.matrix.reshape(len(data.ptr) - 1, data.matrix.shape[-1], data.matrix.shape[-1])

        num_atoms = 0
        num_edges = 0

        for graph_idx in range(data.ptr.shape[0] - 1):
            # slices in the full matrix per atom, based on each atom's AO count
            slices = [0]
            for atom_z in data.atoms[range(data.ptr[graph_idx], data.ptr[graph_idx + 1])]:
                z = int(atom_z.item())
                slices.append(slices[-1] + len(self.orbital_mask[z]))

            # diagonal blocks
            for node_idx_abs in range(data.ptr[graph_idx], data.ptr[graph_idx + 1]):
                node_idx = node_idx_abs - num_atoms
                z = int(data.atoms[node_idx].item())
                orb_mask = self.orbital_mask[z]

                r0, r1 = slices[node_idx], slices[node_idx + 1]
                block = data.matrix[graph_idx][r0:r1, r0:r1]
                diagonal_matrix[node_idx][orb_mask][:, orb_mask] = block

            # off-diagonal blocks (edges in data.edge_index)
            for edge_index_idx in range(num_edges, data.edge_index.shape[1]):
                dst, src = data.edge_index[:, edge_index_idx]
                if dst > data.ptr[graph_idx + 1] or src > data.ptr[graph_idx + 1]:
                    break
                num_edges += 1

                z_dst = int(data.atoms[dst].item())
                z_src = int(data.atoms[src].item())
                orb_mask_dst = self.orbital_mask[z_dst]
                orb_mask_src = self.orbital_mask[z_src]

                graph_dst = int(dst.item()) - num_atoms
                graph_src = int(src.item()) - num_atoms

                rd0, rd1 = slices[graph_dst], slices[graph_dst + 1]
                rs0, rs1 = slices[graph_src], slices[graph_src + 1]
                block = data.matrix[graph_idx][rd0:rd1, rs0:rs1]

                non_diagonal_matrix[edge_index_idx][orb_mask_dst][:, orb_mask_src] = block

            num_atoms += data.ptr[graph_idx + 1] - data.ptr[graph_idx]

        return diagonal_matrix, non_diagonal_matrix
