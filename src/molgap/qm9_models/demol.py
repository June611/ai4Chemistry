"""Paper-based DeMol reimplementation, equations 5--10 and Appendix B.

No author implementation was linked in the inspected paper/proceedings. Choices
not specified there are listed in docs/features/qm9-demol-transformer-m/README.md.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class GaussianBias(nn.Module):
    def __init__(self, kernels, heads, width, types=100):
        super().__init__()
        self.register_buffer("means", torch.arange(kernels) * width / kernels)
        self.sigma = width / kernels
        self.alpha = nn.Embedding(types, 1)
        self.beta = nn.Embedding(types, 1)
        nn.init.ones_(self.alpha.weight)
        nn.init.zeros_(self.beta.weight)
        self.project = nn.Sequential(
            nn.Linear(kernels, kernels), nn.GELU(), nn.Linear(kernels, heads)
        )

    def forward(self, distance, pair_type):
        x = self.alpha(pair_type) * distance[..., None] + self.beta(pair_type)
        phi = torch.exp(-0.5 * ((x - self.means) / self.sigma).square())
        phi = phi / (math.sqrt(2 * math.pi) * self.sigma)
        return self.project(phi).permute(0, 3, 1, 2)


class AttentionUpdate(nn.Module):
    """Post-norm residual attention + FFN, with explicit safe padding rows."""

    def __init__(self, dim, ffn, heads, attention_dropout, activation_dropout):
        super().__init__()
        self.heads, self.width = heads, dim // heads
        if dim % heads:
            raise ValueError("DeMol dimension must be divisible by heads")
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn), nn.GELU(), nn.Dropout(activation_dropout), nn.Linear(ffn, dim)
        )
        self.attention_dropout = attention_dropout

    def forward(self, query, context, bias, allowed, query_mask):
        b, qn, dim = query.shape
        kn = context.shape[1]

        def split(x):
            return x.reshape(b, -1, self.heads, self.width).transpose(1, 2)

        # Padded/isolated query rows get one dummy key; zeroed after the update.
        dummy = torch.arange(kn, device=query.device) == 0
        safe = allowed | (~allowed.any(-1, keepdim=True) & dummy)
        additive = bias.masked_fill(~safe[:, None], float("-inf"))
        value = (
            F.scaled_dot_product_attention(
                split(self.q(query)),
                split(self.k(context)),
                split(self.v(context)),
                attn_mask=additive,
                dropout_p=self.attention_dropout if self.training else 0.0,
            )
            .transpose(1, 2)
            .reshape(b, qn, dim)
        )
        x = self.norm1(query + self.out(value))
        x = self.norm2(x + self.ffn(x))
        return x * query_mask[..., None]


class DoubleHelix(nn.Module):
    def __init__(self, dim, ffn, heads, attention_dropout, activation_dropout):
        super().__init__()
        args = dim, ffn, heads, attention_dropout, activation_dropout
        self.atom = AttentionUpdate(*args)
        self.bond = AttentionUpdate(*args)
        self.atom_from_bond = AttentionUpdate(*args)
        self.bond_from_atom = AttentionUpdate(*args)

    def forward(
        self,
        atom,
        bond,
        abias,
        bbias,
        across,
        bcross,
        amask,
        bmask,
        incidence,
        atom_mask,
        bond_mask,
    ):
        atom = self.atom(atom, atom, abias, amask, atom_mask)
        bond = self.bond(bond, bond, bbias, bmask, bond_mask)
        # Bidirectional cross updates use the same pre-cross states.
        a = self.atom_from_bond(atom, bond, across, incidence, atom_mask)
        b = self.bond_from_atom(bond, atom, bcross, incidence.transpose(1, 2), bond_mask)
        # An atom without chemical bonds retains its self-channel state.
        a = torch.where(incidence.any(-1)[..., None], a, atom)
        return a, b


class DeMolAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        p = dict(config)
        dim, ffn = p.pop("embed_dim", 768), p.pop("ffn_dim", 768)
        heads, kernels = p.pop("num_heads", 32), p.pop("num_kernel", 128)
        layers = p.pop("num_layers", 12)
        self.cutoff = p.pop("cutoff", 5.0)
        self.radius_factor = p.pop("covalent_radius_factor", 1.15)
        self.use_checkpoint = p.pop("activation_checkpointing", True)
        self.bond_regularization = p.pop("bond_regularization", False)
        dropout = p.pop("attention_dropout", 0.1)
        activation_dropout = p.pop("activation_dropout", 0.0)
        self.atom_embedding = nn.Embedding(512 * 9 + 1, dim, padding_idx=0)
        self.bond_embedding = nn.Embedding(512 * 3 + 1, dim, padding_idx=0)
        self.atom_spd = nn.Embedding(512, heads)
        self.bond_spd = nn.Embedding(512, heads)
        self.atom_geometry = GaussianBias(kernels, heads, p.pop("distance_width", 8.0))
        self.bond_geometry = GaussianBias(kernels, heads, 8.0)
        self.torsion_geometry = GaussianBias(kernels, heads, 2.0)
        self.layers = nn.ModuleList(
            [DoubleHelix(dim, ffn, heads, dropout, activation_dropout) for _ in range(layers)]
        )
        self.readout = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1))
        self.bond_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        # RDKit periodic-table covalent radii, Angstrom, fixed input constants.
        from rdkit import Chem

        radii = [Chem.GetPeriodicTable().GetRcovalent(z) if z else 0.0 for z in range(10)]
        self.register_buffer("radii", torch.tensor(radii))
        self._bond_loss = None
        if p:
            raise ValueError(f"Unknown DeMol parameters: {sorted(p)}")

    def forward(self, batch):
        z, pos, valid = batch["charges"], batch["positions"], batch["atom_mask"]
        bs, n = z.shape
        feat = batch["node_features"] + torch.arange(9, device=z.device) * 512 + 1
        atom = self.atom_embedding(feat).sum(-2) * valid[..., None]
        # Directed bond nodes make the paper's signed bond-vector cosine encoding
        # independent of arbitrary undirected edge endpoint/atom ordering.
        edges = torch.cat((batch["bonds"], batch["bonds"].flip(-1)), 1)
        bf = batch["bond_features"].repeat(1, 2, 1)
        bm = batch["bond_mask"].repeat(1, 2)
        left, right = edges[..., 0], edges[..., 1]
        bid = torch.arange(bs, device=z.device)[:, None]
        bond = atom[bid, left] + atom[bid, right]
        bf = bf + torch.arange(3, device=z.device) * 512 + 1
        bond = (bond + self.bond_embedding(bf).sum(-2)) * bm[..., None]
        atom_pair = valid[:, :, None] & valid[:, None, :]
        dist = torch.cdist(pos, pos)
        types = z[:, :, None] * 10 + z[:, None, :]
        abias = self.atom_geometry(dist, types)
        abias = abias + self.atom_spd(batch["spatial"].clamp_max(511)).permute(0, 3, 1, 2)
        amask = atom_pair & (dist < self.cutoff)
        midpoint = (pos[bid, left] + pos[bid, right]) / 2
        vector = F.normalize(pos[bid, right] - pos[bid, left], dim=-1)
        bdist = torch.cdist(midpoint, midpoint)
        cosine = (vector @ vector.transpose(1, 2)).clamp(-1, 1)
        btypes = (batch["bond_features"][..., 0] + 1).repeat(1, 2)
        bt = btypes[:, :, None] * 10 + btypes[:, None, :]
        bbias = self.bond_geometry(bdist, bt) + self.torsion_geometry(cosine + 1, bt)
        sp = batch["spatial"]
        bsp = (
            torch.minimum(
                torch.minimum(
                    sp[bid[..., None], left[:, :, None], left[:, None, :]],
                    sp[bid[..., None], left[:, :, None], right[:, None, :]],
                ),
                torch.minimum(
                    sp[bid[..., None], right[:, :, None], left[:, None, :]],
                    sp[bid[..., None], right[:, :, None], right[:, None, :]],
                ),
            )
            + 1
        )
        same = (left[:, :, None] == left[:, None, :]) & (right[:, :, None] == right[:, None, :])
        bsp = bsp.masked_fill(same, 0)
        bbias = bbias + self.bond_spd(bsp.clamp_max(511)).permute(0, 3, 1, 2)
        incidence = (torch.arange(n, device=z.device)[None, :, None] == left[:, None]) | (
            torch.arange(n, device=z.device)[None, :, None] == right[:, None]
        )
        incidence = incidence & valid[:, :, None] & bm[:, None]
        adjacent = (incidence.float().transpose(1, 2) @ incidence.float()) > 0
        # Conjugacy relation is transitive connectivity within conjugated bonds.
        conjugated = (batch["bond_features"][..., 2] != 0).repeat(1, 2) & bm
        conjugacy = adjacent & conjugated[:, :, None] & conjugated[:, None, :]
        for _ in range(math.ceil(math.log2(max(1, edges.shape[1])))):
            conjugacy = conjugacy | ((conjugacy.float() @ conjugacy.float()) > 0)
        bmask = (adjacent | conjugacy) & bm[:, :, None] & bm[:, None, :]
        # Cross positional bias: atom-to-bond midpoint distance and its transpose.
        cross_dist = torch.cdist(pos, midpoint)
        cross_type = z[:, :, None] * 10 + btypes[:, None, :]
        across = self.atom_geometry(cross_dist, cross_type)
        bcross = across.transpose(-1, -2)
        args = abias, bbias, across, bcross, amask, bmask, incidence, valid, bm
        for layer in self.layers:
            if self.training and self.use_checkpoint:
                atom, bond = checkpoint(layer, atom, bond, *args, use_reentrant=False)
            else:
                atom, bond = layer(atom, bond, *args)
        if self.training and self.bond_regularization:
            logits = self.bond_head(atom[:, :, None] + atom[:, None, :]).squeeze(-1)
            threshold = self.radius_factor * (self.radii[z][:, :, None] + self.radii[z][:, None, :])
            pairs = atom_pair & torch.triu(
                torch.ones(n, n, device=z.device, dtype=torch.bool), diagonal=1
            )
            target = (dist <= threshold).to(logits.dtype)
            self._bond_loss = (
                F.binary_cross_entropy_with_logits(logits[pairs], target[pairs])
                if pairs.any()
                else atom.sum() * 0
            )
        else:
            self._bond_loss = None
        apool = atom.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        bpool = bond.sum(1) / bm.sum(1, keepdim=True).clamp_min(1)
        return self.readout(torch.cat((apool, bpool), -1)).squeeze(-1)

    def auxiliary_loss(self):
        if self._bond_loss is None:
            raise RuntimeError("Bond regularization is only available after a training forward")
        return self._bond_loss
