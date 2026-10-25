"""Backbone-independent heads. All outputs are in per-task Z-score space."""

import torch
from torch import nn


class IndependentTaskHead(nn.Module):
    def __init__(self, input_dim, n_tasks, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
                for _ in range(n_tasks)
            ]
        )
        self.outputs = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_tasks)])

    def features(self, h):
        return torch.stack([p(h) for p in self.projectors], dim=1)

    def predict(self, tokens):
        return torch.cat([head(tokens[:, j]) for j, head in enumerate(self.outputs)], dim=-1)

    def forward(self, h):
        return self.predict(self.features(h))


class _MLoREBlock(nn.Module):
    """One MLoRE module (CVPR'24, arXiv:2403.17749), adapted from dense-prediction
    convolutions to per-molecule vectors: 1x1/3x3 convolutions become linear layers
    and the spatial-attention router reduces to a linear router on the task vector.
    Faithful to the reference MOEBlock: task-sharing linear generic path with stopped
    input gradients, task-sharing linear low-rank experts with diverse ranks selected
    by a per-task noisy top-k router (pre-softmax masking), a task-specific low-rank
    expert scaled by a learned mix gate, and BN -> GELU -> output projection.
    """

    def __init__(self, input_dim, hidden_dim, n_tasks, rank_list, spe_rank, topk, pre_softmax):
        super().__init__()
        if not 1 <= topk <= len(rank_list):
            raise ValueError("top-k must select between one and all shared experts")
        self.n_tasks = n_tasks
        self.num_experts = len(rank_list)
        self.desert_k = self.num_experts - topk
        self.pre_softmax = pre_softmax
        self.expert_down = nn.ModuleList([nn.Linear(hidden_dim, r) for r in rank_list])
        self.expert_up = nn.ModuleList([nn.Linear(r, hidden_dim) for r in rank_list])
        self.share = nn.Linear(hidden_dim, hidden_dim)
        self.conv1 = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(n_tasks)])
        self.spe_down = nn.ModuleList([nn.Linear(hidden_dim, spe_rank) for _ in range(n_tasks)])
        self.spe_up = nn.ModuleList([nn.Linear(spe_rank, hidden_dim) for _ in range(n_tasks)])
        self.conv3 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_tasks)])
        self.bn_all = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_tasks)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_tasks)])
        self.router = nn.ModuleList(
            [nn.Linear(hidden_dim, 2 * self.num_experts + 1) for _ in range(n_tasks)]
        )
        self.activate = nn.GELU()

    def forward_task(self, x, t):
        out = self.conv1[t](x)
        prob_all = self.router[t](out)
        logits = prob_all[:, : self.num_experts]
        stds = prob_all[:, self.num_experts : 2 * self.num_experts]
        prob_mix = prob_all[:, 2 * self.num_experts :]
        if self.training:
            logits = logits + torch.randn_like(logits) * stds
        if self.pre_softmax:
            desert = torch.topk(logits, self.desert_k, dim=1, largest=False)[1]
            route = torch.softmax(logits.scatter(1, desert, -1e10), dim=1)
        else:
            dense = torch.softmax(logits, dim=1)
            desert = torch.topk(dense, self.desert_k, dim=1, largest=False)[1]
            route = dense.scatter(1, desert, 0.0)
        mixed = None
        for n in range(self.num_experts):
            term = route[:, n : n + 1] * self.expert_up[n](self.expert_down[n](out))
            mixed = term if mixed is None else mixed + term
        fused = (
            self.bn_all[t](mixed)
            + self.spe_up[t](self.spe_down[t](out)) * prob_mix
            + self.share(out.detach())
        )
        fused = self.activate(self.bn[t](fused))
        return self.conv3[t](fused), route


class MLoREHead(nn.Module):
    """MLoRE multi-task head: `num_modules` stacked MLoRE modules per task (module
    i+1 consumes module i's task features), then a per-task linear readout. Route
    distributions are exposed for the coefficient-of-variation load-balancing loss.
    """

    def __init__(
        self,
        input_dim,
        n_tasks=7,
        hidden_dim=128,
        rank_list=None,
        spe_rank=4,
        topk=9,
        pre_softmax=True,
        num_modules=2,
        balance_weight=0.0005,
    ):
        super().__init__()
        rank_list = tuple(rank_list) if rank_list else tuple(range(4, 61, 4))
        if num_modules < 1:
            raise ValueError("At least one MLoRE module is required")
        dims = [input_dim] + [hidden_dim] * (num_modules - 1)
        self.blocks = nn.ModuleList(
            [
                _MLoREBlock(dims[i], hidden_dim, n_tasks, rank_list, spe_rank, topk, pre_softmax)
                for i in range(num_modules)
            ]
        )
        self.n_tasks = n_tasks
        self.balance_weight = balance_weight
        self.outputs = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_tasks)])
        self.last_route_probs = None

    def forward(self, h):
        routes_per_module = []
        feats = h
        for block in self.blocks:
            nxt, routes = [], []
            for t in range(self.n_tasks):
                block_in = feats[:, t] if feats.dim() == 3 else feats
                f, r = block.forward_task(block_in, t)
                nxt.append(f)
                routes.append(r)
            feats = torch.stack(nxt, dim=1)
            routes_per_module.append(routes)
        self.last_route_probs = routes_per_module
        return torch.cat([self.outputs[t](feats[:, t]) for t in range(self.n_tasks)], dim=-1)

    def balance_loss(self):
        """Squared coefficient of variation of mean per-expert route probabilities,
        summed over stacked modules (reference MultiTaskLoss load-balancing term)."""
        total = None
        for routes in self.last_route_probs:
            mean_prob = torch.cat(routes, dim=0).mean(dim=0)
            term = (mean_prob.std() / mean_prob.mean()) ** 2
            total = term if total is None else total + term
        return total
