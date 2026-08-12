import torch
import torch.nn as nn
import torch.nn.functional as F

# Batch hypergraph over the NON-TEXT modalities only. Text is never a vertex (leak-free): a refined
# gallery embedding must not absorb its own caption. Doc hyperedges are block-diagonal; semantic
# mutual-kNN edges are the only cross-doc wiring and exist at train time only.


def doc_incidence(B, mask, device, present=None):
    """H_doc (|V|, B): hyperedge j connects the non-text vertices of doc j. Block-diagonal.

    present (B, k1) 0/1: a MISSING modality (zero-filled feature) is disconnected from its doc edge
    (weight 0), so it neither feeds nor receives graph messages and never pollutes the fusion. None
    => every vertex connected (complete-modality behaviour, unchanged)."""
    k1 = len(mask)
    H = torch.zeros(B * k1, B, device=device)
    idx = torch.arange(B, device=device)
    for m in range(k1):
        H[idx * k1 + m, idx] = 1.0 if present is None else present[:, m]
    return H


@torch.no_grad()
def mutual_knn_adj(t_frozen, k=4, edge_dropout=0.3, training=True, sim_std=None, stats=None,
                   weighted=False, gen_seed=None):

    B = t_frozen.shape[0]
    k = min(k, max(2, B // 4))                                 # adaptive: selective fraction of the shard
    if B <= k + 1:
        return None
    t = F.normalize(t_frozen.float(), dim=-1)
    sim = t @ t.T
    sim.fill_diagonal_(float('-inf'))
    nn_idx = sim.topk(k, dim=-1).indices
    adj = torch.zeros(B, B, device=t.device)
    adj.scatter_(1, nn_idx, 1.0)
    adj = adj * adj.T                                          # mutual
    if sim_std is not None and sim_std > 0:
        off = sim[sim > -1e30]
        thr = off.mean() + sim_std * off.std()                # adaptive per-batch floor
        adj = adj * (sim >= thr).float()
    if training and edge_dropout > 0:
        if gen_seed is None:
            _r = torch.rand(B, B, device=t.device)
        else:
            # With a GLOBAL graph every rank rebuilds the same adjacency independently, so the
            # dropout draw must match across ranks -- otherwise each rank refines with a different
            # graph and DDP averages gradients of four different functions. A seed shared by all
            # ranks and advanced per step keeps the mask identical without extra communication.
            _g = torch.Generator(device=t.device)
            _g.manual_seed(int(gen_seed))
            _r = torch.rand(B, B, device=t.device, generator=_g)
        keep = (_r > edge_dropout).float()
        keep = torch.minimum(keep, keep.T)                    # keep symmetric
        adj = adj * keep
    if weighted:
        # Edge STRENGTH = caption cosine (w_ij = cos(c_i, c_j)), applied only here, at the end.
        # topk / mutual / threshold / dropout above are SET operations and must run on the binary
        # mask: scattering cosines before `adj * adj.T` would square them (0.9*0.9=0.81) and turn a
        # mutually-negative pair positive. Negatives are clamped away -- a negative incidence weight
        # would subtract messages and can make a degree sum cancel toward zero. sim is symmetric and
        # the mask is symmetric, so the weighted adjacency stays symmetric.
        adj = adj * sim.clamp(min=0.0)
    if stats is not None:
        # Edge COUNT alone does not distinguish a graph wiring genuine topic-mates from one wiring
        # arbitrary pairs -- top-k always fills its quota, so an unstructured batch yields about as
        # many edges as a structured one. The mean cosine of RETAINED edges is the quality signal:
        # if it sits near the batch mean, the semantic edges carry little more relation than chance.
        bin_adj = (adj > 0).float()                           # weights would otherwise skew counts
        n_edge = bin_adj.sum().item() / 2.0                   # symmetric -> undirected count
        deg = bin_adj.sum(dim=1)
        off = sim[sim > -1e30]
        e_sim = sim[adj > 0]
        stats.update(B=B, k=k, edges=n_edge,
                     deg_mean=deg.mean().item(), isolated=int((deg == 0).sum().item()),
                     edge_cos=e_sim.mean().item() if e_sim.numel() else float('nan'),
                     batch_cos=off.mean().item(), batch_cos_std=off.std().item())
    return adj


def semantic_incidence(adj, B, mask, device, present=None):
    """H_sem (|V|, B): semantic edge j connects the non-text vertices of doc j and its mutual
    neighbours. adj: (B, B) from mutual_knn_adj (or None).

    present (B, k1) 0/1: same contract as doc_incidence -- a MISSING modality is disconnected from
    the semantic edges too, so it neither sends nor receives graph messages and stays exactly zero.
    Without it a zero vertex still joined every semantic edge, was filled in by its neighbours, and
    then read as PRESENT downstream (present_from_feats runs on the refined features), so training
    scored an imputed full-arity volume while inference scored a masked lower-arity one.
    present=None restores that earlier imputing behaviour."""
    if adj is None or adj.sum() == 0:
        return None
    k1 = len(mask)
    members = adj + torch.eye(B, device=device)
    H = members.repeat_interleave(k1, dim=0)
    if present is not None:
        # repeat_interleave emits rows doc-major (row j*k1+m belongs to doc j, modality m), which is
        # exactly how a (B, k1) presence tensor flattens -> one reshape aligns them.
        H = H * present.reshape(-1, 1)
    return H


class GatedHGNN(nn.Module):
    """<=2 layers of V->E->V message passing with a zero-init gated residual: at step 0 the model
    is the plain AL-heads baseline; the graph only refines.

        F_E  = GELU( D_E^-1 H^T  F_V W_V )
        F_Vn = GELU( D_V^-1 H    F_E W_E )
        F_V <- F_V + tanh(gate_l) * F_Vn
    """

    def __init__(self, k=512, n_layers=2, gate_init=1.0):
        super().__init__()
        assert n_layers <= 2, 'more layers = over-smoothing'
        self.n_layers = n_layers
        self.W_V = nn.ModuleList([nn.Linear(k, k, bias=False) for _ in range(n_layers)])
        self.W_E = nn.ModuleList([nn.Linear(k, k, bias=False) for _ in range(n_layers)])
        # gate_init=1.0 means tanh(1.0)=0.76: a RANDOMLY INITIALISED HGNN is injected at 76%
        # strength into a strongly pretrained representation from step 0, and over a one-epoch run
        # (~530 steps) the module must first learn not to damage that representation before it can
        # help. Observed gate traces fall (1.0 -> 0.83 -> 0.71), i.e. the optimiser spends its
        # budget switching the module off. A small init makes refinement opt-in: the encoders stay
        # intact and the gate only grows if the graph earns it -- which is also a far more
        # interpretable figure than one that decays.
        # Do NOT set exactly 0: tanh'(0)=1 so the gate itself still learns, but the update to W_V /
        # W_E is scaled by tanh(gate)=0, so the module receives no gradient and never starts. ~0.1
        # keeps it alive at ~10% strength.
        self.gates = nn.Parameter(torch.full((n_layers,), float(gate_init)))
        self.edge_head = nn.Linear(k, k, bias=False)

    @staticmethod
    def _norm(H):
        d = H.sum(dim=0).clamp(min=1.0)                        # edge degree
        dv = H.sum(dim=1).clamp(min=1.0)                       # vertex degree
        return H / d.unsqueeze(0), H / dv.unsqueeze(1)

    def forward(self, z, mask, H_doc, H_sem=None, present=None):
        """z: dict non-text modality -> (B, K) L2-normed (no 'T'). present (B,k1) 0/1 marks which
        modalities a doc actually has; a missing one is masked out of the doc edge (H_doc) and of the
        fusion mean, so it neither passes messages nor dilutes h. present=None -> complete behaviour.
        Returns z_hat (L2-normed non-text mods), h (B,K) L2-normed doc embedding, h_prenorm."""
        order = tuple(mask)
        B = z[order[0]].shape[0]
        k1 = len(order)
        F_V = torch.stack([z[m] for m in order], dim=1).reshape(B * k1, -1)

        H = H_doc if H_sem is None else torch.cat([H_doc, H_sem], dim=1)
        H_e, H_v = self._norm(H)

        for l in range(self.n_layers):
            F_E = F.gelu(H_e.T @ self.W_V[l](F_V))
            _msg = H_v @ self.W_E[l](F_E)
            F_Vn = F.gelu(_msg) if l < self.n_layers - 1 else _msg
            F_V = F_V + torch.tanh(self.gates[l]) * F_Vn

        V = F_V.reshape(B, k1, -1)
        z_hat = {m: F.normalize(V[:, i], dim=-1) for i, m in enumerate(order)}
        if present is None:
            h_prenorm = self.edge_head(V.mean(dim=1))
        else:
            w = present.unsqueeze(-1)                                      # (B, k1, 1)
            h_prenorm = self.edge_head((V * w).sum(1) / w.sum(1).clamp(min=1.0))   # mean over PRESENT only
        h = F.normalize(h_prenorm, dim=-1)
        return z_hat, h, h_prenorm
