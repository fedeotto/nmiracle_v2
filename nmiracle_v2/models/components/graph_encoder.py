"""
Wrapper around GraphMVP's GNN that:
  1. Accepts a PyG Batch and returns padded per-node token tensors [B, max_nodes, d_model]
  2. Optionally loads pretrained GraphMVP weights
"""
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from nmiracle_v2.models.components.molecule_gnn_model import GNNComplete  # noqa: E402  (GraphMVP)
GRAPHMVP_EMB_DIM = 300  # default GraphMVP embedding dimension

class GraphEncoder(nn.Module):
    """
    Encodes molecular graphs as sequences of node embeddings.

    Returns:
        node_tokens : [B, max_nodes, d_model]  (padded)
        node_mask   : [B, max_nodes]  bool, True = padding position
    """

    def __init__(
        self,
        d_model: int = 128,
        gnn_num_layer: int = 5,
        gnn_emb_dim: int = GRAPHMVP_EMB_DIM,
        gnn_JK: str = "last",
        gnn_drop_ratio: float = 0.0,
        gnn_type: str = "gin",
        pretrained_path: str | None = None,
        freeze: bool = True,
    ):
        super().__init__()

        self.gnn = GNNComplete( #note we need ogb version 1.3.4 to load the complete checkpoint
            num_layer=gnn_num_layer,
            emb_dim=gnn_emb_dim,
            JK=gnn_JK,
            drop_ratio=gnn_drop_ratio,
            gnn_type=gnn_type,
        )

        if pretrained_path is not None:
            self._load_gnn_weights(pretrained_path)

        if freeze:
            for p in self.gnn.parameters():
                p.requires_grad = False

        # Project GNN node dim → transformer d_model
        self.proj = nn.Sequential(
            nn.Linear(gnn_emb_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def _load_gnn_weights(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        # GraphMVP checkpoints may store the GNN under 'molecule_model' or at top level
        state = ckpt.get("model", ckpt)
        # Strip 'molecule_model.' prefix if present
        gnn_state = {
            k.replace("molecule_model.", ""): v
            for k, v in state.items()
            if k.startswith("molecule_model.") or not any(
                k.startswith(p) for p in ["proj", "pool", "graph_pred"]
            )
        }
        missing, unexpected = self.gnn.load_state_dict(gnn_state, strict=False)
        loaded = len(self.gnn.state_dict()) - len(missing)
        print(f"GraphEncoder: loaded {loaded}/{len(self.gnn.state_dict())} GNN params from {path}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected[:5]}")

    def forward(self, graph_batch):
        """
        Args:
            graph_batch: torch_geometric.data.Batch with fields x, edge_index, edge_attr, batch

        Returns:
            node_tokens : [B, max_nodes, d_model]
            node_mask   : [B, max_nodes]  True = padding
        """
        # GNN produces flat node embeddings [total_nodes, gnn_emb_dim]
        node_emb = self.gnn(
            graph_batch.x,
            graph_batch.edge_index,
            graph_batch.edge_attr,
        )

        # Project to d_model
        node_emb = self.proj(node_emb)  # [total_nodes, d_model]

        # Scatter into dense [B, max_nodes, d_model], mask = True where padded
        node_tokens, node_mask = to_dense_batch(node_emb, graph_batch.batch)
        node_mask = ~node_mask  # to_dense_batch returns True=valid; invert for key_padding_mask convention

        return node_tokens, node_mask
