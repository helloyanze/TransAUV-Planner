from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:
    class OceanPathTransformer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.grid_size = int(config.get("grid_size", 20))
            self.d_model = int(config.get("d_model", 128))
            self.nhead = int(config.get("nhead", 8))
            self.num_layers = int(config.get("num_layers", 4))
            self.predict_steps = int(config.get("predict_steps", 5))
            self.input_dim = int(config.get("input_dim", 6))

            self.spatial_embed = nn.Linear(3, self.d_model // 2)
            self.feature_embed = nn.Linear(self.input_dim, self.d_model // 2)
            self.time_embed = nn.Embedding(int(config.get("time_vocab_size", 32)), self.d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.d_model * 4,
                dropout=0.1,
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

            self.h_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Linear(self.d_model // 2, 1),
                nn.Softplus(),
            )
            self.weight_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Linear(self.d_model // 2, 6),
                nn.Softmax(dim=-1),
            )
            self.current_pred_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Linear(self.d_model // 2, 3),
            )

        def encode_node(self, coords, features, time_step):
            spatial = self.spatial_embed(coords.float())
            feature = self.feature_embed(features.float())
            time = self.time_embed(time_step)
            return torch.cat([spatial, feature], dim=-1) + time

        def forward(self, coords, features, time_steps, goal_coords):
            tokens = self.encode_node(coords, features, time_steps)
            goal_embed = self.spatial_embed(goal_coords.float()).expand(-1, tokens.size(1), -1)
            tokens[:, :, : self.d_model // 2] += goal_embed

            encoded = self.transformer_encoder(tokens)
            h_pred = self.h_head(encoded).squeeze(-1)
            weights_pred = self.weight_head(encoded)
            current_pred = self.current_pred_head(encoded)
            return h_pred, weights_pred, current_pred


    class PathPlanningLoss(nn.Module):
        def __init__(self, admissibility_weight=2.0):
            super().__init__()
            self.admissibility_weight = admissibility_weight

        def forward(self, h_pred, h_true, w_pred, w_true, current_pred, current_true):
            loss_h = nn.MSELoss()(h_pred, h_true)
            overestimate = torch.clamp(h_pred - h_true, min=0)
            loss_admissible = (overestimate**2).mean()
            loss_w = nn.KLDivLoss(reduction="batchmean")(torch.log(w_pred + 1e-8), w_true)
            loss_current = nn.MSELoss()(current_pred, current_true)
            total_loss = (
                loss_h
                + self.admissibility_weight * loss_admissible
                + 0.5 * loss_w
                + 0.3 * loss_current
            )
            return total_loss, {
                "loss_h": float(loss_h.detach().item()),
                "loss_admissible": float(loss_admissible.detach().item()),
                "loss_w": float(loss_w.detach().item()),
                "loss_current": float(loss_current.detach().item()),
            }
else:
    class OceanPathTransformer:  # pragma: no cover
        def __init__(self, *_args, **_kwargs):
            raise ImportError("PyTorch is required to use OceanPathTransformer.")


    class PathPlanningLoss:  # pragma: no cover
        def __init__(self, *_args, **_kwargs):
            raise ImportError("PyTorch is required to use PathPlanningLoss.")
