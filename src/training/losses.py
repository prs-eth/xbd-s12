"""Loss factory."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_factory(cfg: dict) -> nn.Module:
    """
    Factory function to create loss functions based on configuration.

    Note: Only CE loss kept, all others were just used during experimentation.

    Args:
        cfg: Configuration dictionary with keys:
            - name: Name of the loss function ('combo', 'ce', 'dice', 'focal')
            - loss_weights: (for 'combo') Dictionary mapping loss names to weights, e.g., {'ce': 0.5, 'dice': 1.0, 'focal': 0.3}
            - ignore_index: Index to ignore in labels
            ...
    """

    if "name" not in cfg:
        raise ValueError("Loss configuration must include 'name' key.")

    name = cfg["name"].lower()
    cfg = {k: v for k, v in cfg.items() if k != "name"}
    if name == "ce":
        print(f"Using Cross-Entropy Loss with config: {cfg}")
        return CELoss(**cfg)
    else:
        raise ValueError(f"Loss '{name}' not recognized. Available losses: 'ce', 'dice', 'focal', 'combo', 'changemamba'")


class CELoss(nn.Module):
    """Cross-Entropy Loss for semantic segmentation with ignore index support."""

    def __init__(self, ignore_index: int = 99, reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the Cross-Entropy loss.

        Args:
            preds: (B, C, H, W) logits for multiclass or (B, 1, H, W)/(B,H,W) logits for binary
            labels: (B, H, W) class indices for multiclass or (B, H, W) binary labels
        """
        if len(preds.shape) == 3:
            preds = preds.unsqueeze(1)  # (B, 1, H, W)

        # Binary case: preds is (B, 1, H, W)
        if preds.shape[1] == 1:
            preds = preds.squeeze(1)  # (B, H, W)
            mask = labels != self.ignore_index

            if mask.sum() == 0:
                return preds.sum() * 0.0  # Maintains gradient flow

            preds_masked = preds[mask]
            labels_masked = labels[mask].float()

            loss = F.binary_cross_entropy_with_logits(preds_masked, labels_masked, reduction="none")

            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            elif self.reduction == "none":
                # Reshape back to original spatial dimensions
                result = torch.zeros_like(preds)
                result[mask] = loss
                return result
            else:
                raise ValueError(f"Invalid reduction mode: {self.reduction}")

        # Multiclass case: preds is (B, C, H, W)
        else:
            loss = F.cross_entropy(preds, labels, ignore_index=self.ignore_index, reduction=self.reduction)
            return loss
