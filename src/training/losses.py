import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_factory(cfg: dict) -> nn.Module:
    """
    Factory function to create loss functions based on configuration.

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
    elif name == "dice":
        print(f"Using Dice Loss with config: {cfg}")
        return DiceLoss(**cfg)
    elif name == "focal":
        print(f"Using Focal Loss with config: {cfg}")
        return FocalLoss(**cfg)
    elif name == "combo":
        print(f"Using Combo Loss with config: {cfg}")
        if "loss_weights" not in cfg:
            raise ValueError("For 'combo' loss, 'loss_weights' must be specified in the configuration.")
        return ComboLoss(**cfg)
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


class DiceLoss(nn.Module):
    """Dice Loss for semantic segmentation with ignore index support."""

    def __init__(self, ignore_index: int = 99, smooth: float = 1.0, weight: torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.reduction = reduction

        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: (B, C, H, W) logits for multiclass or (B, 1, H, W)/(B, H, W) logits for binary
            labels: (B, H, W) class indices for multiclass or (B, H, W) binary labels
        """
        if len(preds.shape) == 3:
            preds = preds.unsqueeze(1)  # (B, 1, H, W)

        mask = labels != self.ignore_index

        if mask.sum() == 0:
            return preds.sum() * 0.0  # Maintains gradient flow

        # Binary case
        if preds.shape[1] == 1:
            preds = preds.squeeze(1)  # (B, H, W)
            preds_sigmoid = torch.sigmoid(preds[mask])
            labels_masked = labels[mask].float()

            intersection = (preds_sigmoid * labels_masked).sum()
            union = preds_sigmoid.sum() + labels_masked.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            return 1.0 - dice

        # Multiclass case
        else:
            preds_softmax = F.softmax(preds, dim=1)  # (B, C, H, W)
            B, C, H, W = preds.shape

            # Flatten spatial dimensions
            preds_flat = preds_softmax.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
            labels_flat = labels.reshape(-1)  # (B*H*W)
            mask_flat = mask.reshape(-1)  # (B*H*W)

            # Apply mask
            preds_masked = preds_flat[mask_flat]  # (N, C)
            labels_masked = labels_flat[mask_flat]  # (N)

            # One-hot encode labels
            labels_one_hot = F.one_hot(labels_masked, num_classes=C).float()  # (N, C)

            # Compute Dice per class
            intersection = (preds_masked * labels_one_hot).sum(dim=0)  # (C)
            union = preds_masked.sum(dim=0) + labels_one_hot.sum(dim=0)  # (C)

            dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)

            # Apply class weights if provided
            if self.weight is not None:
                if self.reduction == "mean":
                    return 1.0 - (dice_per_class * self.weight).mean()
                elif self.reduction == "sum":
                    return 1.0 - (dice_per_class * self.weight).sum()
                else:  # 'none'
                    return 1.0 - dice_per_class * self.weight
            else:
                if self.reduction == "mean":
                    return 1.0 - dice_per_class.mean()
                elif self.reduction == "sum":
                    return 1.0 - dice_per_class.sum()
                else:  # 'none'
                    return 1.0 - dice_per_class


class FocalLoss(nn.Module):
    """Focal Loss for semantic segmentation with ignore index support."""

    def __init__(self, ignore_index: int = 99, alpha: float | torch.Tensor = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.reduction = reduction

        # Register alpha as buffer if it's a tensor, otherwise store as attribute
        if isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = alpha

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: (B, C, H, W) logits for multiclass or (B, 1, H, W)/(B,H,W) logits for binary
            labels: (B, H, W) class indices for multiclass or (B, H, W) binary labels
        """
        if len(preds.shape) == 3:
            preds = preds.unsqueeze(1)  # (B, 1, H, W)

        mask = labels != self.ignore_index

        if mask.sum() == 0:
            return preds.sum() * 0.0  # Maintains gradient flow

        # Binary case
        if preds.shape[1] == 1:
            preds = preds.squeeze(1)  # (B, H, W)
            preds_masked = preds[mask]
            labels_masked = labels[mask].float()

            # BCE loss
            bce_loss = F.binary_cross_entropy_with_logits(preds_masked, labels_masked, reduction="none")

            # Focal modulation
            probs = torch.sigmoid(preds_masked)
            pt = labels_masked * probs + (1 - labels_masked) * (1 - probs)
            focal_weight = (1 - pt) ** self.gamma

            # Alpha weighting (scalar for binary)
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha[0]  # Use first element for binary
            else:
                alpha_t = self.alpha

            alpha_weight = labels_masked * alpha_t + (1 - labels_masked) * (1 - alpha_t)

            loss = alpha_weight * focal_weight * bce_loss

            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            else:
                # Reshape back to original spatial dimensions
                result = torch.zeros_like(preds)
                result[mask] = loss
                return result

        # Multiclass case
        else:
            B, C, H, W = preds.shape

            # Flatten
            preds_flat = preds.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
            labels_flat = labels.reshape(-1)  # (B*H*W)
            mask_flat = mask.reshape(-1)  # (B*H*W)

            # Apply mask
            preds_masked = preds_flat[mask_flat]  # (N, C)
            labels_masked = labels_flat[mask_flat]  # (N)

            # CE loss
            ce_loss = F.cross_entropy(preds_masked, labels_masked, reduction="none")

            # Focal modulation
            probs = F.softmax(preds_masked, dim=1)
            pt = probs[range(len(labels_masked)), labels_masked]
            focal_weight = (1 - pt) ** self.gamma

            # Alpha weighting
            if isinstance(self.alpha, torch.Tensor):
                # Per-class alpha
                alpha_t = self.alpha[labels_masked]
            else:
                # Scalar alpha
                alpha_t = self.alpha

            loss = alpha_t * focal_weight * ce_loss

            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            else:  # 'none'
                # Reshape back to original spatial dimensions
                result = torch.zeros(B * H * W, device=preds.device)
                result[mask_flat] = loss
                return result.reshape(B, H, W)


class ComboLoss(nn.Module):
    """Combines multiple losses with specified weights."""

    def __init__(
        self,
        loss_weights: dict[str, float],
        ignore_index: int = 99,
        dice_smooth: float = 1.0,
        dice_weight: torch.Tensor = None,
        focal_alpha: float | torch.Tensor = 0.25,
        focal_gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            loss_weights: Dictionary mapping loss names to weights, e.g., {'ce': 0.5, 'dice': 1.0, 'focal': 0.3}
            ignore_index: Index to ignore in labels
            dice_smooth: Smoothing parameter for Dice loss
            dice_weight: Per-class weights for Dice loss
            focal_alpha: Alpha parameter for Focal loss (scalar or per-class tensor)
            focal_gamma: Gamma parameter for Focal loss
            reduction: Reduction method for all losses
        """
        super().__init__()
        self.loss_weights = loss_weights
        self.losses = nn.ModuleDict()

        assert all(k in ["ce", "dice", "focal"] for k in loss_weights.keys()), "Invalid loss name in loss_weights"

        # Initialize requested losses
        if "ce" in loss_weights:
            self.losses["ce"] = CELoss(ignore_index=ignore_index, reduction=reduction)

        if "dice" in loss_weights:
            self.losses["dice"] = DiceLoss(ignore_index=ignore_index, smooth=dice_smooth, weight=dice_weight, reduction=reduction)

        if "focal" in loss_weights:
            self.losses["focal"] = FocalLoss(ignore_index=ignore_index, alpha=focal_alpha, gamma=focal_gamma, reduction=reduction)

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: (B, C, H, W) logits for multiclass or (B, 1, H, W) logits for binary
            labels: (B, H, W) class indices

        Returns:
            Combined weighted loss
        """
        total_loss = 0.0

        for loss_name, weight in self.loss_weights.items():
            if loss_name in self.losses:
                loss_value = self.losses[loss_name](preds, labels)
                total_loss = total_loss + weight * loss_value
            else:
                raise ValueError(f"Loss '{loss_name}' not recognized. Available: 'ce', 'dice', 'focal'")

        return total_loss

    def get_individual_losses(self, preds: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute individual losses without combining them.
        Useful for logging/monitoring.

        Returns:
            Dictionary mapping loss names to their values
        """
        losses = {}

        for loss_name in self.loss_weights.keys():
            if loss_name in self.losses:
                losses[loss_name] = self.losses[loss_name](preds, labels)

        return losses
