import statistics

import numpy as np
import torch

from src.training.utils import apply_buffer_around_buildings


class xBDS12Metrics:
    """
    Metrics class for xBD-style building damage assessment with separate localization and damage classification.
    Computes F1-scores for localization and damage classes, ignoring no-data pixels (value 99).
    """

    def __init__(
        self,
        num_dmg_classes: int = 2,
        dmg_classes_names: list | None = ["intact", "damaged"],
        ignore_index: int = 99,
        localization_only: bool = False,
        extra_buffer_for_evaluation: int | list[str] = None,
    ):
        """
        Initialize the metrics tracker.

        Args:
            num_dmg_classes (int): Number of damage classes (default: 2 for intact and damaged)
            dmg_classes_names (list, optional): Names of damage classes for reporting. If None, defaults to ["intact", "damaged"] or generic names.
            ignore_index (int): Label value to ignore in metrics (default: 99 for no-data)
            localization_only (bool): If True, only compute localization metrics
            extra_buffer_for_evaluation (int | list[str], optional): Additional buffer(s) to apply around buildings. Metrics with extra buffer(s)
                will have suffix "_buf{buffer}" in their names.
        """
        self.localization_only = localization_only
        self.ignore_index = ignore_index
        if extra_buffer_for_evaluation is None:
            self.extra_buffers = []
        else:
            self.extra_buffers = [extra_buffer_for_evaluation] if isinstance(extra_buffer_for_evaluation, int) else extra_buffer_for_evaluation

        if not localization_only:
            self.num_dmg_classes = num_dmg_classes
            self.dmg_classes_names = dmg_classes_names
        self.reset()

    def reset(self):
        """Reset all accumulated metrics."""
        # Localization metrics (building vs background)
        self.loc_tp = 0.0
        self.loc_fp = 0.0
        self.loc_fn = 0.0

        for buffer in self.extra_buffers:
            setattr(self, f"loc_tp_buf{buffer}", 0.0)
            setattr(self, f"loc_fp_buf{buffer}", 0.0)
            setattr(self, f"loc_fn_buf{buffer}", 0.0)

        if not self.localization_only:
            # Damage classification metrics (per class)
            self.dmg_tp = [0.0] * self.num_dmg_classes
            self.dmg_fp = [0.0] * self.num_dmg_classes
            self.dmg_fn = [0.0] * self.num_dmg_classes

    def update(self, labels: torch.Tensor, preds_loc: torch.Tensor, preds_dmg: torch.Tensor = None):
        """
        Update metrics with new predictions and labels.

        Args:
            labels (torch.Tensor): Ground truth labels [B, H, W] (eg 0=background, 1=intact, 2=damaged, 99=no_data)
            preds_loc (torch.Tensor): Localization predictions [B, H, W] (boolean or 0/1)
            preds_dmg (torch.Tensor): Damage predictions [B, H, W] (1=intact, 2=damaged, etc.)
        """

        if not self.localization_only:
            assert preds_dmg is not None

        # Create masks
        if self.ignore_index is not None:
            no_data_mask = labels == self.ignore_index
            valid_mask = ~no_data_mask
        else:
            valid_mask = torch.ones_like(labels, dtype=torch.bool)

        # Building mask (any non-zero, non-99 label is a building)
        buildings_gt = ((labels != 0) & valid_mask).long()

        # Convert predictions to same format
        preds_loc = preds_loc.long()

        # Only consider valid pixels
        valid_preds_loc = preds_loc[valid_mask]
        valid_buildings_gt = buildings_gt[valid_mask]

        # Update localization metrics
        self.loc_tp += torch.sum((valid_preds_loc == 1) & (valid_buildings_gt == 1)).cpu().item()
        self.loc_fp += torch.sum((valid_preds_loc == 1) & (valid_buildings_gt == 0)).cpu().item()
        self.loc_fn += torch.sum((valid_preds_loc == 0) & (valid_buildings_gt == 1)).cpu().item()

        # Update localization metrics with extra buffers
        for buffer in self.extra_buffers:
            # Apply buffer around buildings in ground truth
            buffered_labels = apply_buffer_around_buildings(labels, buffer=buffer, nodata_value=self.ignore_index)
            if self.ignore_index is not None:
                buffered_no_data_mask = buffered_labels == self.ignore_index
                buffered_valid_mask = ~buffered_no_data_mask
            else:
                buffered_valid_mask = torch.ones_like(buffered_labels, dtype=torch.bool)

            buffered_buildings_gt = ((buffered_labels != 0) & buffered_valid_mask).long()
            buffered_valid_preds_loc = preds_loc[buffered_valid_mask]
            buffered_valid_buildings_gt = buffered_buildings_gt[buffered_valid_mask]

            # Update metrics for this buffer
            tp_buf = torch.sum((buffered_valid_preds_loc == 1) & (buffered_valid_buildings_gt == 1)).cpu().item()
            fp_buf = torch.sum((buffered_valid_preds_loc == 1) & (buffered_valid_buildings_gt == 0)).cpu().item()
            fn_buf = torch.sum((buffered_valid_preds_loc == 0) & (buffered_valid_buildings_gt == 1)).cpu().item()

            setattr(self, f"loc_tp_buf{buffer}", getattr(self, f"loc_tp_buf{buffer}") + tp_buf)
            setattr(self, f"loc_fp_buf{buffer}", getattr(self, f"loc_fp_buf{buffer}") + fp_buf)
            setattr(self, f"loc_fn_buf{buffer}", getattr(self, f"loc_fn_buf{buffer}") + fn_buf)

        # Update damage classification metrics (only consider if there are any building pixels)
        if not self.localization_only and torch.sum(buildings_gt) > 0:  # Only proceed if there are building pixels

            labels_flat = labels.view(-1)
            preds_dmg_flat = preds_dmg.view(-1)
            buildings_gt_flat = buildings_gt.view(-1).bool()
            valid_labels_dmg = labels_flat[buildings_gt_flat]
            valid_preds_dmg = preds_dmg_flat[buildings_gt_flat]

            # Update damage classification metrics for each class
            for class_idx in range(1, self.num_dmg_classes + 1):  # Classes 1, 2, ...
                gt_class = valid_labels_dmg == class_idx
                pred_class = valid_preds_dmg == class_idx

                self.dmg_tp[class_idx - 1] += torch.sum(pred_class & gt_class).cpu().item()
                self.dmg_fp[class_idx - 1] += torch.sum(pred_class & ~gt_class).cpu().item()
                self.dmg_fn[class_idx - 1] += torch.sum(~pred_class & gt_class).cpu().item()

    def _compute_f1_score(self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> tuple:
        """
        Compute F1, precision, and recall from TP, FP, FN.

        Returns:
            tuple: (f1_score, precision, recall)
        """
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1, precision, recall

    def _compute_iou(self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> float:
        """
        Compute Intersection over Union (IoU) from TP, FP, FN.

        Returns:
            float: IoU score
        """
        iou = tp / (tp + fp + fn + 1e-8)
        return iou

    def compute(self) -> dict:
        """
        Compute all metrics.

        Returns:
            Dict containing all computed metrics
        """

        # Localization metrics
        f1_loc, prec_loc, rec_loc = self._compute_f1_score(self.loc_tp, self.loc_fp, self.loc_fn)
        iou_loc = self._compute_iou(self.loc_tp, self.loc_fp, self.loc_fn)
        d_metrics = {"F1_loc": f1_loc, "precision_loc": prec_loc, "recall_loc": rec_loc, "IoU_loc": iou_loc}

        # Localization with extra buffers
        for buffer in self.extra_buffers:
            tp_buf = getattr(self, f"loc_tp_buf{buffer}")
            fp_buf = getattr(self, f"loc_fp_buf{buffer}")
            fn_buf = getattr(self, f"loc_fn_buf{buffer}")
            f1_loc_buf, prec_loc_buf, rec_loc_buf = self._compute_f1_score(tp_buf, fp_buf, fn_buf)
            iou_loc_buf = self._compute_iou(tp_buf, fp_buf, fn_buf)
            d_metrics.update(
                {
                    f"F1_loc_buf{buffer}": f1_loc_buf,
                    f"precision_loc_buf{buffer}": prec_loc_buf,
                    f"recall_loc_buf{buffer}": rec_loc_buf,
                    f"IoU_loc_buf{buffer}": iou_loc_buf,
                }
            )

        if self.localization_only:
            return d_metrics

        # Damage F1 scores per class
        f1_dmg_classes = []
        prec_dmg_classes = []
        rec_dmg_classes = []
        iou_dmg_classes = []

        for i in range(self.num_dmg_classes):
            f1, prec, rec = self._compute_f1_score(self.dmg_tp[i], self.dmg_fp[i], self.dmg_fn[i])
            iou = self._compute_iou(self.dmg_tp[i], self.dmg_fp[i], self.dmg_fn[i])
            f1_dmg_classes.append(f1)
            prec_dmg_classes.append(prec)
            rec_dmg_classes.append(rec)
            iou_dmg_classes.append(iou)

        # F1_dmg: Harmonic mean of all damage classes

        f1_dmg = statistics.harmonic_mean(f1_dmg_classes)
        f1_final = 0.3 * f1_loc + 0.7 * f1_dmg

        # mIoU for damage classes
        mIoU = np.mean(iou_dmg_classes)

        # Update metrics dictionary
        d_metrics.update(
            {
                "F1_final": f1_final,
                "F1_dmg": f1_dmg,
                "mIoU_dmg": mIoU,
            }
        )

        # Add F1_final for extra buffers
        for buffer in self.extra_buffers:
            f1_loc_buf = d_metrics[f"F1_loc_buf{buffer}"]
            f1_final_buf = 0.3 * f1_loc_buf + 0.7 * f1_dmg
            d_metrics[f"F1_final_buf{buffer}"] = f1_final_buf

        # Add per-class damage metrics
        class_names = self.dmg_classes_names if self.dmg_classes_names is not None else [f"dmg_class_{i+1}" for i in range(self.num_dmg_classes)]
        for i, class_name in enumerate(class_names):
            d_metrics[f"F1_{class_name}"] = f1_dmg_classes[i]
            d_metrics[f"precision_{class_name}"] = prec_dmg_classes[i]
            d_metrics[f"recall_{class_name}"] = rec_dmg_classes[i]
            d_metrics[f"IoU_{class_name}"] = iou_dmg_classes[i]

        return d_metrics

    def __str__(self) -> str:
        """String representation of current metrics."""
        metrics = self.compute()
        if self.localization_only:
            return (
                f"xBDS12Metrics (Localization only):\n"
                f"  F1_loc: {metrics['F1_loc']:.4f}\n"
                f"  Precision_loc: {metrics['precision_loc']:.4f}\n"
                f"  Recall_loc: {metrics['recall_loc']:.4f}\n"
                f"  IoU_loc: {metrics['IoU_loc']:.4f}\n"
            )
        else:
            str = (
                f"xBDS12Metrics:\n"
                f"  F1_final: {metrics['F1_final']:.4f}\n"
                f"  F1_loc: {metrics['F1_loc']:.4f}\n"
                f"  F1_dmg: {metrics['F1_dmg']:.4f}\n"
                f"  IoU_loc: {metrics['IoU_loc']:.4f}\n"
                f"  mIoU_dmg: {metrics['mIoU_dmg']:.4f}\n"
            )
            for buffer in self.extra_buffers:
                str += f"  F1_loc_buf{buffer}: {metrics[f'F1_loc_buf{buffer}']:.4f}\n"
                str += f"  F1_final_buf{buffer}: {metrics[f'F1_final_buf{buffer}']:.4f}\n"
            return str
