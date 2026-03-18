import torch.utils.data as tdata


def get_weighted_sampler(ds: tdata.Dataset, use_simplified_classes: bool = True) -> tdata.WeightedRandomSampler:
    """
    Returns a WeightedRandomSampler for the given dataset based on precomputed number of pixels per class.

    Each class is assigned a weight w_i inversely proportional to its frequency in the training set. Then each patch
    is assigned a weight based on the classes it contains.

    Args:
        ds (tdata.Dataset): The xBD12Dataset for which to compute the weights.
        use_simplified_classes (bool): Whether to use simplified damage classes (Background, Intact, Damaged) or original ones. Defaults to True.

    """

    # Get valid columns (so we exclude unclassified and no_data pixels from the weighting)
    if use_simplified_classes:
        # Merge original damage classes into a single "damaged" class
        ds.meta["N_px_damaged"] = ds.meta[["N_px_minor", "N_px_major", "N_px_destroyed"]].sum(axis=1)
        valid_class_cols = ["N_px_background", "N_px_intact", "N_px_damaged"]
    else:
        valid_class_cols = ["N_px_background", "N_px_intact", "N_px_minor", "N_px_major", "N_px_destroyed"]

    # Compute number of pixels per class for each sample
    total_px_per_class = ds.meta[valid_class_cols].sum()
    total_px = total_px_per_class.sum()  # total number of pixels per class across all samples

    # Inverse frequency weighting: we give a higher weight to classes that are less frequent in the dataset.
    weight_per_class = total_px / total_px_per_class

    # Multiply the pixel counts by the class weights to get a raw weight for each sample
    raw_row_weights = (ds.meta[valid_class_cols] * weight_per_class).sum(axis=1)

    # Number of valid pixels per sample
    valid_px_per_row = ds.meta[valid_class_cols].sum(axis=1)

    # Average weight per valid pixel for each sample
    average_weight_per_row = raw_row_weights / valid_px_per_row

    # Map weights to dataset samples
    ds.meta["uid_weights"] = average_weight_per_row  # add weights to dataset meta
    sample_weights = ds.meta["uid_weights"].to_list()

    return tdata.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
