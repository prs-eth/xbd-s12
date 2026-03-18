"""Precompute stats for normalization. This should not be used as the file stats.json is provided with the dataset."""

import json

import numpy as np
import rasterio
import torch
from tqdm import tqdm

from src.training.dataloaders import get_dataloaders
from src.utils.time import timeit


# ====== Precompute stats (min/max, 1st/99th percentile)
@timeit
def precompute_stats(main_folder, split="train"):
    """Precompute stats (min/max, 1st/99th percentile) across all images in the dataset"""

    d_stats = {}
    for sensor in ["s1", "s2"]:

        fps = list((main_folder / split / sensor).glob("*.tif"))
        n_bands = read_tif(fps[0]).shape[0]
        bins = get_bins(sensor)

        print(f"Precomputing stats for {sensor} ({n_bands} bands, {len(fps)} images)")
        mins = np.zeros([n_bands, len(fps)])
        maxs = np.zeros([n_bands, len(fps)])
        hist = np.zeros([n_bands, len(bins) - 1])
        for i, fp in tqdm(enumerate(fps), total=len(fps)):

            img = read_tif(fp)
            mins[:, i] = np.min(img, axis=(1, 2))
            maxs[:, i] = np.max(img, axis=(1, 2))

            for b in range(n_bands):
                hist[b] += np.histogram(img[b].flatten(), bins=bins)[0]

        perc1, perc99 = [], []
        for b in range(n_bands):
            cumsum = np.cumsum(hist[b]) / np.sum(hist[b])
            perc1.append(bins[np.where(cumsum > 0.01)[0][0]])
            perc99.append(bins[np.where(cumsum > 0.99)[0][0]])

        d_stats[sensor] = {
            "min": [float(v) for v in mins.min(axis=1)],
            "max": [float(v) for v in maxs.max(axis=1)],
            "1st": [float(v) for v in perc1],
            "99th": [float(v) for v in perc99],
        }
        print(d_stats[sensor])

    with open(main_folder / "stats.json", "w") as f:
        json.dump(d_stats, f, indent=4)


def read_tif(fp):
    with rasterio.open(fp) as src:
        return src.read().squeeze()


def get_bins(sensor):
    """Reasonable bins for histogram"""
    if sensor == "s2":
        bins = np.array(range(0, 22000))
    elif sensor == "s1":
        bins = np.linspace(-65, 40, 20000)
    else:
        raise ValueError(f"Unknown sensor {sensor}")
    return bins


# ====== Precompute stats (mean/std)
@timeit
def precompute_mean_std(main_folder):
    """Quick and dirty"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rs_s1 = RunningStatsButFast((2,), [0, 2, 3]).to(device)
    rs_s2 = RunningStatsButFast((12,), [0, 2, 3]).to(device)
    dls = get_dataloaders(which_split="hafner", modalities=["s1", "s2"], normalize_data=False)
    dl = dls["train"]
    for batch in tqdm(dl):
        s1 = torch.cat([batch["images"]["s1_pre"], batch["images"]["s1_post"]], dim=0).to(device)
        s2 = torch.cat([batch["images"]["s2_pre"], batch["images"]["s2_post"]], dim=0).to(device)
        rs_s1(s1)
        rs_s2(s2)

    # Store in dict
    d_stats = {}
    d_stats["s1"] = {
        "mean": [float(v) for v in rs_s1.mean.cpu().numpy()],
        "std": [float(v) for v in rs_s1.std.cpu().numpy()],
    }
    d_stats["s2"] = {
        "mean": [float(v) for v in rs_s2.mean.cpu().numpy()],
        "std": [float(v) for v in rs_s2.std.cpu().numpy()],
    }

    fp_stats = main_folder / "stats.json"
    if not fp_stats.exists():
        with open(fp_stats, "w") as f:
            json.dump(d_stats, f, indent=4)
    else:
        with open(fp_stats, "r") as f:
            existing_stats = json.load(f)
        for sensor in d_stats.keys():
            existing_stats[sensor].update(d_stats[sensor])
        with open(fp_stats, "w") as f:
            json.dump(existing_stats, f, indent=4)
    print("Mean and std stats saved.")


class RunningStatsButFast(torch.nn.Module):
    def __init__(self, shape, dims):
        """
        From https://gist.github.com/calebrob6/1ef1e64bd62b1274adf2c6f91e20d215

        Initializes the RunningStatsButFast method.

        A PyTorch module that can be put on the GPU and calculate the multidimensional
        mean and variance of inputs online in a numerically stable way. This is useful
        for calculating the channel-wise mean and variance of a big dataset because you
        don't have to load the entire dataset into memory.

        Uses the "Parallel algorithm" from: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        Similar implementation here: https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py#L5

        Access the mean, variance, and standard deviation of the inputs with the
        `mean`, `var`, and `std` attributes.

        Example:
        ```
        rs = RunningStatsButFast((12,), [0, 2, 3])
        for inputs, _ in dataloader:
            rs(inputs)
        print(rs.mean)
        print(rs.var)
        print(rs.std)
        ```

        Args:
            shape: The shape of resulting mean and variance. For example, if you
                are calculating the mean and variance over the 0th, 2nd, and 3rd
                dimensions of inputs of size (64, 12, 256, 256), this should be 12.
            dims: The dimensions of your input to calculate the mean and variance
                over. In the above example, this should be [0, 2, 3].
        """
        super(RunningStatsButFast, self).__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("std", torch.ones(shape))
        self.register_buffer("count", torch.zeros(1))
        self.dims = dims

    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, dim=self.dims)
            batch_var = torch.var(x, dim=self.dims)
            batch_count = torch.tensor(x.shape[self.dims[0]], dtype=torch.float)

            n_ab = self.count + batch_count
            m_a = self.mean * self.count
            m_b = batch_mean * batch_count
            M2_a = self.var * self.count
            M2_b = batch_var * batch_count

            delta = batch_mean - self.mean

            self.mean = (m_a + m_b) / (n_ab)
            # we don't subtract -1 from the denominator to match the standard Numpy/PyTorch variances
            self.var = (M2_a + M2_b + delta**2 * self.count * batch_count / (n_ab)) / (n_ab)
            self.count += batch_count
            self.std = torch.sqrt(self.var + 1e-8)

    def forward(self, x):
        self.update(x)
        return x


if __name__ == "__main__":
    from src.constants import READY_PATH

    main_folder = READY_PATH / "xbd_sentinel_v2"
    precompute_stats(main_folder, split="train")
    precompute_mean_std(main_folder)
