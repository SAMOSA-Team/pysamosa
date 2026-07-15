"""JIT-accelerated bootstrap sampling utilities for PM2.5 data experimentation."""

import numpy as np
import pandas as pd
from numba import jit, njit

np.random.seed(1)


def drop_nan(arr):
    mask = np.empty(arr.shape[0], dtype=np.bool_)
    for i in range(arr.shape[0]):
        mask[i] = not np.isnan(arr[i])
    return arr[mask]


@njit
def apply_mean(arr, axis):
    """JIT-compatible averaging along a given axis.

    Args:
        arr: Array to be averaged.
        axis: Axis to average along.

    Returns:
        Averaged value.
    """
    return arr.sum(axis=axis) / arr.shape[axis]


@njit
def block_agg_np(array, block_size):
    """JIT-compatible block aggregation (mean per block).

    Args:
        array: Array to be aggregated.
        block_size: Number of elements per block.

    Returns:
        Array of per-block means.
    """
    array_chunks = np.array_split(array, len(array) // block_size)
    return np.array([np.mean(chunk) for chunk in array_chunks])


@njit
def block_agg_np2(array, block_size):
    """
    JIT-compatible block aggregation function.
    """
    result = np.empty((array.shape[0], array.shape[1] // block_size), dtype=array.dtype)
    for i in range(array.shape[0]):
        row_chunks = np.array_split(array[i], array.shape[1] // block_size)
        result[i, :] = np.array([np.mean(chunk) for chunk in row_chunks])
    return result


@njit
def shannon_entropy(dist):
    bin_min = int(np.floor(np.nanmin(dist) + 0.1))
    bin_max = int(np.ceil(np.nanmax(dist) + 0.1))
    n_bins = int(np.ceil(bin_max - (bin_min - 1)))

    if n_bins <= 2:
        n_bins = 3

    bins = np.linspace(bin_min, bin_max, n_bins)

    pk, _ = np.histogram(dist, bins=bins)
    pk = np.divide(pk, np.sum(pk))

    filtered_indices = pk != 0
    return -np.nansum(pk[filtered_indices] * np.log(pk[filtered_indices]))


@njit
def relative_entropy(p_dist, q_dist):
    bin_min = int(np.floor(np.nanmin(q_dist) + 0.1))
    bin_max = int(np.ceil(np.nanmax(q_dist) + 0.1))
    n_bins = int(np.ceil(bin_max - (bin_min - 1)))

    if n_bins <= 2:
        n_bins = 3

    bins = np.linspace(bin_min, bin_max, n_bins)

    pk, _ = np.histogram(p_dist, bins=bins)
    pk = np.divide(pk, np.sum(pk))

    qk, _ = np.histogram(q_dist, bins=bins)
    qk = np.divide(qk, np.sum(qk))
    qk = np.add(qk, 0.0000000000001)

    filtered_indices = pk != 0
    return np.nansum(
        pk[filtered_indices] * np.log(pk[filtered_indices] / qk[filtered_indices])
    )


@jit(nopython=True)
def sampler(obs, n, p):
    """Draw 1000 bootstrap samples of size *n* and return their *p*-th percentile.

    Args:
        obs: Observations to sample from.
        n: Sample size.
        p: Percentile to compute for each sample (0–100).

    Returns:
        Array of 1000 percentile values.
    """
    sample = np.empty(1000)
    for i in range(0, 1000):
        dist = np.random.choice(obs, size=n, replace=False)
        sample[i] = np.percentile(dist, p)
    return sample


@jit(nopython=True)
def drift_sampler(obs, n, p, dx):
    """Bootstrap sampler with linearly increasing noise drift.

    Args:
        obs: 1-D observation array to sample from.
        n: Number of elements chosen without replacement per sample.
        p: Percentile to compute for each sample.
        dx: Drift scaling factor; noise standard deviation grows by *dx* per step.

    Returns:
        Array of 1000 percentile values from noisy observations.
    """
    sample = np.zeros((1000,))
    for i in range(0, 1000):
        x = 0
        noisy_obs = np.zeros(
            len(obs),
        )
        for j in range(0, len(obs)):
            x += dx
            noisy_obs[j] = np.random.normal(0, x) * obs[j] + obs[j]
        dist = np.random.choice(noisy_obs, size=n, replace=False)
        sample[i] = np.percentile(dist, p)
    return sample


@jit(nopython=True)
def replacement_sampler(obs, n, step, p, nl1=3, nl2=0.05):
    """Bootstrap sampler with a two-phase noise model that changes at *step*.

    Args:
        obs: Observations to sample from.
        n: Sample size drawn per bootstrap iteration.
        step: Index at which noise level switches from *nl1* to *nl2*.
        p: Percentile to compute (0–100).
        nl1: Noise scaling coefficient for indices < *step*.
        nl2: Noise scaling coefficient for indices >= *step*.

    Returns:
        Array of 1000 percentile values from noise-added observations.
    """
    sample = np.zeros((1000,))
    for i in range(0, 1000):
        noisy_obs = np.empty_like(obs)
        for j in range(0, step):
            std = (5 + nl1 * obs[j]) / 3
            noisy_obs[j] = np.random.normal(0, std) + obs[j]
        for j in range(step, len(noisy_obs)):
            std = (5 + nl2 * obs[j]) / 3
            noisy_obs[j] = np.random.normal(0, std) + obs[j]
        dist = np.random.choice(noisy_obs, size=n, replace=False)
        sample[i] = np.percentile(dist, p)
    return sample


@jit(nopython=True)
def short_sampler(obs, n, p):
    """Bootstrap sampler using contiguous sub-segments of length *n*.

    Args:
        obs: Observation array.
        n: Sub-segment length.
        p: Percentile to compute from each sub-segment.

    Returns:
        Array of 1000 sub-segment percentile values.
    """
    sample = np.zeros((1000,))
    ind = np.arange(0, len(obs))
    for i in range(0, 1000):
        step = np.random.choice(ind[:-n])
        step_obs = obs[step : step + n]
        dist = np.random.choice(step_obs, size=n, replace=False)
        sample[i] = np.percentile(dist, p)
    return sample


@jit(nopython=True)
def stratified_sampler(obs, ids, n, p):
    """Stratified bootstrap sampler: draw *n* ids with replacement and compute the *p*-th percentile.

    Args:
        obs: 2-D observation array (rows = ids, cols = time steps).
        ids: 1-D array of stratum ids to sample from.
        n: Number of ids to sample per bootstrap iteration.
        p: Percentile to compute.

    Returns:
        Array of 1000 stratified-sample percentile values.
    """
    sample = np.zeros((1000,))
    for i in range(0, 1000):
        sub_ind = np.random.choice(ids, n, replace=True)
        dist = [obs[sub_ind[i], i] for i in range(0, obs.shape[1])]
        sample[i] = np.percentile(dist, p)
    return sample


@jit(nopython=True)
def multi_block_sampler(obs, ids, n, p):
    """Multi-block bootstrap: splice non-overlapping blocks and compute the *p*-th percentile.

    Args:
        obs: 1-D observation array.
        ids: 1-D index array for *obs*.
        n: Block size sliced from each random start position.
        p: Percentile to compute from each spliced distribution.

    Returns:
        Array of 1000 distribution-sample percentile values.
    """
    sample = np.zeros((1000,))
    nblocks = np.floor(len(obs) / (len(obs) - n))
    for i in range(0, 1000):
        dist = []
        for _block in nblocks:
            start = np.random.choice(ids, size=1, replace=False)
            dist.append(obs[start : start + (len(obs) - n)])
            ids = ids[:start] + ids[start + (len(obs) - n) :]
        sample[i] = np.percentile(dist, p)
    return sample


def find_continuous(df, gap_length=0):
    """Identify and tag continuous non-NaN groups in a DataFrame with 'A' and 'B' columns.

    Args:
        df: DataFrame with 'A' and 'B' measurement columns.
        gap_length: Maximum gap length (in rows) still considered part of a continuous segment.

    Returns:
        Tuple of (annotated DataFrame with 'group' column, group-count DataFrame).
    """
    df["group"] = (
        (df["A"].isnull() & df["B"].isnull())
        != (df["A"].shift().isnull() & df["B"].shift().isnull())
    ).cumsum()

    # count consecutive NaN groups
    df_na_n = pd.DataFrame(
        df[df["A"].isnull() & df["B"].isnull()]["group"].value_counts()
    ).reset_index()
    df_na_n.columns = ["group", "count"]

    # find group numbers of groups which are NOT 'small' gaps
    not_gaps = df_na_n[df_na_n["count"] > gap_length]["group"].values

    # recompute the groups after removing 'small' gaps
    df["group"] = np.where(
        df["group"].isin(not_gaps), np.nan, df["group"].ffill().bfill()
    )
    df["group"] = (
        (df["A"].isnull() & df["B"].isnull())
        != (df["A"].shift().isnull() & df["B"].shift().isnull())
    ).cumsum()

    # Assign NaN to the group number of the NaN values
    df.loc[df["A"].isnull() & df["B"].isnull(), "group"] = np.nan

    # Ignore NaN groups while counting
    df_group = pd.DataFrame.from_dict(
        {
            k: v
            for k, v in df[df["group"].notna()]
            .groupby("group")["group"]
            .count()
            .items()
        },
        orient="index",
        columns=["group_count"],
    )
    df_group = df_group.sort_values(by="group_count", ascending=False)

    return df, df_group


@njit
def agg_sampler(obs, agg_n, scale=0.1, agg_method=0):
    p = np.arange(1, 100, 1)
    sample = np.empty((1000, 99))

    for i in range(0, 1000):

        a_obs = obs.copy()
        b_obs = obs.copy()

        if agg_method > 0:
            s_obs = np.empty_like(a_obs)

        for j in range(len(obs)):

            if obs[j] < 50:
                var = 5 / 3
            else:
                var = obs[j] * (scale / 3)

            a_obs[j] += np.random.normal(loc=0, scale=var)

            if agg_method > 0:
                b_obs[j] += np.random.normal(loc=0, scale=var)

        if agg_method == 0:
            sample[i, :] = np.percentile(block_agg_np(a_obs, agg_n), p)

        elif agg_method == 1:
            s_obs = apply_mean(np.vstack((a_obs, b_obs)), 0)
            sample[i, :] = np.percentile(block_agg_np(s_obs, agg_n), p)

        elif agg_method == 2:
            sample[i, :] = np.percentile(
                block_agg_np2(np.vstack((a_obs, b_obs)), agg_n), p
            )

    return sample


@njit
def get_agg_sampler(arr_cont):
    percentile = np.arange(1, 100, 1)
    agg_n = np.arange(1, 169, 1)

    arr_sample_0 = np.empty((1000, len(percentile), len(agg_n)))
    arr_sample_1 = np.empty((1000, len(percentile), len(agg_n)))
    # arr_sample_2 = np.empty((1000, len(percentile), len(agg_n)))

    for j in range(len(agg_n)):
        arr_sample_0[:, :, j] = agg_sampler(obs=arr_cont, agg_n=agg_n[j], agg_method=0)
        arr_sample_1[:, :, j] = agg_sampler(obs=arr_cont, agg_n=agg_n[j], agg_method=1)
        # arr_sample_2[:, :, j] = agg_sampler(obs=arr_cont, agg_n=agg_n[j], agg_method=2)

    return arr_sample_0, arr_sample_1  # , arr_sample_2


@njit
def noisy_sampler(obs, n, noise_level=0.1):
    """Bootstrap sampler that adds channel noise before drawing a sample of size *n*.

    Args:
        obs: Observations to sample from.
        n: Sample size drawn per bootstrap iteration.
        noise_level: Noise standard deviation as a fraction of the observation value.

    Returns:
        Array of shape (99, 1000) — all percentiles for 1000 bootstrap draws.
    """
    percentile = np.arange(1, 100, 1)
    sample = np.empty((len(percentile), 1000))

    for i in range(0, 1000):

        a_obs = obs.copy()
        b_obs = obs.copy()

        for j in range(len(obs)):

            if obs[j] < 50:
                var = 5 / 3
            else:
                var = obs[j] * (noise_level / 3)

            a_obs[j] += np.random.normal(loc=0, scale=var)
            b_obs[j] += np.random.normal(loc=0, scale=var)
        noisy_obs = 0.5 * (a_obs + b_obs)
        dist = np.random.choice(noisy_obs, size=n, replace=False)
        sample[:, i] = np.percentile(dist, percentile)
    return sample


@njit
def get_sampler_bounds(arr_cont):
    """Compute noisy bootstrap percentile bounds across completeness levels (1%–99%).

    Args:
        arr_cont: 1-D continuous observation array.

    Returns:
        Array of shape (99, 99, 1000) — percentiles × completeness × bootstrap draws.
    """
    completeness = np.round(np.arange(0.01, 1, 0.01), 2)
    percentile = np.arange(1, 100, 1)

    n_samples = np.ceil(len(arr_cont) * completeness)

    arr_sample = np.empty((int(len(percentile)), int(len(completeness)), 1000))

    for j, n in enumerate(n_samples):
        arr_sample[:, j, :] = noisy_sampler(arr_cont, int(n))

    return arr_sample


@njit
def block_sampler(obs, ids, n, noise_level=0.1):
    """Block bootstrap sampler with channel noise: slice a contiguous block of length *n* and compute all percentiles.

    Args:
        obs: 1-D observation array.
        ids: 1-D index array for *obs*.
        n: Block length.
        noise_level: Noise fraction applied to each observation before sampling.

    Returns:
        Array of shape (99, 1000) — all percentiles for 1000 bootstrap draws.
    """
    percentile = np.arange(1, 100, 1)
    sample = np.empty((len(percentile), 1000))
    for i in range(0, 1000):
        a_obs = obs.copy()
        b_obs = obs.copy()

        for j in range(len(obs)):

            if obs[j] < 50:
                var = 5 / 3
            else:
                var = obs[j] * (noise_level / 3)

            a_obs[j] += np.random.normal(loc=0, scale=var)
            b_obs[j] += np.random.normal(loc=0, scale=var)
        noisy_obs = 0.5 * (a_obs + b_obs)
        start = np.random.choice(ids[:-n], size=1, replace=False)[0]
        dist = noisy_obs[start : start + n]
        sample[:, i] = np.percentile(dist, percentile)
    return sample


@njit
def get_block_sampler_bounds(arr_cont):
    """Compute block-bootstrap percentile bounds across completeness levels (1%–99%).

    Args:
        arr_cont: 1-D continuous observation array.

    Returns:
        Array of shape (99, 99, 1000) — percentiles × completeness × bootstrap draws.
    """

    arr_ids = np.arange(0, len(arr_cont))

    completeness = np.round(np.arange(0.01, 1, 0.01), 2)
    percentile = np.arange(1, 100, 1)

    # Calculate n_samples once for each completeness value
    n_samples = np.floor(len(arr_cont) * completeness)

    arr_sample = np.empty((len(percentile), len(completeness), 1000))

    for j, n in enumerate(n_samples):
        arr_sample[:, j, :] = block_sampler(arr_cont, arr_ids, int(n))

    return arr_sample


@njit
def tradeoff_sampler(obs, noise_level, thresh):
    sample = np.empty(1000)
    for i in range(0, 1000):
        noisy_obs = np.zeros_like(obs)
        while np.count_nonzero(noisy_obs) == 0:
            a_obs = obs.copy()
            b_obs = obs.copy()
            for j, _ in enumerate(obs):
                b_obs[j] = b_obs[j] + np.abs(
                    np.random.normal(loc=0, scale=b_obs[j] * noise_level)
                )
            noisy_obs = 0.5 * (a_obs + b_obs)
            disagree = (2 * np.abs(a_obs - b_obs)) / (a_obs + b_obs)
            noisy_obs = np.where(disagree < thresh, noisy_obs, np.nan)
        sample[i] = np.nanmedian(noisy_obs)
    return sample


@njit
def get_tradeoff_sampler(arr_cont):
    threshold = np.round(np.arange(0.01, 1.00, 0.01), 2)
    noise = np.round(np.arange(0.01, 1.00, 0.01), 2)

    arr_sample = np.empty((len(threshold), len(noise), 1000))

    for i, t in enumerate(threshold):
        for j, n in enumerate(noise):
            arr_sample[i, j, :] = tradeoff_sampler(arr_cont, n, t)

    return arr_sample


@njit
def agreement_sampler(obs, n, loc=0, scale=0.1, percentile="all", thresh=None):
    if percentile == "all":
        percentile = np.arange(1, 100, 1)
    sample = np.empty((len(percentile), 1000))
    for i in range(0, 1000):
        a_obs = np.copy(obs)
        b_obs = np.copy(obs)
        for j, _ in enumerate(obs):
            if (loc == 0) & (scale > 0) & (scale < 1):
                a_obs[j] = np.random.normal(loc=a_obs[j], scale=a_obs[j] * scale)
                b_obs[j] = np.random.normal(loc=b_obs[j], scale=b_obs[j] * scale)
            elif (loc == 0) & (scale > 1):
                a_obs[j] = np.random.normal(loc=a_obs[j], scale=scale)
                b_obs[j] = np.random.normal(loc=a_obs[j], scale=scale)
            elif (loc > 0) & (loc < 1) & (scale == 0):
                a_obs[j] = a_obs[j] * loc + a_obs[j]
            elif (loc > 0) & (loc < 1) & (scale > 0) & (scale < 1):
                a_obs[j] = np.random.normal(
                    loc=a_obs[j] + a_obs[j] * loc, scale=a_obs[j] * scale
                )
                b_obs[j] = np.random.normal(loc=b_obs[j], scale=b_obs[j] * scale)
            elif (loc > 0) & (loc < 1) & (scale > 1):
                a_obs[j] = np.random.normal(loc=a_obs[j] + a_obs[j] * loc, scale=scale)
                b_obs[j] = np.random.normal(loc=b_obs[j], scale=scale)
            elif (loc > 1) & (scale == 0):
                a_obs[j] = a_obs[j] + loc
            elif (loc > 1) & (scale > 0) & (scale < 1):
                a_obs[j] = np.random.normal(loc=a_obs[j] + loc, scale=a_obs[j] * scale)
                b_obs[j] = np.random.normal(loc=b_obs[j], scale=b_obs[j] * scale)
            else:
                a_obs[j] = np.randoms.normal(loc=a_obs[j] + loc, scale=scale)
                b_obs[j] = np.random.normal(loc=b_obs[j], scale=scale)
        noisy_obs = 0.5 * (a_obs + b_obs)
        disagree = (2 * (((a_obs - b_obs) ** 2) ** 0.5)) / (a_obs + b_obs)
        if thresh is not None:
            noisy_obs = drop_nan(np.where(disagree <= thresh, noisy_obs, np.nan))
        if n != len(obs):
            dist = np.random.choice(noisy_obs, size=int(n), replace=False)
        else:
            dist = noisy_obs
        sample[:, i] = np.percentile(dist, percentile)
    return sample


@njit
def get_agreement_wander(arr_cont):
    agreement_level = np.round(np.arange(0.01, 1.00, 0.01), 2)
    percentile = np.arange(1, 100, 1)

    arr_sample = np.empty((len(percentile), len(agreement_level), 1000))
    for j, _s in enumerate(agreement_level):
        arr_sample[:, j, :] = agreement_sampler(
            arr_cont, len(arr_cont) - 1, loc=0, scale=agreement_level[j]
        )

    return arr_sample


@njit
def get_agreement_fixed(arr_cont):
    agreement_level = np.round(np.arange(0.01, 1, 0.01), 2)
    percentile = np.arange(1, 100, 1)

    arr_sample = np.empty((len(percentile), len(agreement_level), 1000))

    for j, level in enumerate(agreement_level):
        arr_sample[:, j, :] = agreement_sampler(
            arr_cont, len(arr_cont) - 1, loc=level, scale=0.1
        )

    return arr_sample


@njit
def get_joint_sampler(arr_cont):
    """Compute joint agreement×threshold bootstrap samples at the median.

    Args:
        arr_cont: 1-D continuous observation array.

    Returns:
        Array of shape (n_thresholds, n_noise_levels, 1, 1000).
    """

    threshold = np.round(np.arange(0, 1, 0.01), 2)
    noise_level = np.array([0.1])  # np.round(np.arange(0, 1, 0.01), 2)
    percentile = np.array([50])

    arr_sample = np.empty((len(threshold), len(noise_level), len(percentile), 1000))

    print("hi mom!")

    for i, t in enumerate(threshold):
        for j, o in enumerate(noise_level):
            arr_sample[i, j, :, :] = agreement_sampler(
                arr_cont, n=arr_cont.shape[0], thresh=t, scale=o, percentile=percentile
            )

    return arr_sample


def get_noisy_sampler_bounds(df_cont):
    """Compute 2.5 / 50 / 97.5 percentile bounds from noisy bootstrap across noise levels.

    Args:
        df_cont: DataFrame with a 'mu' column of continuous observations.

    Returns:
        Tuple of (lower, median, upper) DataFrames indexed by percentile, columns = noise levels.
    """

    arr_cont = df_cont.mu.values

    noise_level = np.round(np.arange(0, 3, 0.03), 2)
    percentile = np.arange(1, 100, 1)

    arr_sample = np.empty((len(percentile), len(noise_level), 1000))

    for j, n in enumerate(noise_level):
        arr_sample[:, j, :] = noisy_sampler(arr_cont, len(arr_cont) - 1, n)

    df_lower = pd.DataFrame(
        np.percentile(arr_sample, 2.5, axis=2), index=percentile, columns=noise_level
    )
    df_med = pd.DataFrame(
        np.percentile(arr_sample, 50, axis=2), index=percentile, columns=noise_level
    )
    df_upper = pd.DataFrame(
        np.percentile(arr_sample, 97.5, axis=2), index=percentile, columns=noise_level
    )

    return df_lower, df_med, df_upper


def get_replacement_sampler(df_cont):
    """Compute 2.5 / 50 / 97.5 percentile bounds from replacement bootstrap across replacement steps.

    Args:
        df_cont: DataFrame with a 'mu' column of continuous observations.

    Returns:
        Tuple of (lower, median, upper) DataFrames indexed by percentile, columns = replacement steps.
    """
    arr_cont = df_cont.mu.values

    replacement_step = np.arange(1, len(arr_cont) - 2)
    percentile = np.arange(1, 100, 1)

    arr_sample = np.empty((len(percentile), len(replacement_step), 1000))

    for i, p in enumerate(percentile):
        for j, n in enumerate(replacement_step):
            arr_sample[i, j, :] = replacement_sampler(arr_cont, len(arr_cont) - 1, n, p)

    df_lower = pd.DataFrame(
        np.percentile(arr_sample, 2.5, axis=2),
        index=percentile,
        columns=replacement_step,
    )
    df_med = pd.DataFrame(
        np.percentile(arr_sample, 50, axis=2),
        index=percentile,
        columns=replacement_step,
    )
    df_upper = pd.DataFrame(
        np.percentile(arr_sample, 97.5, axis=2),
        index=percentile,
        columns=replacement_step,
    )
    return df_lower, df_med, df_upper
