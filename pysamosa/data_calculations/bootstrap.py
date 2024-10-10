"""
Bootstrap Methods
Last Updated: May 9, 2024
Runs bootstrapping scenarios for data experimentation with JIT.
@author: markjcampmier
"""

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
    """
    JIT-compatible averaging function

    Args:
        arr (np.array): Array to be averaged
        axis (int): Axis to be averaged

    Returns:
        (float): Averaged value
    """
    return arr.sum(axis=axis) / arr.shape[axis]


@njit
def block_agg_np(array, block_size):
    """
    JIT-compatible block aggregation function

    Args:
        array (np.array): Array to be aggregated
        block_size (int): Block size
    Returns:
        (np.array): Chunk aggregated array
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

    bins = np.linspace(bin_min,
                       bin_max,
                       n_bins)

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

    bins = np.linspace(bin_min,
                       bin_max,
                       n_bins)

    pk, _ = np.histogram(p_dist, bins=bins)
    pk = np.divide(pk, np.sum(pk))

    qk, _ = np.histogram(q_dist, bins=bins)
    qk = np.divide(qk, np.sum(qk))
    qk = np.add(qk, 0.0000000000001)

    filtered_indices = pk != 0
    return np.nansum(pk[filtered_indices] * np.log(pk[filtered_indices] / qk[filtered_indices]))


@jit(nopython=True)
def sampler(obs, n, p):
    """
    Sampler function to generate a sample of the given observations.

    The function generates a sample of size 'n' from the specified observations 'obs'
    and computes the p-th percentile for each sample. This process is then
    repeated a total of 1000 times.

    Args:
        obs (array-like): The observations from which to generate samples.
        n (int): The number of samples to generate.
        p (float): The percentile to compute for each sample.

    Returns:
        array_like: A numpy array of shape (1000,), representing the computed
        percentiles of the samples.
    """
    sample = np.empty(1000)
    for i in range(0, 1000):
        dist = np.random.choice(obs, size=n, replace=False)
        sample[i] = np.percentile(dist, p)
    return sample


@jit(nopython=True)
def drift_sampler(obs, n, p, dx):
    """
    Generates a sample array from a given observation where each observation is subjected to noise and drift.

    Generates noisy observations and then computing the `p` percentile of those chosen elements.
    The noise applied to observations has a mean of 0 and a drift component that increases
    linearly with the observed value, scaled by factor `dx`.

    Args:
        obs (numpy.ndarray): The input observation to be sampled. A 1D `numpy` array.
        n (int): The number of elements to choose randomly without replacement for each sample.
        p (float): The percentile to calculate on the selected elements for each sample.
        dx (float): The scaling factor for the drift component in the noise applied to observations.

    Returns:
        numpy.ndarray: A `numpy` array of size 1000, each element being the `p` percentile of `n` randomly chosen
        elements from the noisy observations.
    """
    sample = np.zeros((1000,))
    for i in range(0, 1000):
        x = 0
        noisy_obs = np.zeros(len(obs), )
        for j in range(0, len(obs)):
            x += dx
            noisy_obs[j] = np.random.normal(0, x) * obs[j] + obs[j]
        dist = np.random.choice(noisy_obs, size=n, replace=False)
        sample[i] = np.percentile(dist, p)
    return sample


@jit(nopython=True)
def replacement_sampler(obs, n, step, p, nl1=3, nl2=0.05):
    """
    Generates a random noise-added sample from the given observations,
    calculate the p-th percentile of these new observations to form a sample,
    and repeats this process to create an array of such percentiles.

    Args:
        obs (np.ndarray): A numpy array of observations from which to draw samples.
        n (int): The size of the sample to draw.
        step (int): The point at which the noise level changes within the observation array.
        p (int or float): The percentile value to compute, which must be between 0 and 100 inclusive.
        nl1 (float, optional): The standard deviation of the normal distribution to draw noise from for
          observation indices less than 'step'. Defaults to 0.6.
        nl2 (float, optional): The standard deviation of the normal distribution to draw noise from for
          observation indices greater or equal to 'step'. Defaults to 0.

    Returns:
        np.ndarray: A numpy array of size 1000 containing the percentiles of noise-added observations.

    Notes:
        The function is decorated with numba's jit decorator with nopython mode for performance improvement.
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
    """
    Generates a sample of a given size by randomly selecting sub-segments of the input
    observation, determining a percentile of each segment, and composing the output sample
    from these percentiles.

    Args:
        obs (np.ndarray): A numpy array containing the observations.
        n (int): The size of samples to be drawn from 'obs'. Corresponds to the length of
            each sub-segment that is randomly selected from obs.
        p (int or float): The percentile to extract from each sub-segment of 'obs'. This
            percentile measurement of the selected sub-segment forms one element of the
            returned sample.

    Returns:
        np.ndarray: A numpy array of size 1000 consisting of the 'p'-th percentile of 'n'
            sized random sub-segments from the 'obs' array.
        """
    sample = np.zeros((1000,))
    ind = np.arange(0, len(obs))
    for i in range(0, 1000):
        step = np.random.choice(ind[:-n])
        step_obs = obs[step:step + n]
        dist = np.random.choice(step_obs, size=n, replace=False)
        sample[i] = np.percentile(dist, p)
    return sample


@jit(nopython=True)
def stratified_sampler(obs, ids, n, p):
    """Performs a stratified sampling on data, and then calculates the percentile.

    This function, optimized with Numba's jit decorator for speed, performs stratified
    sampling on the input data and then calculates and returns percentiles.

    Args:
        obs (numpy.array): A 2D numpy array containing the data observations.
        ids (numpy.array): A 1D numpy array containing the ID values of the observations.
        n (int): The number of sampling to perform on the 'ids' array
        p (int or float): The percentile to extract from each sub-segment of 'obs'. This
                          percentile measurement of the selected sub-segment forms one element of the
                          returned sample.

    Returns:
        numpy.array: A numpy array of size 1000 with the calculated percentiles of the
        stratified sample.
    """
    sample = np.zeros((1000,))
    for i in range(0, 1000):
        sub_ind = np.random.choice(ids, n, replace=True)
        dist = [obs[sub_ind[i], i] for i in range(0, obs.shape[1])]
        sample[i] = np.percentile(dist, p)
    return sample


@jit(nopython=True)
def multi_block_sampler(obs, ids, n, p):
    """
    Generate a distribution sample based on slicing observation data into multiple blocks.

    This function performs multi-block sampling on given observation data. The function
    computes the number of blocks, then for each block, randomly selects a position
    from the observation ids, slices a block of data from that position, and
    calculates a percentile of that sliced block. This process is repeated 1000 times
    to generate a distribution sample.

    Args:
        obs (np.array): 1-D array of observation data.
        ids (np.array): 1-D array containing indices of the observation data.
        n (int): The size of the block to be sliced from the observation data.
        p (float): The percentile of the sliced block data to be calculated and
            added to the sample.

    Returns:
        np.array: A 1-Dimensional array of size 1000 containing distribution
            sample
    """
    sample = np.zeros((1000,))
    nblocks = np.floor(len(obs) / (len(obs) - n))
    for i in range(0, 1000):
        dist = []
        for block in nblocks:
            start = np.random.choice(ids, size=1, replace=False)
            dist.append(obs[start:start + (len(obs) - n)])
            ids = ids[:start] + ids[start + (len(obs) - n):]
        sample[i] = np.percentile(dist, p)
    return sample


def find_continuous(df, gap_length=0):
    """
        This function identifies and tags continuous groups with non-NaN values in the given DataFrame.
    It allows for gaps of a defined length, and assigns NaN group number for NaN values in the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame with 'A' and 'B' columns that needs to be
                               analysed for continuous non-NaN segments.
        gap_length (int, optional): The maximum gap length allowed within a continuous non-NaN segment.
                                    Defaults to 0, i.e., no gap allowed.

    Returns:
        df (pandas.DataFrame): The input DataFrame enriched with a new 'group' column storing
                               the group number for each non-NaN segment.
        df_group (pandas.DataFrame): A DataFrame where the index are the group numbers and
                                    the values are the count of values belonging to each group.
    """
    df['group'] = (
            (df['A'].isnull() & df['B'].isnull()) != (df['A'].shift().isnull() & df['B'].shift().isnull())).cumsum()

    # count consecutive NaN groups
    df_na_n = pd.DataFrame(df[df['A'].isnull() & df['B'].isnull()]['group'].value_counts()).reset_index()
    df_na_n.columns = ['group', 'count']

    # find group numbers of groups which are NOT 'small' gaps
    not_gaps = df_na_n[df_na_n['count'] > gap_length]['group'].values

    # recompute the groups after removing 'small' gaps
    df['group'] = np.where(df['group'].isin(not_gaps), np.nan, df['group'].ffill().bfill())
    df['group'] = (
            (df['A'].isnull() & df['B'].isnull()) != (df['A'].shift().isnull() & df['B'].shift().isnull())).cumsum()

    # Assign NaN to the group number of the NaN values
    df.loc[df['A'].isnull() & df['B'].isnull(), 'group'] = np.nan

    # Ignore NaN groups while counting
    df_group = pd.DataFrame.from_dict(
        {k: v for k, v in df[df['group'].notna()].groupby('group')['group'].count().items()},
        orient='index', columns=['group_count'])
    df_group = df_group.sort_values(by='group_count', ascending=False)

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
            sample[i, :] = np.percentile(block_agg_np2(np.vstack((a_obs, b_obs)), agg_n), p)

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
    """
    Noisy sampler function to generate a sample of the given observations but with some noise.

    The function generates a sample of size 'n' from the specified observations 'obs',
    after adding specific noise data. After noise addition, the p-th percentile
    for each sample is computed. This process is then repeated a total of 1000 times.

    Args:
        obs (array-like): The observations from which to generate samples.
        noise_level (float): The standard deviation of the Normal distribution
            from which to draw the noise.
        n (int): The number of samples to generate.

    Returns:
        array-like: A numpy array of shape (1000,), representing the computed percentiles
        of the samples with the added noise.
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
    """
    Calculate the percentile bounds of a data set divided by a group index.

    Args:
        arr_cont (array): The input array.

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame]: A tuple containing three pandas DataFrame objects.
            - The first DataFrame contains the lower percentile bounds, with percentile value as index
                and completeness as columns.
            - The second DataFrame contains the median percentile bounds, with percentile value as index
                and completeness as columns.
            - The third DataFrame contains the upper percentile bounds, with percentile value as index
                and completeness as columns.

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
    """
    Generate a distribution sample based on slicing observation data into blocks.

    This function performs block sampling on given observation data. The function
    randomly selects a position from the observation data, slices a block of data
    from that position, and calculates a percentile of that sliced block. This
    process is repeated 1000 times to generate a distribution sample.

    Args:
        obs (np.array): 1-D array of observation data.
        ids (np.array): 1-D array containing indices of the observation data.
        n (int): The size of the block to be sliced from the observation data.
        noise_level (float): The standard deviation of the Normal distribution
            from which to draw the noise.

    Returns:
        np.array: A 1-Dimensional array of size 1000 containing distribution
            sample.
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
        dist = noisy_obs[start:start + n]
        sample[:, i] = np.percentile(dist, percentile)
    return sample


@njit
def get_block_sampler_bounds(arr_cont):
    """
    Calculate the percentile bounds of a data set divided by a group index.

    Args:
        arr_cont (array): The input DataFrame. Should contain a 'mu' column.

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame]: A tuple containing three pandas DataFrame objects.
            - The first DataFrame contains the lower percentile bounds, with percentile value as index
                and completeness as columns.
            - The second DataFrame contains the median percentile bounds, with percentile value as index
                and completeness as columns.
            - The third DataFrame contains the upper percentile bounds, with percentile value as index
                and completeness as columns.

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
                b_obs[j] = b_obs[j] + np.abs(np.random.normal(loc=0, scale=b_obs[j] * noise_level))
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
def agreement_sampler(obs, n, loc=0, scale=0.1, percentile='all', thresh=None):
    if percentile == 'all':
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
                a_obs[j] = np.random.normal(loc=a_obs[j] + a_obs[j] * loc, scale=a_obs[j] * scale)
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
    for j, s in enumerate(agreement_level):
        arr_sample[:, j, :] = agreement_sampler(arr_cont, len(arr_cont) - 1,
                                                loc=0, scale=agreement_level[j])

    return arr_sample


@njit
def get_agreement_fixed(arr_cont):
    agreement_level = np.round(np.arange(0.01, 1, 0.01), 2)
    percentile = np.arange(1, 100, 1)

    arr_sample = np.empty((len(percentile), len(agreement_level), 1000))

    for i, p in enumerate(percentile):
        for j, s in enumerate(agreement_level):
            arr_sample[i, j, :] = agreement_sampler(arr_cont, len(arr_cont) - 1, p,
                                                    loc=agreement_level[j], scale=0.1)

    return arr_sample


@njit
def get_joint_sampler(arr_cont):
    """
    Calculate the percentile bounds of a data set divided by a group index.

    Args:
        arr_cont (np.array): The input array.

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame]: A tuple containing three pandas DataFrame objects.
            - The first DataFrame contains the lower percentile bounds, with percentile value as index
                and completeness as columns.
            - The second DataFrame contains the median percentile bounds, with percentile value as index
                and completeness as columns.
            - The third DataFrame contains the upper percentile bounds, with percentile value as index
                and completeness as columns.

    """

    threshold = np.round(np.arange(0, 1, 0.01), 2)
    noise_level = np.array([0.1])  # np.round(np.arange(0, 1, 0.01), 2)
    percentile = np.array([50])

    arr_sample = np.empty((len(threshold), len(noise_level), len(percentile), 1000))

    print('hi mom!')

    for i, t in enumerate(threshold):
        for j, o in enumerate(noise_level):
            arr_sample[i, j, :, :] = agreement_sampler(arr_cont, n=arr_cont.shape[0],
                                                       thresh=t, scale=o,
                                                       percentile=percentile)

    return arr_sample


def get_noisy_sampler_bounds(df_cont):
    """
    Calculate the percentile bounds of a data set divided by a group index.

    Args:
        df_cont (DataFrame): The input DataFrame. Should contain a 'mu' column.

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame]: A tuple containing three pandas DataFrame objects.
            - The first DataFrame contains the lower percentile bounds, with percentile value as index
                and completeness as columns.
            - The second DataFrame contains the median percentile bounds, with percentile value as index
                and completeness as columns.
            - The third DataFrame contains the upper percentile bounds, with percentile value as index
                and completeness as columns.

    """

    arr_cont = df_cont.mu.values

    noise_level = np.round(np.arange(0, 3, 0.03), 2)
    percentile = np.arange(1, 100, 1)

    arr_sample = np.empty((len(percentile), len(noise_level), 1000))

    for i, p in enumerate(percentile):
        for j, n in enumerate(noise_level):
            arr_sample[i, j, :] = noisy_sampler(arr_cont, n, len(arr_cont) - 1, p)

    df_lower = pd.DataFrame(np.percentile(arr_sample, 2.5, axis=2), index=percentile, columns=noise_level)
    df_med = pd.DataFrame(np.percentile(arr_sample, 50, axis=2), index=percentile, columns=noise_level)
    df_upper = pd.DataFrame(np.percentile(arr_sample, 97.5, axis=2), index=percentile, columns=noise_level)

    return df_lower, df_med, df_upper


def get_replacement_sampler(df_cont):
    """
    Calculate the percentile bounds of a data set divided by a group index.

    Args:
        df_cont (DataFrame): The input DataFrame. Should contain a 'mu' column.

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame]: A tuple containing three pandas DataFrame objects.
            - The first DataFrame contains the lower percentile bounds, with percentile value as index
                and completeness as columns.
            - The second DataFrame contains the median percentile bounds, with percentile value as index
                and completeness as columns.
            - The third DataFrame contains the upper percentile bounds, with percentile value as index
                and completeness as columns.

    """
    arr_cont = df_cont.mu.values

    replacement_step = np.arange(1, len(arr_cont) - 2)
    percentile = np.arange(1, 100, 1)

    arr_sample = np.empty((len(percentile), len(replacement_step), 1000))

    for i, p in enumerate(percentile):
        for j, n in enumerate(replacement_step):
            arr_sample[i, j, :] = replacement_sampler(arr_cont, len(arr_cont) - 1, n, p)

    df_lower = pd.DataFrame(np.percentile(arr_sample, 2.5, axis=2), index=percentile, columns=replacement_step)
    df_med = pd.DataFrame(np.percentile(arr_sample, 50, axis=2), index=percentile, columns=replacement_step)
    df_upper = pd.DataFrame(np.percentile(arr_sample, 97.5, axis=2), index=percentile, columns=replacement_step)
    return df_lower, df_med, df_upper
