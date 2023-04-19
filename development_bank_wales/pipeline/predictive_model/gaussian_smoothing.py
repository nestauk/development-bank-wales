# File: development_bank_wales/pipeline/predictive_model/gaussian_smoothing.py
"""Gaussian smoothing for modelling competence."""

# ---------------------------------------------------------------------------------

import numpy as np
import scipy

# ---------------------------------------------------------------------------------


def pcl_gaussian_smooth(dists, vals, sg_percentile=1.7):
    """Gaussian smoothing for given features distances and values.

    Args:
        dists (np.array): ndarray of shape (n_pts, n_pts)
            Squareform of all pairwise distances between points of the point
            cloud, as computed e.g. by scipy.spatial.distance.pdist.
        vals (np.array): ndarray of shape (n_pts, n_dims)
             Values to be smoothed defined for each point. The different dimensions
            are smoothed independently.
        sg_percentile (float, optional): To determine the sigma (sg) of the Gaussian, the n-th percentile
            of all distances in dists is used. This value specifies that n. Defaults to 1.7.

    Returns:
        smooth: ndarray of shape (n_pts, n_dims). Smoothed version of `vals`.
    """

    # Generate Gaussian function for smoothing
    def gaussian_factory(mu, sg):
        gaussian = (
            lambda x: 1
            / (sg * np.sqrt(2.0 * np.pi))
            * np.exp(-1 / 2 * ((x - mu) / sg) ** 2.0)
        )
        return gaussian

    gaussian_func = gaussian_factory(0.0, np.percentile(dists, sg_percentile))

    # Smoothen the distances
    gaud = gaussian_func(dists)

    # Use smoothened distances to smoothen values
    smooth = np.empty_like(vals)
    for dim in range(vals.shape[1]):
        smooth[:, dim] = np.sum(gaud * vals[:, dim], axis=1) / np.sum(gaud, axis=1)

    # Done
    return smooth


def get_smoothed_labels(
    features, labels, n_samples=10000, flip_indices=False, random_state=42
):
    """Compute feature distances and get smoothed labels, given features and labels.
    Optionally, flip True samples to False to study effect.

    Args:
        features (np.array): Features for creating feature space and computing distances.
        labels (np.array): Original labels to be smoothed.
        n_samples (int, optional): Number of samples to use.
            Caution: computing distances is computationally expensive. Defaults to 10000.
        flip_indices (bool, optional): Whether or not to flip every 1/100 of all True samples to False.
            Defaults to False.
        random_state (int, optional): Random state for consistent results. Defaults to 42.

    Returns:
        smoothed_labels: np.array
            Smoothed labels.
        labels: np.array
            Updated labels. If filp_indices is True then every 100th sample is flipped.
        flipped_indices: np.array
            Indices for flipped samples.

    """

    rand = np.random.RandomState(random_state)

    features = features[:n_samples]
    labels = labels[:n_samples]

    # Flip 1/100 of True samples from True to False
    if flip_indices:

        flipped_indices = rand.choice(
            np.where(labels)[0], round(n_samples / 100), replace=False
        )
        labels[flipped_indices] = False

    else:
        flipped_indices = []

    # Reshape labels
    labels = labels.astype(float)
    labels = labels.reshape((-1, 1))

    # Compute feature distance and get smoothed labels
    dists = scipy.spatial.distance.pdist(features)
    dists = scipy.spatial.distance.squareform(dists)
    smoothed_labels = pcl_gaussian_smooth(dists, labels, sg_percentile=1.7)

    return smoothed_labels, labels, flipped_indices
