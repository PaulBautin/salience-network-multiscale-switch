"""Intracortical T1 profiles and the microstructure profile covariance (MPC) gradient."""
import logging

import nibabel as nib
import numpy as np
from brainspace.gradient.gradient import GradientMaps
from scipy.stats import zscore

logger = logging.getLogger(__name__)


def load_t1_salience_profiles(t1_files: list, mask: np.ndarray) -> np.ndarray:
    """
    Load T1 intensity profiles for a pre-masked set of vertices across all subjects.

    Parameters
    ----------
    t1_files : list
        List of paths to .gii profile files, one per subject.
    mask : np.ndarray of bool, shape (n_vertices,)
        Boolean vertex mask selecting the vertices to load (e.g. from compute_network_mask).

    Returns
    -------
    t1_stack : np.ndarray
        Stack of profiles with shape (n_subjects, n_depths, n_network_vertices).
    """
    n_files = len(t1_files)
    if n_files == 0:
        raise FileNotFoundError("No files found")
    if not np.any(mask):
        raise ValueError("mask is all-False: no vertices selected.")
    logger.info(f"Loading profiles for {n_files} subjects...")
    t1_salience_profiles = np.stack([nib.load(f).darrays[0].data[:, mask] for f in t1_files])
    logger.info(f"Final array shape: {t1_salience_profiles.shape}")
    return t1_salience_profiles


def compute_t1map(t1_salience_profiles: np.ndarray) -> np.ndarray:
    """Return the z-scored mean profile collapsed over subjects and depths.

    Parameters
    ----------
    t1_salience_profiles : np.ndarray, shape (n_subjects, n_depths, n_vertices)
        Pre-masked T1 profiles for the network of interest.

    Returns
    -------
    np.ndarray, shape (n_vertices,)
    """
    return zscore(np.mean(t1_salience_profiles, axis=(0, 1)), nan_policy='omit')


def partial_corr_with_covariate(X: np.ndarray, covar: np.ndarray) -> np.ndarray:
    """
    Compute the Fisher z-transformed partial correlation matrix between 
    vertices, controlling for a single covariate.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_features, n_vertices)
        Data matrix (e.g., intensity profiles across depths).
    covar : np.ndarray, shape (n_features,)
        Covariate to control for (e.g., the mean spatial profile).
    
    Returns
    -------
    MPC : np.ndarray, shape (n_vertices, n_vertices)
        Fisher z-transformed partial correlation matrix.
    """
    n_features, _ = X.shape
    # Design matrix: intercept + raw covariate 
    X_covar = np.column_stack([np.ones(n_features), covar])
    # Regression for all vertices at once 
    beta, _, _, _ = np.linalg.lstsq(X_covar, X, rcond=None)
    # Calculate residuals
    residuals = X - (X_covar @ beta)              
    # Correlation matrix of residuals across vertices
    R = np.corrcoef(residuals, rowvar=False) 
    # Fisher z-transform with safe error state handling
    with np.errstate(divide='ignore', invalid='ignore'):
        MPC = np.arctanh(R)
        MPC = np.nan_to_num(MPC, nan=0, posinf=0, neginf=0) 
    return MPC


def compute_t1_gradient(
    t1_salience_profiles: list | np.ndarray,
    n_components: int = 10,
    sparsity: float = 0.9,
    random_state: int = 0,
    return_lambdas: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute T1 MPC gradients and return the z-scored first component.

    The share of the first gradient, lambda_1 / sum(lambda) over the ``n_components``
    computed components (the relative eigenvalue usually reported as its variance
    explained), is logged as its mean, SD and range across subjects.

    Parameters
    ----------
    t1_salience_profiles : list or np.ndarray, shape (n_subjects, n_depths, n_vertices)
        Pre-masked T1 profiles for the network of interest.
    n_components : int, default=10
        Number of gradient components to extract.
    sparsity : float, default=0.9
        Sparsity threshold for GradientMaps.
    random_state : int, default=0
        Seed for the diffusion-map embedding, pinned for run-to-run
        reproducibility. Note the eigenvector *sign* is mathematically arbitrary
        and is not fixed by the seed; downstream code treats signs as produced.
    return_lambdas : bool, default=False
        Also return the per-subject eigenvalues.

    Returns
    -------
    np.ndarray, shape (n_vertices,)
        Z-scored first gradient component.
    lambdas : np.ndarray, shape (n_subjects, n_components)
        Per-subject diffusion-map eigenvalues, if ``return_lambdas``.
    """
    logger.info("Computing T1 gradients...")
    # Calculate the mean profile for each subject across all vertices (axis=2)
    t1_mean_profiles = np.nanmean(t1_salience_profiles, axis=2)

    # Compute MPC for each subject cleanly
    t1_salience_mpc = [
        partial_corr_with_covariate(subj_data, covar=mean_profile)
        for subj_data, mean_profile in zip(t1_salience_profiles, t1_mean_profiles)
    ]

    # Fit GradientMaps
    gm_t1 = GradientMaps(
        n_components=n_components,
        random_state=random_state,
        approach='dm',
        kernel='normalized_angle',
        alignment='procrustes'
    )
    gm_t1.fit(t1_salience_mpc, sparsity=sparsity)

    # Extract and log gradient lambdas
    t1_gradients = np.mean(np.asarray(gm_t1.aligned_), axis=0)
    lambdas = np.asarray(gm_t1.lambdas_, dtype=float)  # (n_subjects, n_components)
    logger.info(f"Gradient lambdas: {lambdas.mean(axis=0)}")
    share = lambdas[:, 0] / lambdas.sum(axis=1)
    logger.info(f"Gradient 1 variance explained (lambda_1 / sum of {lambdas.shape[1]} lambdas): "
                f"mean {share.mean():.1%} (SD {share.std(ddof=1) if share.size > 1 else 0:.1%}, "
                f"range {share.min():.1%}-{share.max():.1%}) across {share.size} subjects")

    gradient = zscore(t1_gradients[:, 0], nan_policy='omit')
    return (gradient, lambdas) if return_lambdas else gradient
