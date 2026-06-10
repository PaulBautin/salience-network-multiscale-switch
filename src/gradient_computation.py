import logging

import numpy as np
from brainspace.gradient.gradient import GradientMaps
from scipy.stats import zscore

logger = logging.getLogger(__name__)


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
) -> np.ndarray:
    """
    Compute T1 MPC gradients and return the z-scored first component.

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

    Returns
    -------
    np.ndarray, shape (n_vertices,)
        Z-scored first gradient component.
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
    mean_lambdas = np.mean(np.asarray(gm_t1.lambdas_), axis=0)
    logger.info(f"Gradient lambdas: {mean_lambdas}")

    return zscore(t1_gradients[:, 0], nan_policy='omit')