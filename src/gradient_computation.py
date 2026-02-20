import logging
import numpy as np
from brainspace.gradient.gradient import GradientMaps
from scipy.stats import zscore

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

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
    df_yeo_surf, 
    t1_salience_profiles: list | np.ndarray, 
    network: str = 'SalVentAttn',
    n_components: int = 10,
    sparsity: float = 0.9
):
    """
    Compute T1 gradients from MPC and map them to a surface dataframe.
    
    Parameters
    ----------
    df_yeo_surf : pandas.DataFrame
        DataFrame containing network assignments.
    t1_salience_profiles : list or np.ndarray, shape (n_subjects, n_features, n_vertices)
        Subject profile data.
    network : str, default='SalVentAttn'
        Target network to map the gradients to.
    n_components : int, default=10
        Number of gradient components to extract.
    sparsity : float, default=0.9
        Sparsity threshold for GradientMaps.
        
    Returns
    -------
    df_yeo_surf : pandas.DataFrame
        The updated DataFrame.
    """
    logging.info(f"Computing T1 gradients for {network}...")
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
        random_state=None, 
        approach='dm', 
        kernel='normalized_angle', 
        alignment='procrustes'
    )
    gm_t1.fit(t1_salience_mpc, sparsity=sparsity)
    
    # Extract and log gradient lambdas
    t1_gradients = np.mean(np.asarray(gm_t1.aligned_), axis=0)
    mean_lambdas = np.mean(np.asarray(gm_t1.lambdas_), axis=0)
    logging.info(f"Gradient lambdas: {mean_lambdas}")
    
    # Update the dataframe
    df_out = df_yeo_surf.copy()
    mask = df_out['network'].eq(network)
    df_out.loc[mask, f't1_gradient1_{network}'] = zscore(t1_gradients[:, 0], nan_policy='omit')
    
    return df_out