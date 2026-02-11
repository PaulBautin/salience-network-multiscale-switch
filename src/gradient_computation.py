import numpy as np


def partial_corr_with_covariate(X, covar):
    """
    Compute partial correlation matrix between vertices, controlling for covariate.
    
    Parameters
    ----------
    X : array, shape (n_features, n_vertices)
        Data matrix.
    covar : array, shape (n_features,)
        Covariate to control for.
    
    Returns
    -------
    R : array, shape (n_vertices, n_vertices)
        Partial correlation matrix.
    """

    n_features, n_vertices = X.shape
    # z-score each vertex across profiles
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    # z-score covariate
    covar_z = (covar - covar.mean()) / covar.std(ddof=1)
    # design matrix: intercept + covariate 
    # shape (n_profiles, 2)
    X_covar = np.column_stack([np.ones(n_features), covar_z])
    # regression for all vertices at once 
    # beta shape (2, n_vertices)
    beta, _, _, _ = np.linalg.lstsq(X_covar, X_z, rcond=None)
    # predicted values and residuals
    # shape (n_profiles, n_vertices)
    X_hat = X_covar @ beta   
    # same shape as X            
    residuals = X_z - X_hat              
    # correlation matrix of residuals across vertices
    # shape (n_vertices, n_vertices)
    R = np.corrcoef(residuals, rowvar=False) 
    # Fisher z-transform
    with np.errstate(divide='ignore', invalid='ignore'):
        MPC = np.arctanh(R)
        MPC[np.isnan(MPC)] = 0
        MPC[np.isinf(MPC)] = 0 
    return MPC