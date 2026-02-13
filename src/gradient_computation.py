import numpy as np
from brainspace.gradient.gradient import GradientMaps


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


def compute_t1_gradient(df_yeo_surf, t1_salience_profiles, network='SalVentAttn'):
    t1_salience_mpc = [partial_corr_with_covariate(subj_data, covar=t1_mean_profile) for subj_data, t1_mean_profile in zip(t1_salience_profiles[:, :, :], np.nanmean(t1_salience_profiles, axis=2))]
    gm_t1 = GradientMaps(n_components=10, random_state=None, approach='dm', kernel='normalized_angle', alignment='procrustes')
    gm_t1.fit(t1_salience_mpc, sparsity=0.9)
    t1_gradients = np.mean(np.asarray(gm_t1.aligned_), axis=0)
    print("gradient lambdas: {}".format(np.mean(np.asarray(gm_t1.lambdas_), axis=0)))
    df_yeo_surf.loc[df_yeo_surf['network'].eq(network), 't1_gradient1_' + network] = t1_gradients[:, 0]
    return df_yeo_surf