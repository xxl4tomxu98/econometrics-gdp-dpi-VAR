from pybats.latent_factor import latent_factor, dlm_coef_fxn, dlm_coef_forecast_fxn
## Latent factor functions for linear predictor lambda
def lambda_fxn(date, mod, k, **kwargs):
    """
    function that returns mean and variance of linear predictor, lambda
    :param date: date index
    :param mod: model that is being run
    :param k: forecast horizon
    :param kwargs: other arguments
    :return: mean and variance of lambda
    """
    return (mod.F.T @ mod.m).copy().reshape(-1), (mod.F.T @ mod.C @ mod.F).copy()


def lambda_forecast_fxn(date, mod, k, forecast_path = False, **kwargs):
    """
    functions that return forecast mean and variance, potentially covariance of lambda
    (if forecast_path is True)
    :param date: date index
    :param mod: model that is running
    :param k: forecast horizon
    :param forecast_path: True or False
    :param kwargs: other arguments
    :return: forecast mean and variance, potentially covariance, of lambda (if forecast_path is True)
    """
    lambda_mean = []
    lambda_var = []

    if forecast_path:
        lambda_cov = [np.zeros([1, h]) for h in range(1, k)]
    for j in range(1, k + 1):
        f, q = mod.get_mean_and_var(mod.F, mod.a.reshape(-1), mod.R)
        lambda_mean.append(f.copy())
        lambda_var.append(q.copy())

    if forecast_path:
        if j > 1:
            for i in range(1, j):
                lambda_cov[j-2][i-1] = mod.F.T @ forecast_R_cov(mod, i, j) @ mod.F

    if forecast_path:
        return lambda_mean, lambda_var, lambda_cov
    else:
        return lambda_mean, lambda_var

lambda_lf = latent_factor(gen_fxn = lambda_fxn, gen_forecast_fxn = lambda_forecast_fxn)


## Latent factor functions for scaled model coefficients
def dlm_coef_scale_fxn(date, mod, scale = None, idx = None,
                       scale_which = None, **kwargs):
    """
    function that gets the mean and variance of coefficent latent factor
    :param date: date index
    :param mod: model that is being run
    :param scale: scalars that used to scale the mean and variance, as known 
     fixed values. For example, covariates of models that use this latent factor. 
     Should be in pandas dataframe with scalars in columns and dates as index
    :param scale_which: index of coefficents to be scaled by series in scale 
     (need to be within idx)
    :param idx: index of coefficents desired to extract
    :param kwargs: other arguments
    :return: mean and variance of scaled coefficents
    """
    if scale is None:
        return dlm_coef_fxn(date, mod, idx, **kwargs)
    if idx is None:
        idx = np.arange(0, len(mod.m))
    if not set(scale_which).issubset(set(idx)):
        ValueError("scale_which needs to be subset of idx")
    m_scale, C_scale = mod.m.copy(), mod.C.copy()
    scale_matrix = np.identity(C_scale.shape[0])
    scale_matrix[np.ix_(scale_which, scale_which)] = scale.loc[date].values * \
                                                     scale_matrix[np.ix_(scale_which, scale_which)]
    m_scale = scale_matrix@m_scale
    C_scale = scale_matrix@C_scale@scale_matrix
    return (m_scale[idx]).reshape(-1), (C_scale[np.ix_(idx, idx)]).copy()


def dlm_coef_scale_forecast_fxn(date, mod, k, scale = None, idx=None, scale_which = None, \
                                forecast_path=False, **kwargs):
    """
    function that compute the forecast mean, variance and, potentially covariance
    (if forecast_path is True)
    :param date: date index
    :param mod: model that is being run
    :param k: forecast horizon
    :param scale: scalars that used to scale the mean and variance, as known fixed values.
     For example, covariates of models that use this latent factor. Should be in pandas data
     frame with scalars in columns and dates as index
    :param scale_which: index of coefficents to be scaled by series in scale 
     (need to be within idx)
    :param idx: index of coefficents desired to extract
    :param forecast_path: True or False
    :param kwargs: other arguments
    :return: forecast mean, variance and potentially covariance (if forecast_path is True)
    """
    if scale is None:
        return dlm_coef_forecast_fxn(date, mod, k, idx=None, forecast_path=False, **kwargs)
    if idx is None:
        idx = np.arange(0, len(mod.m))
        p = len(idx)
    if not set(scale_which).issubset(set(idx)):
        ValueError("scale_which needs to be subset of idx")
    dlm_coef_mean = []
    dlm_coef_var = []
    if forecast_path:
        dlm_coef_cov = [np.zeros([p, p, h]) for h in range(1, k)]
    for j in range(1, k + 1):
        a, R = forecast_aR(mod, j)
    a_scale = a.copy()
    R_scale = R.copy()
    scale_matrix = np.identity(R_scale.shape[0])
    scale_matrix[np.ix_(scale_which, scale_which)] = scale.loc[date].values*scale_matrix[np.ix_(scale_which,scale_which)]
    a_scale = scale_matrix@a_scale
    R_scale = scale_matrix@R_scale@scale_matrix
    dlm_coef_mean.append(a_scale[idx].copy().reshape(-1))
    dlm_coef_var.append(R_scale[np.ix_(idx, idx)].copy())
    if forecast_path:
        if j > 1:
            for i in range(1, j):
                R_cov_scale = forecast_aR(mod, i)[1]
        R_cov_scale = scale_matrix@R_cov_scale@scale_matrix
        Gk = np.linalg.matrix_power(mod.G, j - i)
    dlm_coef_cov[j-2][:,:,i-1] = (Gk@R_cov_scale)[np.ix_(idx, idx)]
    
    if forecast_path:
        return dlm_coef_mean, dlm_coef_var, dlm_coef_cov
    else:
        return dlm_coef_mean, dlm_coef_var


dlm_coef_scale_lf = latent_factor(gen_fxn = dlm_coef_fxn, \
                                  gen_forecast_fxn=dlm_coef_forecast_fxn)

