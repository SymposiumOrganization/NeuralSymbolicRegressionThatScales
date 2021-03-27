from .dclasses import Equation

def evaluate_model(model_predict,
                   benchmark_name,
                   equation_idx,
                   num_test_points,
                   pointwise_acc_rtol,
                   pointwise_acc_atol):
    """
    model_predict is a callable that takes an X of shape
    (n_datapoints, n_variables) and returns scalar predictions
    """

    gt_equation, num_variables, supp = load_equation(benchmark_name,
                                                     equation_idx)
    print(f'gt_equation: {gt_equation}')

    metrics = {'gt_equation': gt_equation,
               'num_variables': num_variables,}
    for iid_ood_mode in ['iid', 'ood']:
        X, y = get_data(gt_equation, num_variables, supp, num_test_points,
                        iid_ood_mode=iid_ood_mode)

        y_pred = model_predict(X)
        assert y_pred.shape == (X.shape[0],)

        if np.iscomplex(y_pred).any():
            warnings.warn('Complex values found!')
            y_pred = np.real(y_pred)

        pointwise_acc = get_pointwise_acc(y, y_pred,
                                          rtol=pointwise_acc_rtol,
                                          atol=pointwise_acc_atol)
        acc_key = _get_acc_key(iid_ood_mode)
        metrics[acc_key] = pointwise_acc

        # Drop all indices where the ground truth is NaN or +-inf
        assert y.ndim == 1
        valid_idxs = ~np.isnan(y) & ~np.isinf(y)
        metrics[f'frac_valid_{iid_ood_mode}'] = valid_idxs.sum() / num_test_points
        y = y[valid_idxs]
        y_pred = y_pred[valid_idxs]
        assert y.shape[0] == valid_idxs.sum()

        # Replace NaN or infinity prediction
        replace_prediction_idxs = np.isnan(y_pred) | np.isinf(y_pred)
        metrics[f'frac_replace_{iid_ood_mode}'] = replace_prediction_idxs.sum() / y.shape[0]
        y_pred[replace_prediction_idxs] = 0.0

        # Add default metrics
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        metrics[f'r2_{iid_ood_mode}'] = r2
        metrics[f'mse_{iid_ood_mode}'] = mse

    return metrics

def get_data_reject_nan(eq: Equation, cfg):
    n_attempts_max = 100
    X, y = get_data(gt_equation, num_variables, supp, num_eval_points, iid_ood_mode)
    for _ in range(n_attempts_max):
        nans = np.isnan(y)
        if not nans.any():
            break

        n_nans = nans.sum()
        X[nans], y[nans] = get_data(gt_equation, num_variables, supp, n_nans,
                                    iid_ood_mode)
    if nans.any():
        raise ValueError('Could not sample valid points for equation '
                         f'{gt_equation} supp={supp}')
    return X, y

def get_data(eq: Equation, cfg):
    """
    iid_ood_mode: if set to "iid", sample uniformly from the support as given
                  by supp; if set to "ood", sample from a larger support

    """
    sym = []
    vars_list = []
    for i, var in enumerate(["x", "y", "z"]):
        if i < num_variables:
            # U means random uniform sampling.
            # Currently this is the only mode we support.
            # Decide what to do about the linspace mode.
            assert 'U' in supp[var], (f'Support modes {supp[var].keys()} not '
                                      f'implemented! Decide what to do about '
                                      f'those.')
            l, h = supp[var]['U']
            if iid_ood_mode == 'iid':
                x = np.random.uniform(l, h, int(num_eval_points))
            elif iid_ood_mode == 'ood':
                support_length = h - l
                assert support_length > 0
                x = np.random.uniform(l-support_length, h+support_length,
                                      int(num_eval_points))
            else:
                raise ValueError(f'Invalid iid_ood_mode: {iid_ood_mode}')
            sym.append(x)
            vars_list.append(var)

    X = np.column_stack(sym)
    assert X.ndim == 2
    assert X.shape[1] <= 3
    assert X.shape[0] == num_eval_points
    y = evaluate_func(gt_equation, vars_list, X)
    return X, y


def load_equation(eq: Equation, cfg):
    n_attempts_max = 100
    X, y = get_data(eq, cfg)
    for _ in range(n_attempts_max):
        nans = np.isnan(y)
        if not nans.any():
            break

        n_nans = nans.sum()
        X[nans], y[nans] = get_data(eq, n_nans,
                                    iid_ood_mode)
    if nans.any():
        raise ValueError('Could not sample valid points for equation '
                         f'{gt_equation} supp={supp}')
    return X, y
