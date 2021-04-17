import argparse
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
import hydra

#from misc_utils.misc_utils import get_combined_df, get_root_dirs, reroute_path
#from utils import get_variables, evaluate_func

POINTWISE_ACC_RTOL = 0.05
POINTWISE_ACC_ATOL = 0.001
NUM_TEST_POINTS = 10_000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dirs', type=str, required=True, nargs='+')

    args = parser.parse_args()
    return args


def rename_benchmark(benchmark_name):
    """We updated our benchmark name from "ours" to "ours-nc"."""
    if benchmark_name == 'ours':
        return 'ours-nc'
    return benchmark_name


def standardize_equation(equation):
    """
    Replace x1 -> x, x2 -> y, x3 -> z in order to work with equivalent formats.
    Also, add 0 times all variables in order to make all variables appear always
    """
    if equation is None or (not isinstance(equation, str) and np.isnan(equation)):
        return None
    equation = equation.replace('x1', 'x')
    equation = equation.replace('x2', 'y')
    equation = equation.replace('x3', 'z')
    return equation


def patch_benchmark_name(args_df):
    """For experiments before 2021-02-03, there was no key 'benchmark_name'.
    They were all using the AI Feynman dataset
    """
    output_dirs = args_df.output_dir
    output_dir = output_dirs[0]
    date = Path(output_dir).parent.name.split('_')[0]
    print(f'date: {date}')
    if datetime.strptime(date, '%Y-%m-%d') < datetime(2021, 2, 3):
        print('Patching benchmark_name!')
        args_df['benchmark_name'] = 'ai_feymann'

@hydra.main(config_name="fit")
def main(cfg):
    return collect_results(cfg.name)


def collect_results(root_dirs):
    # Get all args and equations and other results into a dataframe
    # root_dirs = get_root_dirs(root_dir=None, root_dirs=root_dirs)
    # combined_df = get_combined_df(root_dirs,
    #                               args_df_modifiers=[patch_benchmark_name],
    #                               run_dir_pattern='run*',
    #                               args_filename='args.json',
    #                               metrics_filename='results.json')

    # Manual fixes of the dataframe
    combined_df['equation'] = combined_df['equation'].map(standardize_equation)
    combined_df['benchmark_name'] = combined_df['benchmark_name'].map(rename_benchmark)

    print(f'combined_df: {combined_df}')
    print('columns')
    pprint(list(combined_df.columns))

    eval_rows = []

    for idx, df_row in tqdm(combined_df.iterrows(), desc='Evaluate equations...'):
        # print(f'df_row.equation_idx: {df_row.equation_idx}')
        # print(f'model_name: {df_row.model_name}')
        # print(f'equation: {df_row.equation}')
        # print(f'get_variables(equation): {get_variables(df_row.equation)}')
        # print(f'num_test_points: {NUM_TEST_POINTS}')


        if df_row.equation is not None:
            assert getattr(df_row, 'model_path', None) is None
            metrics = evaluate_equation(
                pred_equation=df_row.equation,
                benchmark_name=df_row.benchmark_name,
                equation_idx=df_row.equation_idx,
                num_test_points=NUM_TEST_POINTS,
                pointwise_acc_rtol=POINTWISE_ACC_RTOL,
                pointwise_acc_atol=POINTWISE_ACC_ATOL)
        else:
            model_path = reroute_path(df_row.model_path, df_row.output_dir,
                root_dirs)
            metrics = evaluate_sklearn(
                model_path=model_path,
                benchmark_name=df_row.benchmark_name,
                equation_idx=df_row.equation_idx,
                num_test_points=NUM_TEST_POINTS,
                pointwise_acc_rtol=POINTWISE_ACC_RTOL,
                pointwise_acc_atol=POINTWISE_ACC_ATOL
            )
        eval_row = df_row.to_dict()
        eval_row.update(metrics)
        eval_rows.append(eval_row)

    eval_df = pd.DataFrame(eval_rows)

    eval_df.to_csv(Path(root_dirs[0]) / 'eval_df_v2.csv')

    return eval_df, root_dirs


def get_pointwise_acc(y_true, y_pred, rtol, atol):
    # r = roughly_equal(y_true, y_pred, rel_threshold)
    r = np.isclose(y_true, y_pred, rtol=rtol, atol=atol, equal_nan=True)
    return r.mean()


def evaluate_equation(pred_equation, benchmark_name, equation_idx,
                      num_test_points, pointwise_acc_rtol,
                      pointwise_acc_atol):
    """Evaluate equation `pred_equation` given as a string."""

    def model_predict(X):
        pred_variables = get_variables(pred_equation)
        # assert len(pred_variables) <= X.shape[1]
        return evaluate_func(pred_equation, pred_variables, X)

    return evaluate_model(model_predict, benchmark_name, equation_idx,
                          num_test_points,
                          pointwise_acc_rtol, pointwise_acc_atol)


def evaluate_sklearn(model_path, benchmark_name, equation_idx,
                      num_test_points, pointwise_acc_rtol,
                      pointwise_acc_atol):
    """Evaluate sklearn model at model_path."""

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    def model_predict(X):
        return model.predict(X)

    return evaluate_model(model_predict, benchmark_name, equation_idx,
                          num_test_points,
                          pointwise_acc_rtol, pointwise_acc_atol)


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

    gt_equation, num_variables, supp = benchmark.load_equation(benchmark_name,
                                                     equation_idx)
    print(f'gt_equation: {gt_equation}')

    metrics = {'gt_equation': gt_equation,
               'num_variables': num_variables,}
    for iid_ood_mode in ['iid', 'ood']:
        X, y = benchmark.get_data(gt_equation, num_variables, supp, num_test_points,
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


def _get_acc_key(iid_ood_mode):
    return (f'pointwise_acc_r{POINTWISE_ACC_RTOL:.2}_'
            f'a{POINTWISE_ACC_ATOL:.2}_{iid_ood_mode}')


def plot_durations(combined_df):
    ax = sns.stripplot('nesymres_beam_size', 'duration', data=combined_df, size=3)
    ax.set_yscale('log')
    ax.set_ylabel('duration (seconds)')
    sns.despine(ax=ax)
    ax.grid(axis='y')
    ax.grid(axis='y', which='minor', alpha=0.2)
    plt.show()


if __name__ == '__main__':
    breakpoint()
    eval_df, root_dirs = main()