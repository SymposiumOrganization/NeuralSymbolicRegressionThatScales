import hydra
from nesymres import benchmark

def load_data():
    df = pd.read_csv(EQUATIONS_PATH)
    df.set_index(['benchmark', 'num'], inplace=True)
    return df    


def load_equation(benchmark_name, equation_idx):
    df = load_data()
    benchmark_row = df.loc[(benchmark_name, equation_idx)]
    gt_equation = benchmark_row['gt_expr']
    supp = ast.literal_eval(benchmark_row['support'])
    num_variables = len(supp)
    return gt_equation, num_variables, supp



def get_model(cfg):
    if cfg.model_name == 'brenden':
        """Brenden is not compatible with latest python version"""
        from models.brenden import get_brenden  
        return get_brenden(args.brenden_n_epochs)
        
    elif cfg.model_name == 'nesymres':
        from models.nesymres import get_nesymres  # Only import if needed
        return get_nesymres(args.nesymres_checkpoint_path,
                            args.nesymres_beam_size,
                            args.nesymres_bfgs,
                            args.nesymres_bfgs_n_restarts,
                            args.nesymres_complexity_reg_coef)
    elif args.model_name == 'genetic_prog':
        from models.genetic_prog import get_genetic_prog
        return get_genetic_prog(args.genetic_prog_population_size)
    elif args.model_name == 'gaussian_proc':
        from models.gaussian_proc import get_gaussian_proc
        return get_gaussian_proc(args.gaussian_proc_n_restarts)
    else:
        raise ValueError(f'Unknown model_name: {args.model_name}')


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


@hydra.main("evaluate")
def main(cfg):
    targe_path = hydra.utils.to_absolute_path(Path("results")/cfg.name)
    model = get_model(cfg)
    eq = load_equation(cfg.benchmark_name,idx)
    gt_equation, num_variables, supp = get_robust_data(eq, cfg)
    X_train, y_train = get_data_reject_nan(
        gt_equation,
        num_variables,
        supp,
        args.num_eval_points,
        iid_ood_mode='iid')
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    duration = time.perf_counter() - start_time
    if hasattr(model, 'get_equation'):
        equation = model.get_equation()
        model_path = None
    else:
        equation = None
        model_path = str(targe_path / 'model.pkl')
        try:
            delattr(model, '_programs')
        except AttributeError:
            pass

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        output_data = {
            'duration': duration,
            'equation': equation,
            'model_path': model_path,
            'platform_node': get_platform_node(),
        }
        if hasattr(model, 'metrics') and isinstance(model.metrics, dict):
            output_data.update(model.metrics)
        with open(Path(cfg.name) / 'results.json', 'w') as f:
            json.dump(output_data, f, indent=2)

if __name__=="__main__":
    main()