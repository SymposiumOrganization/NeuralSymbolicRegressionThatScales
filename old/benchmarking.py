import click
from nesymres.utils import load_data
import hydra
from nesymres.benchmark import load_equation

@hydra.main(config_name="benchmarking")
def main(cfg):
    benchmark = load_data(cfg.path)
    for eq in benchmarks.eqs:
        
        gt_equation, num_variables, supp = load_equation(args.benchmark_name,
                                                            args.equation_idx)
        X_train, y_train = get_data_reject_nan(
            gt_equation,
            num_variables,
            supp,
            args.num_eval_points,
            iid_ood_mode='iid')
        model = get_model(args)
        model.fit(X_train, y_train)
        equation = model.get_equation()

if __name__=="__main__":
    main()