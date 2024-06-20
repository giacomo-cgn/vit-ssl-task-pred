import json
import copy
import os

from src.utils import read_command_line_args
from search_hyperparams import search_hyperparams
from main import exec_experiment


# Read args from command line
original_args = read_command_line_args()

print("READING CONFIG ...")


# Read config.json
with open('config.json') as f:
    config = json.load(f)

    for k, v in config["common_params"].items():
        original_args.__setattr__(k, v)
        # Add also variant with param name with "-" substituted with "_" and vice versa
        original_args.__setattr__(k.replace("_", "-"), v)
        original_args.__setattr__(k.replace("-", "_"), v)

    

    if config["experiments"] is not None:
        # Execute hyperparameters search
        for experiment in config["experiments"]:
            if "name" in experiment:
                name = experiment["name"]
            else:
                name = ''

            print(f"Running experiment: {experiment}")
            args = copy.deepcopy(original_args)

            # Apply experiment specific params
            for k, v in experiment.items():
                if k not in experiment["hyperparams_search"] and k != "name" and k != "hyperparams_search":
                    args.__setattr__ (k, v)
                    # Add also variant with param name with "-" substituted with "_" and vice versa
                    args.__setattr__(k.replace("_", "-"), v)
                    args.__setattr__(k.replace("-", "_"), v)

            print("experiment.hyperparams:", experiment["hyperparams_search"])
            
            # Run hyperparam search
            log_dir = os.path.join(os.path.join(original_args.save_folder, f'{original_args.dataset}'),
                            f'mbtr{original_args.tr_mb_size}_mbrep{original_args.repl_mb_size}_k{original_args.mb_passes}')
            search_hyperparams(args, hyperparams_dict=experiment["hyperparams_search"], parent_log_folder=log_dir, experiment_name=name)
    else:
        # Hyperparams search not needed
        log_dir = os.path.join(os.path.join(original_args.save_folder, f'{original_args.dataset}'),
                            f'mbtr{original_args.tr_mb_size}_mbrep{original_args.repl_mb_size}_k{original_args.mb_passes}')
        original_args.__setattr__('save-folder', log_dir)
        original_args.__setattr__('save_folder', log_dir)
        exec_experiment(**original_args.__dict__)