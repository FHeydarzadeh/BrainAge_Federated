import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import flwr as fl
from flwr.server.strategy import FedAvg, QFedAvg
from flwr.server import start_server
import os
import datetime
import torch
import torch.nn as nn
from collections import OrderedDict
import pickle
import warnings
warnings.filterwarnings("ignore")

from utils.model import Net, test, load_model
from utils.data import get_centralized_dataloader
from utils.strategy import CustomizedQFedAvg




def get_on_fit_config(config: DictConfig):
    """
    Return function that prepares config to send to clients.
    """

    def fit_config_fn(server_round: int):
        # This function will be executed by the strategy in its
        # `configure_fit()` method.

        return {
            'lr': config.lr,
            'num_epochs': config.num_epochs,
            'patience': config.patience,
            'server_round': server_round
        }
    
    print('\n### fit_config was extracted ###')

    return fit_config_fn



def get_on_evaluate_config_fn(config: DictConfig):
    """
    Return function that prepare evaluation config
    """
    def evaluate_config_fn(server_round: int):
        return {
            'server_round': server_round,
            'num_rounds': config.num_rounds
        }
    
    print('\n### evaluate_config was extracted ###')

    return evaluate_config_fn



def get_evaluate_fn(config: DictConfig):
    """
    Define function for global evaluation on the centralized dataset located on the server.
    """
    
    def evaluate_fn(server_round: int, parameters, cfg):
        # This function is called by the strategy's `evaluate()` method
        # and receives as input arguments the current round number and the
        # parameters of the global model.
        # this function takes these parameters and evaluates the global model
        # on a evaluation / test dataset.

        cfg = config
        num_rounds = cfg.num_rounds
        model = Net
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size = cfg.batch_size  
        eval_criterion = nn.L1Loss(reduction='sum')
        centralized_val_dataset_path = './data/centralized_val_dataset.csv'
        centralized_test_dataset_path = './data/centralized_test_dataset.csv'

        centralized_val_loader = get_centralized_dataloader(centralized_val_dataset_path, batch_size)
        centralized_test_loader = get_centralized_dataloader(centralized_test_dataset_path, batch_size)

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.to(device)


        if server_round != num_rounds:
            loss, corr, mae, count, sub_ids, true_ages, pred_ages = test(model, centralized_val_loader, eval_criterion) 

        else:
            loss, corr, mae, count, sub_ids, true_ages, pred_ages = test(model, centralized_test_loader, eval_criterion) 

        # Report the global (centralized) loss and any other metric (inside a dictionary).
        return loss, {'corr': corr, 'mae': mae, 'count': count, 'sub_id': sub_ids, 'true_age': true_ages, 'pred_age': pred_ages}

    return evaluate_fn





os.environ['HYDRA_FULL_ERROR'] = '1'

# Hydra is a yaml based configuration system.
# It can parse all the configurations (which are required as our FL settings) from base.yaml
# A decorator for Hydra. This tells hydra to by default load the config in conf/base.yaml
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    print('Configuration', OmegaConf.to_yaml(cfg))

    # Hydra automatically creates a directory (<this directory>/outputs/<date>/<time>) for your experiment.
    # Retrieve the path to the outputs to save the results.
    save_path = HydraConfig.get().runtime.output_dir

    num_clients = 3
    num_rounds = cfg.num_rounds
    wood_model_path = './models/wood/wood_T1.pt'
    initial_global_model = load_model(wood_model_path)


    fedavg_strategy = FedAvg(
        min_fit_clients = num_clients,
        min_evaluate_clients =num_clients,
        min_available_clients = num_clients,
        evaluate_fn = get_evaluate_fn(cfg),
        on_fit_config_fn = get_on_fit_config(cfg.config_fit),
        on_evaluate_config_fn = get_on_evaluate_config_fn(cfg),
        initial_parameters = initial_global_model
    )


    # when q=0 it acts as FedAvg, so we only use this as the strategy and set q in the command line.
    qfedavg_strategy = CustomizedQFedAvg(
        q_param = cfg.q_param,
        min_fit_clients = num_clients,
        min_evaluate_clients =num_clients,
        min_available_clients = num_clients,
        evaluate_fn = get_evaluate_fn(cfg),
        on_fit_config_fn = get_on_fit_config(cfg.config_fit),
        on_evaluate_config_fn =get_on_evaluate_config_fn(cfg),
        initial_parameters = initial_global_model
    )

    

    # Record the start time
    start_time = datetime.datetime.now()
    print(f"\n### Process started at: {start_time} ###\n")

    print('\n### Extracting server rounds... ###')
    server_config=fl.server.ServerConfig(num_rounds=cfg.num_rounds)

    print('\n### Starting server... ###')
    
    
    history = start_server(
        server_address="0.0.0.0:8080",
        config=server_config,
        strategy=fedavg_strategy,       # Adapt the strategy based on your need
        )


    project = cfg.project
    results_dir = './results/centralized/'
    results_path = results_dir + project + '_centralized_results.pkl'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    results = {"history": history, 'losses_centralized': history.losses_centralized, 'losses_distributed': history.losses_distributed, 
               'metrics_centralized': history.metrics_centralized, 'metrics_distributed': history.metrics_distributed}
    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    # Record the end time
    end_time = datetime.datetime.now()
    print(f"\n### Process ended at: {end_time} ###\n")
    duration = end_time - start_time
    print(f"\n### Total processing time: {duration} ###\n")



if __name__ == "__main__":

    main()

    
   