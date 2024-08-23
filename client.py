import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import flwr as fl
from flwr.common import NDArrays, Scalar, ndarrays_to_parameters
from flwr.client import start_client
from typing import Dict
from collections import OrderedDict
import datetime
import warnings
warnings.filterwarnings("ignore")

from utils.utils import initialize_path, get_k_last_models, load_config
from utils.data import get_dataloaders
from utils.model import Net, train, test, save_train_result, save_val_result, save_test_result, average_model_params



class FlowerClient(fl.client.NumPyClient):

    def __init__(self, client_name, project_name, k_folds, train_loaders, val_loaders, test_loader) -> None:
        super().__init__()

        self.client = client_name
        self.project = project_name
        self.project_dir, self.plot_dir, self.model_dir = initialize_path(self.client, self.project)
        self.kf = k_folds
        self.trainloaders = train_loaders
        self.valloaders = val_loaders
        self.testloader = test_loader
        self.model = Net
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        #torch.device("cpu")


    
    def get_parameters(self, config: Dict[str, Scalar], model=None):
        """
        Extract model parameters and return them as a list of numpy arrays.
        """
        if model is None:
            model = self.model
        return [val.cpu().numpy() for _, val in model.state_dict().items()]
    


    def set_parameters(self, parameters):
        """
        Receive parameters and apply them to the local model.
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    
    # This fit method averages the models across k folds and sends the average model.
    # Doesn't work well because the loss is decreasing from fold to fold but on average the loss is not decreasing from one training round to the next.
    # It seems by averaging the model, every round is starting from the beginning!
    # Solution: using next fit method for a new experiment.
    # With "average model" across k folds. Should be used with average model evaluate function.
    # def fit(self, parameters, config):
    #     """
    #     Train model with the parameters received by the server using the data
    #     that belongs to this client. Then, send it back to the server.
    #     """

    #     # copy parameters sent by the server into client's local model.
    #     print('\n### Setting model parameters for training... ###')
    #     self.set_parameters(parameters)
    #     self.model.to(self.device)


    #     # fetch elements in the config sent by the server.
    #     print('\n### Fetching client config for fit... ###')
    #     lr = config["lr"]
    #     num_epochs = config["num_epochs"]
    #     patience = config["patience"]
    #     server_round = config['server_round']

        
    #     optimizer = optim.Adam(self.model.parameters(), lr=lr)
    #     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)
    #     criterion = nn.L1Loss()
    #     eval_criterion = nn.L1Loss(reduction='sum')
    #     model_save_path = self.model_dir + datetime.datetime.now().strftime('%d-%m-%y-%H_%M.pt')

    #     # do local training on k folds
    #     print('\n### Start training... ###')
    #     avg_train_loss = 0          
    #     avg_val_loss = 0            
    #     avg_train_count = 0
    #     avg_val_count = 0

    #     for k in range(self.kf):
    #         trainloader = self.trainloaders[k]
    #         valloader = self.valloaders[k]
    #         train_loss, train_count, val_loss, val_count, _ = train(self.model, optimizer, scheduler, trainloader, valloader, criterion, eval_criterion, model_save_path, num_epochs, patience)
            
    #         avg_train_loss += train_loss / self.kf
    #         avg_val_loss += val_loss / self.kf
    #         avg_train_count += train_count / self.kf
    #         avg_val_count += val_count / self.kf

    #     avg_train_count = int(avg_train_count)
    #     avg_val_count = int(avg_val_count)
        
    #     print('\n### Save train results... ###')
    #     save_train_result(self.project, self.project_dir, server_round, avg_train_loss, avg_train_count, avg_val_loss, avg_val_count)

    #     print('\n### Averaging k last models... ###')
    #     k_last_models = get_k_last_models(self.model_dir, self.kf)
    #     avg_model = average_model_params(k_last_models) 
    #     parameters = self.get_parameters(avg_model)        

    #     # Flower clients need to return three arguments: the updated model, the number
    #     # of examples in the client (although this depends a bit on your choice of aggregation
    #     # strategy), and a dictionary of metrics (here you can add any additional data, but these
    #     # are ideally small data structures)
    #     print('\n### Returning parameters... ###')
    #     return parameters, avg_train_count, {'avg_train_loss': avg_train_loss, 'avg_val_loss': avg_val_loss} 
    # 



    # With "best model" across k folds. Should be used with best model evaluate function.
    # This saves the best model across k folds and sends its parameter to the server.
    def fit(self, best_parameters, config):
        """
        Train model with the parameters received by the server using the data
        that belongs to this client. Then, send it back to the server.
        """

        # copy parameters sent by the server into client's local model.
        print('\n### Setting model parameters for training... ###')
        self.set_parameters(best_parameters)
        self.model.to(self.device)


        # fetch elements in the config sent by the server.
        print('\n### Fetching client config for fit... ###')
        lr = config["lr"]
        num_epochs = config["num_epochs"]
        patience = config["patience"]
        server_round = config['server_round']

        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)
        criterion = nn.L1Loss()
        eval_criterion = nn.L1Loss(reduction='sum')
        model_save_path = self.model_dir + datetime.datetime.now().strftime('%d-%m-%y-%H_%M.pt')

        # do local training on k folds
        print('\n### Start training... ###')
        best_loss = 1e9
        best_train_loss = None
        best_val_loss = None
        best_train_count = 0
        best_val_count = 0

        for k in range(self.kf):
            trainloader = self.trainloaders[k]
            valloader = self.valloaders[k]
            train_loss, train_count, val_loss, val_count, _ = train(self.model, optimizer, scheduler, trainloader, valloader, criterion, eval_criterion, model_save_path, num_epochs, patience)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_train_loss = train_loss
                best_val_loss = val_loss
                best_train_count = train_count
                best_val_count = val_count
                torch.save(self.model.state_dict(), model_save_path)
                

        
        print('\n### Save train results... ###')
        save_train_result(self.project, self.project_dir, server_round, best_train_loss, best_train_count, best_val_loss, best_val_count)

        print('\n### Averaging k last models... ###')
        best_model = get_k_last_models(self.model_dir, 1)
        best_parameters = self.get_parameters(best_model)        

        # Flower clients need to return three arguments: the updated model, the number
        # of examples in the client (although this depends a bit on your choice of aggregation
        # strategy), and a dictionary of metrics (here you can add any additional data, but these
        # are ideally small data structures)
        print('\n### Returning parameters... ###')
        return best_parameters, best_train_count, {'loss': best_train_loss, 'val_loss': best_val_loss}   
    
    

    # # With average model across k folds. Should be used with average model fit function.
    # def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
    #     """
    #     Evaluate final global model on the test set.
    #     """

    #     # fetch elements in the config sent by the server.
    #     print('\n### Fetching client config for evaluate... ###')
    #     num_rounds = config['num_rounds']
    #     server_round = config['server_round']

    #     loss = 0.0
    #     corr = 0.0
    #     mae = 0.0
    #     count = 0

    #     eval_criterion = nn.L1Loss(reduction='sum')

    #     if server_round == num_rounds:
    #         print('\n### Setting model parameters for test evaluation... ###')
    #         self.set_parameters(parameters)

    #         print('\n### Starting test evaluation... ###')
    #         loss, corr, mae, count, sub_ids, true_ages, pred_ages = test(self.model, self.testloader, eval_criterion)
            
    #         print('\n### Saving test results... ###')
    #         save_test_result(self.project, self.project_dir, server_round, loss, corr, mae, count, sub_ids, true_ages, pred_ages)
            
            
    #     else:
    #         print('\n### Starting val evaluation... ###')
    #         for k in range(self.kf):
    #             valloader = self.valloaders[k]
    #             k_loss, k_corr, mae, k_count, sub_ids, true_ages, pred_ages = test(self.model, valloader, eval_criterion)
    #             loss += k_loss / self.kf
    #             count += k_count / self.kf
    #             # corr += k_corr / self.kf
    #         count = int(count)
    #         print('\n### Saving val results... ###')
    #         save_val_result(self.project, self.project_dir, server_round, loss, corr, mae, count, sub_ids, true_ages, pred_ages)
            

    #     return float(loss), count, {}

    

    # With best model across k folds. Should be used with best model fit function.
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """
        Evaluate final global model on the test set.
        """

        # fetch elements in the config sent by the server.
        print('\n### Fetching client config for evaluate... ###')
        num_rounds = config['num_rounds']
        server_round = config['server_round']

        loss = 0.0
        corr = 0.0
        mae = 0.0
        count = 0

        eval_criterion = nn.L1Loss(reduction='sum')

        if server_round == num_rounds:
            print('\n### Setting model parameters for test evaluation... ###')
            self.set_parameters(parameters)

            print('\n### Starting test evaluation... ###')
            loss, corr, mae, count, sub_ids, true_ages, pred_ages = test(self.model, self.testloader, eval_criterion)
            
            print('\n### Saving test results... ###')
            save_test_result(self.project, self.project_dir, server_round, loss, corr, mae, count, sub_ids, true_ages, pred_ages)
            
            
        else:
            print('\n### Starting val evaluation... ###')
            best_loss = 1e9
            loss = None
            count = 0
            corr =0.0
            mae = 0.0
            sub_ids = []
            true_ages = []
            pred_ages = []

            for k in range(self.kf):
                valloader = self.valloaders[k]
                k_loss, k_corr, k_mae, k_count, k_sub_ids, k_true_ages, k_pred_ages = test(self.model, valloader, eval_criterion)

                if k_loss < best_loss:
                    best_loss = k_loss
                    loss = k_loss
                    count = k_count
                    corr = k_corr
                    mae = k_mae
                    sub_ids = k_sub_ids
                    true_ages = k_true_ages
                    pred_ages = k_pred_ages

           
            print('\n### Saving val results... ###')
            save_val_result(self.project, self.project_dir, server_round, loss, corr, mae, count, sub_ids, true_ages, pred_ages)
            
        return float(loss), count, {}
    


def client_fn(config, client_name, project_name, csv_file_path):
    """
    Create and return an instance of client.
    """

    print('\n### Fetching client config for client function... ###')
    batch_size = config['batch_size']
    kf = config['kf']

    print('\n### Getting dataloaders... ###')
    train_loaders, val_loaders, test_loader = get_dataloaders(csv_file_path, batch_size, random_seed=10, k_folds=kf)

    return FlowerClient(client_name, project_name, kf, train_loaders, val_loaders, test_loader).to_client()



def main(args):

    client = args.client
    project = args.project


    csv_file = './data/' + client + '_dataset.csv'
    config_file = load_config('./conf/base.yaml')   # These configs are static configs and read directly from the base.yaml file (not sent by the server)

    # Prepare client function with partial with parsed arguments
    print('\n### Configuring client... ###')
    configuered_client_fn = client_fn(config=config_file, client_name=client, project_name=project, csv_file_path=csv_file)

    # Flower ClientApp: Start the client app with the configured client function
    print('\n### Starting client... ###')
    start_client(
        server_address="127.0.0.1:8080",        # If you run it on the same machine for the client and server: "127.0.0.1:8080" for client.py
        client=configuered_client_fn            # If you run it on the virtual network:'192.168.100.3:8080' 
    )
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Start the Flower client.')
    parser.add_argument('--client', type=str, required=True, help='client name')
    parser.add_argument('--project', type=str, required=True, help='project name')


    args = parser.parse_args()
    main(args)