"""
utils script:
This script provides functions to create or retrieve path and directories.
==> Inspired by the GitHub repository of Alvaro Javier Vargas Guerrero 2024 (https://github.com/AIMS-VUB/BrainAgeFederated.git)
"""

import os
import yaml
import warnings
warnings.filterwarnings("ignore")



def generate_default_dir():
    default_dir = './results/'
    if not os.path.exists(default_dir):
      os.makedirs(default_dir)
    return default_dir



def generate_client_dir(client):
    client_dir = generate_default_dir() + client + '/'
    if not os.path.exists(client_dir):
      os.makedirs(client_dir)
    return client_dir



def generate_project_dir(client, project):
    project_dir = generate_client_dir(client) + project + '/'
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    return project_dir



def generate_plot_dir(project_dir):
    plot_dir = generate_default_dir() + 'plots/'
    if not os.path.exists(plot_dir):
      os.makedirs(plot_dir)
    return plot_dir



def generate_model_dir(project_dir):
    model_dir = project_dir + 'models/'
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
    return model_dir



def initialize_path(client, project):
    project_dir = generate_project_dir(client, project)
    plot_dir = generate_plot_dir(project_dir)
    model_dir = generate_model_dir(project_dir)
    return project_dir, plot_dir, model_dir




def get_k_last_models(model_dir, k):
    """
    Function for retreiving the path to the k most recent pt file in a folder.
    returns: List[str]
    """
    
    pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]                        # List all the pt files present in the folder
    pt_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)   # Sort them by modification date
    k_last_models = [os.path.join(model_dir, f) for f in pt_files[:k]]                        # Get the path of the k last models
    return k_last_models



# This is not used to load config_fit, only for others.
def load_config(yaml_file_path):
    """
    Load configurations in yaml file when initialize the client.
    """
    with open(yaml_file_path, 'r') as f:
        return yaml.safe_load(f)


