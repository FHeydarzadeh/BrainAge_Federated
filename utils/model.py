"""
model script:
This script provides deep learning models and training, validation loops.
==> Inspired by the GitHub repository of Wood et al. 2022 (https://github.com/MIDIconsortium/BrainAge)
"""

from collections import OrderedDict
import torch
from monai.networks.nets import DenseNet201
import pandas as pd
import datetime
import time
import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error
import os
from typing import List
import flwr as fl
import warnings
warnings.filterwarnings("ignore")




Net = DenseNet201(3, 1, 1)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     #torch.device("cpu") 



def convert_state_dict(input_path):
  """
  Function to remove the keywork 'module' from pytorch state_dict (which occurs when model is trained using nn.DataParallel).
  """
  new_state_dict = OrderedDict()
  state_dict = torch.load(input_path, map_location='cpu')
  for k, v in state_dict.items():
    if 'module' in k:
      name = k[7:]  # remove `module.`
    else:
      name = k
    new_state_dict[name] = v
  return new_state_dict



def load_model(model_path=None) -> List:
    """
    Load model parameters from a PyTorch .pt file and convert them to a list of NumPy arrays.
    """

    net = DenseNet201(3, 1, 1)

    if model_path:
        state_dict = convert_state_dict(model_path)
        net.load_state_dict(state_dict, strict=True)

    net.to(DEVICE)
    weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
    parameters = fl.common.ndarrays_to_parameters(weights)
    return parameters



def average_model_params(model_paths):
    parameters = []
    for path in model_paths:
        model = torch.load(path)
        net = DenseNet201(3, 1, 1)
        net.load_state_dict(model)
        parameters.append([val.cpu().numpy() for _, val in net.state_dict().items()])
        avg_parameters = [sum(x) / len(x) for x in zip(*parameters)]
    return avg_parameters 




def save_train_result(project, project_dir, server_round, train_loss, train_count, val_loss, val_count):

    name = project + '_train_results.csv'
    full_path = os.path.join(project_dir, name) 
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({
        'server_round': [server_round],
        'train_loss': [train_loss],
        'train_count': [train_count],
        'val_loss': [val_loss],
        'val_count': [val_count],
        'time': [now]})

    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
        df = pd.concat([df, new_row], ignore_index=True)
    else: 
        df = new_row

    df.to_csv(full_path, index=False)
    print(f'\n### train results saved to {full_path} ###\n')



def save_val_result(project, project_dir, server_round, loss, corr, mae, data_count, sub_ids, true_ages, pred_ages):

    name = project + '_val_results.csv'
    full_path = os.path.join(project_dir, name) 
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({
        'server_round': server_round,
        'loss': loss,
        'corr': corr,
        'mae': mae,
        'count': data_count,
        'sub_id': sub_ids,
        'true_age': true_ages,
        'pred_age': pred_ages,
        'time': now})

    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
        df = pd.concat([df, new_row], ignore_index=True)
    else: 
        df = new_row

    df.to_csv(full_path, index=False)
    print(f'\n### val results saved to {full_path} ###\n')



def save_test_result(project, project_dir, server_round, loss, corr, mae, data_count, sub_ids, true_ages, pred_ages):
    name = project + '_test_results.csv'
    full_path = os.path.join(project_dir, name) 
    pd.DataFrame({
        'server_round': server_round,
        'loss': loss,
        'corr': corr,
        'mae': mae,
        'count': data_count,
        'sub_id': sub_ids,
        'true_age': true_ages,
        'pred_age': pred_ages
        }).to_csv(full_path)
    print(f'\n### test results saved to {full_path} ###\n')



def train(net, optimizer, scheduler, train_loader, valid_loader, criterion, eval_criterion, model_save_path, num_epochs, patience):
    best_loss = 1e9
    num_bad_epochs = 0
    print('**BEGINNING TRAINING***')
    for epoch in range(num_epochs):
        start = time.time()
        train_loss = 0 
        train_count = 0 
        net.train()

        if num_bad_epochs >= patience:
            return None
        for i, data in enumerate(tqdm.tqdm(train_loader)):
            im, age, _ = data
            im = im.to(device=DEVICE, dtype = torch.float)
            age = age.to(device=DEVICE, dtype=torch.float)
            age = age.reshape(-1,1)


            optimizer.zero_grad()
            pred_age = net(im)
            loss = criterion(pred_age, age)

            loss.backward()
            train_count += im.shape[0]

            train_loss += eval_criterion(pred_age, age).sum().detach().item()

            optimizer.step()

            
        train_loss/= train_count  

        val_loss, corr, _, val_count,*_ = test(net, valid_loader, eval_criterion) 
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), model_save_path)
            num_bad_epochs = 0
        else:
            num_bad_epochs += 1   
        
        end = time.time()
        duration = end - start
        lr = optimizer.param_groups[0]['lr']
        print('Epoch: {}, lr: {:.2E}, train loss: {:.1f}, valid loss: {:.1f}, corr: {:.2f}, best loss {:.1f}, number of epochs without improvement: {}'.format(epoch,
             lr, train_loss, val_loss, corr, best_loss, num_bad_epochs))

    return train_loss, train_count, val_loss, val_count, corr



def test(net, dataloader, eval_criterion):
  running_loss = 0
  data_count = 0 
  true_ages = []
  pred_ages = []
  sub_ids =[]

  with torch.no_grad():
      net.eval()
      for k, data in enumerate(tqdm.tqdm(dataloader)):
          im, age, ids = data
          im = im.to(device=DEVICE, dtype = torch.float)
          age = age.to(device=DEVICE, dtype=torch.float)
          age = age.reshape(-1,1)

          pred_age = net(im)
          for pred, chron_age, id in zip(pred_age, age, ids):
              pred_ages.append(pred.item())
              true_ages.append(chron_age.item())
              sub_ids.append(id) 

          running_loss += eval_criterion(pred_age, age).sum().detach().item()
          data_count += im.shape[0]

      loss = running_loss/data_count
      corr_mat = np.corrcoef(true_ages, pred_ages)
      mae = mean_absolute_error(true_ages, pred_ages)
      corr = corr_mat[0,1]

      return loss, corr, mae, data_count, sub_ids, true_ages, pred_ages