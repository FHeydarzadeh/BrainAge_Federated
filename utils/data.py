"""
data script:
This script is used to prepare the data as the input to the net.
==> Inspired by the GitHub repository of Wood et al. 2022 (https://github.com/MIDIconsortium/BrainAge)
"""

import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from monai.transforms import (Compose, LoadImage, ToTensor)
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import warnings
warnings.filterwarnings("ignore")



def img_to_tensor(image_path):
   """
   Prepare preprocessed image for input to net.

   :param image_path: str, path to T1 NIFTI
   :return: tensor, prepared image
   """
   img = nib.load(image_path).get_fdata()  # Load preprocessed NIFTI
   img_tensor = torch.Tensor(img)  # Convert to torch Tensor
   img_tensor = torch.unsqueeze(img_tensor, dim=0)  # Add dimension
   img_tensor = (img_tensor - torch.mean(img_tensor)) / torch.std(img_tensor)  # Normalise tensor

   return img_tensor



class BrainAgeDataset(Dataset):
    """Brain-age fine-tuning dataset"""

    def __init__(self, df, transform = None):
        self.file_frame = df 
        self.transform = transform
        
    def __len__(self):
        return len(self.file_frame)

    def __getitem__(self, idx):
        stack_name = self.file_frame.iloc[idx]['file_name']
        tensor = self.transform(stack_name)  
        tensor = (tensor - tensor.mean())/tensor.std()
        tensor = torch.clamp(tensor,-1, 5)
        tensor = torch.reshape(tensor, (1, 130, 130, 130))
        age = self.file_frame.iloc[idx]['Age'] 
        id = self.file_frame.iloc[idx]['ID']  
        return tensor, age, id
    


def get_dataloaders(csv_file,
                    batch_size,
                    random_seed=10,
                    k_folds=5):
    
    np.random.seed(random_seed)

                         
    df = pd.read_csv(csv_file)
    IDs = df['ID'].unique().tolist()

    np.random.shuffle(IDs)
    
    train_val_ids, test_ids = train_test_split(IDs, test_size=0.15, random_state=random_seed)
    train_val_df = df[df['ID'].isin(train_val_ids)]
    test_df = df[df['ID'].isin(test_ids)]
    
    transforms = Compose([LoadImage(image_only=True, ensure_channel_first=True), ToTensor()])
    train_val_dataset = BrainAgeDataset(train_val_df, transform=transforms)
    test_dataset = BrainAgeDataset(test_df, transform=transforms)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # K-fold cross validation
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    train_loaders = []
    val_loaders = []
    num_train_samples = []
    num_val_samples = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_df)):

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(train_val_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(train_val_dataset, batch_size=batch_size, sampler=val_sampler)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

        num_train_samples.append(len(train_idx))
        num_val_samples.append(len(val_idx))

    # Print number of train and validation samples in each folds
    for i in range(fold+1):
        print(f"Fold {i+1}: train samples {num_train_samples[i]}, validation samples {num_val_samples[i]}")
    # Print number of test samples in total
    print('test samples: {}'.format(len(test_df)))

    return train_loaders, val_loaders, test_loader



def get_centralized_dataloader(csv_file, batch_size):
                       
    df = pd.read_csv(csv_file)
    IDs = df['ID'].unique().tolist()
    
    transforms = Compose([LoadImage(image_only=True, ensure_channel_first=True), ToTensor()])
    centralized_dataset = BrainAgeDataset(df, transform=transforms)

    centralized_loader = DataLoader(centralized_dataset, batch_size=batch_size, shuffle=False)

    return centralized_loader





    
    
