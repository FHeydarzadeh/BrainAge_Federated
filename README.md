# BrainAge_Federated
Implementation of the BrainAge model in a federated learning framework using Flower to predict brain age from MRI data while preserving data privacy.


## How to run:

### Setup
1. Create a new virtual environment with Python 3.10.
2. Install all requirements in *./conf/requirements.txt*
3. Create a folder "data" in the main directory and put the .csv file inside the folder. The csv file should include the following columns:
   - *'ID'*: unique identifier of each subject/MRI scan.
   - *'Age'*: chronological age of the subject in years.
   - *'file_name'*: absolute path to the Nifti file for each subject.

### Data Preparation
4. Name dataset of each client as *'clientName_dataset'*. EX: c1_dataset.csv
5. Name centralized dataset as *'centralized_test_dataset.csv'* and *'centralized_val_dataset.csv'*.

### Running scripts
7. Run *'server.py'* on the server machine with the following command:
   
   ```bash
   server.py project=projectName
   
9. Run *'client.py'* on the client machines with the following command:
   
   ```bash
   pyhton client.py --client clientName --project projectName

### Note
1. The pretrained model is a BrainAge T1 model obtained from [BrainAge T1 model](https://github.com/MIDIconsortium/BrainAge/blob/main/HBM_models/T1/model.pt). This model is inside Models folder (*'./models/wood/wood_T1.pt'*).
2. Some aprameters should be passed in command line when running *'client.py'*, namely client name (*--client*) and project name (*--project*). These are essential as they are used to naming the files when saving results. (Example: *'pyhton client.py --client c1 --project FedAvg --gpu'*)
3. Parameter *'project'* should be overwritten when running *'server.py'* (Eample: *'python server.py project=FedAvg'*)
4. The *'q_param'* for using in qFedAvg strategy should be set in *'./conf/base.yaml'*
