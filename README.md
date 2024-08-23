# BrainAge_Federated
This repository implements the pre-trained BrainAge model within a federated learning framework, using [Flower](https://github.com/adap/flower.git) to predict brain age from MRI data while ensuring data privacy.


## How to run:

### Setup
1. Clone this repository to every computers in your federated learning network. You can choose your desired number of clients. *Note*: You need at least one server and two clients.
2. Create and activate a new virtual environment with Python 3.10 within the cloned repository (suggested name: *'Flower_3.10'*).
3. Install all requirements in *'./conf/requirements.txt'* file:

   ```bash
   pip install -r ./conf/requirements.txt

### Data Preparation   
5. Create a folder named *'data'* in the main directory and add CSV files containing details such as ID, age, and the path to the NIFTI file for each subject. Ensure the CSV includes columns:
   - *'ID'*: unique identifier of each subject/MRI scan.
   - *'Age'*: chronological age of the subject in years.
   - *'file_name'*: absolute path to the Nifti file for each subject.

4. Name each client's dataset as *'clientName_dataset'* (e.g., *'c1_dataset.csv'*). Place these files in the *'data'* folder on the respective clients.
5. Name centralized datasets as *'centralized_test_dataset.csv'* and *'centralized_val_dataset.csv'*. Place these files in the *'data'* folder on the server.

### Running scripts
7. Run *'server.py'* on the server machine with the following command:
   
   ```bash
   python server.py project=projectName
   
9. Run *'client.py'* on the client machines with the following command:
   
   ```bash
   pyhton client.py --client clientName --project projectName

### Note
* The pretrained model is a BrainAge T1 model obtained from [BrainAge T1 model](https://github.com/MIDIconsortium/BrainAge/blob/main/HBM_models/T1/model.pt). This model file is located at *'./models/wood/wood_T1.pt'*.
* Parameters such as client name (*--client*) and project name (*--project*) must be specified when running *'client.py'*, as they are used to name files when saving results. (e.g., *'pyhton client.py --client c1 --project FedAvg'*)
* Overwrite the *'project'* parameter when running *'server.py'*. (e.g., *'python server.py project=FedAvg'*)
* Set the *'q_param'* for the qFedAvg strategy in the *'./conf/base.yaml'*.
