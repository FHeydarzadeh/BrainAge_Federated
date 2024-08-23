import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns




def get_distributed_dfs(client_name, project_name):

    main_dir = './results/'
    client_dir = os.path.join(main_dir, client_name)
    project_dir = os.path.join(client_dir, project_name)
    train_result_path = project_dir + '/' +project_name + '_train_results.csv'
    val_result_path = project_dir + '/' +project_name + '_val_results.csv'
    test_result_path = project_dir + '/' + project_name + '_test_results.csv'
    train_df = pd.read_csv(train_result_path)
    val_df = pd.read_csv(val_result_path)
    test_df = pd.read_csv(test_result_path)
    
    return train_df, val_df, test_df



def get_pkl_data(project_name):

    main_dir = './results/centralized/'
    file_name = project_name +'_centralized_results'
    file_path = main_dir + project_name +'_centralized_results.pkl'

    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    return data 



def create_dataframe(losses, metrics):
    # Convert losses to DataFrame
    df_losses = pd.DataFrame(losses, columns=['server_round', 'loss'])

    df_metrics = pd.DataFrame({
        'server_round': [item[0] for item in metrics['corr']],
        'corr': [item[1] for item in metrics['corr']],
        'mae': [item[1] for item in metrics['mae']],
        'count': [item[1] for item in metrics['count']]
    })

    for variable in ['sub_id', 'true_age', 'pred_age']:
        temp_df = pd.DataFrame(metrics[variable], columns=['server_round', variable])
        df_metrics = pd.merge(df_metrics, temp_df, on='server_round', how='outer')

    df_final = pd.merge(df_losses, df_metrics, on='server_round', how='outer')
    new_df = df_final.copy()
    new_df = new_df.apply(pd.Series.explode)
    
    return new_df



def get_centralized_df(project_name):

    data = get_pkl_data(project_name)
    centralized_losses = data['losses_centralized']
    centralized_metrics = data['metrics_centralized']

    df = create_dataframe(centralized_losses, centralized_metrics)

    # only the last round which is the test round
    filtered_df = df[df['server_round'] == 20]

    # # only the first round which shows the initial model performance
    # filtered_df = df[df['server_round'] == 0]

    return filtered_df



def get_project_dfs(project_name, client_list):

    train_dfs = []
    val_dfs = []
    test_dfs = []

    for client in client_list:
        train_df, val_df, test_df = get_distributed_dfs(client, project_name)

        train_dfs.append((client, train_df))
        val_dfs.append((client, val_df))
        test_dfs.append((client, test_df))

    centralized_df = get_centralized_df(project_name)

    return train_dfs, val_dfs, test_dfs, centralized_df



def get_dfs(client_list, project_list):

    train_df_list = []
    val_df_list = []
    test_df_list = []
    centralized_df_list = []

    for project in project_list:

        train_dfs, val_dfs, test_dfs, centralized_df = get_project_dfs(project, client_list)

        centralized_df_list.append((project, centralized_df))

        for client, train_df in train_dfs:
            train_df_list.append((project, client, train_df))
        for client, val_df in val_dfs:
            val_df_list.append((project, client, val_df))
        for client, test_df in test_dfs:
            test_df_list.append((project, client, test_df))
            

    return train_df_list, val_df_list, test_df_list, centralized_df_list



def plot_train_loss(dfs_list, num_client):

    fig, axes = plt.subplots(1, num_client, figsize=(5 * num_client, 5), sharey=True)
    if num_client == 1:  
        axes = [axes]
    
    min_loss = min(df['train_loss'].min() for _, _, df in dfs_list)
    max_loss = max(df['train_loss'].max() for _, _, df in dfs_list)

    colors = {'FedAvg-AvgModel': 'blue', 'qFedAvg-0.00001-AvgModel': 'green', 'qFedAvg-0.001-AvgModel': 'red', 'qFedAvg-0.2-AvgModel': 'orange',
            'FedAvg-BestModel': 'blue', 'qFedAvg-0.00001-BestModel': 'green', 'qFedAvg-0.001-BestModel': 'red', 'qFedAvg-0.2-BestModel': 'orange',
            
            'FedAvg-Dist1': 'blue', 'qFedAvg-0.000001-Dist1': 'green', 'qFedAvg-0.0001-Dist1': 'red', 'qFedAvg-0.001-Dist1': 'orange',
            'FedAvg-Dist2': 'blue', 'qFedAvg-0.000001-Dist2': 'green', 'qFedAvg-0.0001-Dist2': 'red', 'qFedAvg-0.001-Dist2': 'orange',
            
            'FedAvg-Dist1-pop': 'blue', 'qFedAvg-0.000001-Dist1-pop': 'green', 'qFedAvg-0.0001-Dist1-pop': 'red', 'qFedAvg-0.001-Dist1-pop': 'orange'}

    for idx, (project, client, df) in enumerate(dfs_list):
        
        agg_method = project
        title = client

        color = colors.get(project, 'black')
        
        ax = axes[idx % num_client]
        ax.plot(df['server_round'], df['train_loss'], label=f"{agg_method}", color=color)
        ax.set_title(f"{title}", fontsize=20)
        ax.set_xlabel("Server Round", size='large')
        ax.set_ylabel("Train Loss", size='large')
        ax.legend()
        ax.set_ylim([min_loss * 0.95, max_loss * 1.05])

        # Set the x-axis ticks to be exactly the server rounds
        ax.set_xticks(range(1, 21))  
        ax.set_xticklabels(range(1, 21))

        ax.set_yticks(range(0, 10))  
        ax.set_yticklabels(range(0, 10))

    plt.tight_layout()
    plot_dir = './results/plots/'
    fig.savefig(os.path.join(plot_dir, 'train_loss.png'))



def plot_val_loss(dfs_list, num_client):

    fig, axes = plt.subplots(1, num_client, figsize=(5 * num_client, 5), sharey=True)
    if num_client == 1:  
        axes = [axes]
    
    min_loss = min(df['loss'].min() for _, _, df in dfs_list)
    max_loss = max(df['loss'].max() for _, _, df in dfs_list)

    colors = {'FedAvg-AvgModel': 'blue', 'qFedAvg-0.00001-AvgModel': 'green', 'qFedAvg-0.001-AvgModel': 'red', 'qFedAvg-0.2-AvgModel': 'orange',
            'FedAvg-BestModel': 'blue', 'qFedAvg-0.00001-BestModel': 'green', 'qFedAvg-0.001-BestModel': 'red', 'qFedAvg-0.2-BestModel': 'orange',
            
            'FedAvg-Dist1': 'blue', 'qFedAvg-0.000001-Dist1': 'green', 'qFedAvg-0.0001-Dist1': 'red', 'qFedAvg-0.001-Dist1': 'orange',
            'FedAvg-Dist2': 'blue', 'qFedAvg-0.000001-Dist2': 'green', 'qFedAvg-0.0001-Dist2': 'red', 'qFedAvg-0.001-Dist2': 'orange',
            
            'FedAvg-Dist1-pop': 'blue', 'qFedAvg-0.000001-Dist1-pop': 'green', 'qFedAvg-0.0001-Dist1-pop': 'red', 'qFedAvg-0.001-Dist1-pop': 'orange'}
    
    for idx, (project, client, df) in enumerate(dfs_list):
        
        agg_method = project
        title = client

        color = colors.get(project, 'black')
        
        ax = axes[idx % num_client]
        ax.plot(df['server_round'], df['loss'], label=f"{agg_method}", color=color)
        ax.set_title(f"{title}", fontsize=20)
        ax.set_xlabel("Server Round", size='large')
        ax.set_ylabel("Val Loss", size='large')
        ax.legend()
        ax.set_ylim([min_loss * 0.95, max_loss * 1.05])

        # Set the x-axis ticks to be exactly the server rounds
        ax.set_xticks(range(1, 21))  
        ax.set_xticklabels(range(1, 21))

        ax.set_yticks(range(0, 10))  
        ax.set_yticklabels(range(0, 10))

    plt.tight_layout()
    plot_dir = './results/plots/'
    fig.savefig(os.path.join(plot_dir, 'val_loss.png'))
    #plt.show()



def plot_client_scatter(dfs_list, num_client):

    fig, axes = plt.subplots(3, 4, figsize=(15, 10) , constrained_layout=True)  # 3 rows, 4 columns 

    # Assuming dfs_list contains dataframes for C1, C2, C3 respectively
    for idx, (project, client, df) in enumerate(dfs_list):
        
        agg_method = project
        title = client

        num_rows = 3
        num_cols = 4
        
        col = idx // num_rows  # Integer division to determine column
        row = idx % num_rows  # Remainder to determine row

        ax = axes[row, col] 

        # Plotting
        sns.scatterplot(x='true_age', y='pred_age', data=df, ax=ax, color='blue', alpha=0.6,)

        # Best fit line
        z = np.polyfit(df['true_age'], df['pred_age'], 1)
        p = np.poly1d(z)
        sns.lineplot(x=df['true_age'], y=p(df['true_age']), ax=ax, color='green', linestyle='--', label='Best Fit Line')
        
        # True fit line (y=x line)
        sns.lineplot(x=df['true_age'], y=df['true_age'], ax=ax, color='red', label='True Fit Line')

        # Labeling
        ax.set_xlabel('Chronological Age')
        ax.set_ylabel('Predicted Age')
        ax.set_title(f'MAE: {df["mae"].iloc[0]:.3f}, Corr: {df["corr"].iloc[0]:.3f}')
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to not overlap
    plot_dir = './results/plots/'
    fig.savefig(os.path.join(plot_dir, 'client_scatter_plot.png'))
    # plt.show()



def plot_centralized_scatter(dfs_list, num_client):

    fig, axes = plt.subplots(1, 4, figsize=(15, 5) , constrained_layout=True)  # 1 rows, 4 columns 

    # Assuming dfs_list contains dataframes for C1, C2, C3 respectively
    for idx, (project, df) in enumerate(dfs_list):
        
        agg_method = project

        ax = axes[idx] 

        # Ensure data types are correct
        df['true_age'] = pd.to_numeric(df['true_age'], errors='coerce')
        df['pred_age'] = pd.to_numeric(df['pred_age'], errors='coerce')

        # Plotting
        sns.scatterplot(x='true_age', y='pred_age', data=df, ax=ax, color='blue', alpha=0.6)

        # Best fit line
        z = np.polyfit(df['true_age'], df['pred_age'], 1)
        p = np.poly1d(z)
        sns.lineplot(x=df['true_age'], y=p(df['true_age']), ax=ax, color='green', linestyle='--', label='Best Fit Line')
        
        # True fit line (y=x line)
        sns.lineplot(x=df['true_age'], y=df['true_age'], ax=ax, color='red', label='True Fit Line')

        ax.set_aspect('equal')
        # Labeling
        ax.set_xlabel('Chronological Age')
        ax.set_ylabel('Predicted Age')
        ax.set_title(f'MAE: {df["mae"].iloc[0]:.3f}, Corr: {df["corr"].iloc[0]:.3f}')
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to not overlap
    plot_dir = './results/plots/'
    fig.savefig(os.path.join(plot_dir, 'centralized_scatter_plot.png'))
    # plt.show()




def main(client_list, project_list):

    num_client = len(client_list)

    # each list contains list of (project, client, df) ex: (FedAvg, c1, train_df)
    train_df_list, val_df_list, test_df_list, centralized_df_list = get_dfs(client_list, project_list)

    plot_train_loss(train_df_list, num_client)
    plot_val_loss(val_df_list, num_client)
    plot_client_scatter(test_df_list, num_client)
    plot_centralized_scatter(centralized_df_list, num_client)

    
    

if __name__ == '__main__':

    

    # AvgModel: 'FedAvg-AvgModel', 'qFedAvg-0.00001-AvgModel', 'qFedAvg-0.001-AvgModel', 'qFedAvg-0.2-AvgModel'
    # BestModel: 'FedAvg-BestModel', 'qFedAvg-0.00001-BestModel', 'qFedAvg-0.001-BestModel', 'qFedAvg-0.2-BestModel'
    # Dist1: 'FedAvg-Dist1', 'qFedAvg-0.000001-Dist1', 'qFedAvg-0.0001-Dist1', 'qFedAvg-0.001-Dist1'
    # Dist2: 'FedAvg-Dist2', 'qFedAvg-0.000001-Dist2', 'qFedAvg-0.0001-Dist2', 'qFedAvg-0.001-Dist2'
    # Dist1-pop : 'FedAvg-Dist1-pop', 'qFedAvg-0.000001-Dist1-pop', 'qFedAvg-0.0001-Dist1-pop', 'qFedAvg-0.001-Dist1-pop'
    project_list = ['FedAvg-Dist1-pop', 'qFedAvg-0.000001-Dist1-pop', 'qFedAvg-0.0001-Dist1-pop', 'qFedAvg-0.001-Dist1-pop']
    client_list = ['c1', 'c2', 'c3']
    main(client_list, project_list)

