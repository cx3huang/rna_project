import optuna
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from model import RNAModel  
from dataset import RNADataset, LengthMatchingBatchSampler  
from train import train, test, EarlyStopper, loss   
import pandas as pd

# Define the Optuna objective function
def objective(trial):
    # Hyperparameters to be tuned by Optuna
    learning_rate = 1e-4
    batch_size = 16
    dim = trial.suggest_categorical('dim', [128, 192, 256])
    depth = trial.suggest_int('depth', 6, 12)
    head_size = trial.suggest_categorical('head_size', [16, 32, 64])

    # Dataset preparation
    df = pd.read_csv('data/train_data_QUICK_START.csv')  # Load your dataset
    train_dataset = RNADataset(df, mode='train')
    test_dataset = RNADataset(df, mode='test')

    # DataLoader
    len_train_sampler = LengthMatchingBatchSampler(RandomSampler(train_dataset), batch_size=batch_size, drop_last=True)
    train_loader = DataLoader(train_dataset, 
                                batch_sampler=len_train_sampler, 
                                num_workers=0)
    len_test_sampler = LengthMatchingBatchSampler(SequentialSampler(test_dataset), batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, 
                                batch_sampler=len_test_sampler,
                                num_workers=0)

    # Model initialization
    model = RNAModel(dim=dim, depth=depth, head_size=head_size)
    # model = RNAModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopper = EarlyStopper(patience=5, min_delta=0.05)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_loader, loss, optimizer, epoch)
        test_loss = test(model, test_loader, loss, epoch)
        
        if early_stopper.early_stop(test_loss):
            break

    return test_loss

if __name__ == '__main__':
    NUM_EPOCHS = 20  # Set the number of epochs for each trial
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=6)  # Adjust the number of trials as needed

    # Retrieving the best hyperparameters
    best_params = study.best_params
    print("Best hyperparameters: ", best_params)