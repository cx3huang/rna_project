from model import RNAModel, loss
from dataset import RNADataset, LengthMatchingBatchSampler

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import datetime



def train(model, train_loader, loss_fn, optimizer, epoch=-1):
    total_loss = 0
    all_predictions = []
    all_targets = []
    loss_history = []

    model = model.to(DEVICE)
    model.train()  # Set model in training mode
    for i, (inputs, targets, mask) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        inputs, targets, mask = inputs.to(DEVICE), targets.to(DEVICE), mask.to(DEVICE)
        outputs = model(inputs, mask)
        loss = loss_fn(outputs, targets, mask)
        loss.backward()
        optimizer.step()

        # Track some values to compute statistics
        total_loss += loss.item()

        # Save loss every 100 batches
        if (i % 100 == 0) and (i > 0):
            running_loss = total_loss / (i + 1)
            loss_history.append(running_loss)
            # print(f"Epoch {epoch + 1}, batch {i + 1}: loss = {running_loss:.2f}")

    final_loss = total_loss / len(train_loader)
    # Print average loss and accuracy
    print(f"Epoch {epoch + 1} done. Average train loss = {final_loss:.2f}")
    return final_loss


def test(model, test_loader, loss_fn, epoch=-1):
    """
    Tests a model for one epoch of test data.

    Note:
        In testing and evaluation, we do not perform gradient descent optimization, so steps 2, 5, and 6 are not needed.
        For performance, we also tell torch not to track gradients by using the `with torch.no_grad()` context.

    :param model: PyTorch model
    :param test_loader: PyTorch Dataloader for test data
    :param loss_fn: PyTorch loss function
    :kwarg epoch: Integer epoch to use when printing loss and accuracy

    :returns: Accuracy score
    """
    total_loss = 0
    all_predictions = []
    all_targets = []
    model = model.to(DEVICE)
    model.eval()  # Set model in evaluation mode
    for i, (inputs, targets, mask) in enumerate(test_loader):
        inputs, targets, mask = inputs.to(DEVICE), targets.to(DEVICE), mask.to(DEVICE)
        with torch.no_grad():
            outputs = model(inputs, mask)
            loss = loss_fn(outputs, targets, mask)

            # Track some values to compute statistics
            total_loss += loss.item()

    final_loss = total_loss / len(test_loader)
    # Print average loss and accuracy
    print(f"Epoch {epoch + 1} done. Average test loss = {final_loss:.2f}")
    return final_loss

if __name__ == '__main__':
    df = pd.read_csv('data/train_data_QUICK_START.csv')
    # df = pd.read_csv('rna_project/data/train_data_QUICK_START.csv')

    NFOLDS = 4
    NUM_WORKERS = 12
    BATCH_SIZE = 200
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(DEVICE)
    NUM_EPOCHS = 10
    SHOW = False
    SAVE = False

    st = time.time()
    for fold in [0]:
        # train_dataset = RNADataset(df, mode='train', fold=fold, nfolds=NFOLDS)
        # train_loader = DataLoader(train_dataset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True)
        # test_dataset = RNADataset(df, mode='eval', fold=fold, n_folds=NFOLDS)
        # test_loader = DataLoader(test_dataset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=False)

        train_dataset = RNADataset(df, mode='train', fold=fold, nfolds=NFOLDS)
        train_sampler = RandomSampler(train_dataset)
        len_train_sampler = LengthMatchingBatchSampler(train_sampler, batch_size=BATCH_SIZE, drop_last=True)
        train_loader = DataLoader(train_dataset, 
                                  batch_sampler=len_train_sampler, 
                                  num_workers=NUM_WORKERS)

        test_dataset = RNADataset(df, mode='eval', fold=fold, n_folds=NFOLDS)
        test_sampler = SequentialSampler(test_dataset)
        len_test_sampler = LengthMatchingBatchSampler(test_sampler, batch_size=BATCH_SIZE, drop_last=True)
        test_loader = DataLoader(test_dataset, 
                                 batch_sampler=len_test_sampler,
                                 num_workers=NUM_WORKERS)

        model = RNAModel()
        model = model.to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_losses = []
        test_losses = []

        for epoch in range(NUM_EPOCHS):
            train_loss = train(model, train_loader, loss, optimizer, epoch)
            test_loss= test(model, test_loader, loss, epoch)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

        if SHOW:
            model.eval()
            with torch.no_grad():
                (inputs, targets, mask) = next(iter(test_loader))
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                mask = mask.to(DEVICE)
                outputs = model(inputs, mask)

            i = 0
            m = mask[i, :].cpu().numpy()
            tgt1 = targets[i, :, 0].cpu().numpy()
            tgt2 = targets[i, :, 1].cpu().numpy()
            y1 = outputs[i, :, 0].cpu().numpy()
            y2 = outputs[i, :, 1].cpu().numpy()

            fig, (ax1, ax2) = plt.subplots(1,2)
            fig.suptitle('Model predictions: 2A3 (left) & DMS (right)')

            ### TODO: current mask is all true; need to verify which parts are actually true. ###
            ax1.plot(tgt1[m], label="Target")
            ax1.plot(y1[m], label="Prediction")
            ax1.legend()
            ax2.plot(tgt2[m], label="Target")
            ax2.plot(y2[m], label="Prediction")
            ax2.legend()
            plt.show()

        if SAVE:
            # torch.save(model.state_dict(), 'model_' + str(fold) + '.pth')
            torch.save(model.state_dict(), 'data/model_' + 'e50-lrneg4-lmbs' + '.pth')

    end = time.time()
    avg_elapsed = int(round((end - st)/NUM_EPOCHS))
    print("Time per epoch: " + str(avg_elapsed // 60) + " mins " + str(avg_elapsed % 60) + " secs")

# rna_dataset = RNADataset(df)
# X_train, X_test, y_train, y_test = train_test_split(df[''], np.concatenate([rna_dataset.react_2A3, rna_dataset.react_DMS], axis=-1), test_size=0.20, random_state=42)