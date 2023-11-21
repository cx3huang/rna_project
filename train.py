from model import RNAModel, loss
from dataset import RNADataset

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm



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
    df = pd.read_csv('train_data_QUICK_START.csv')

    nfolds = 4
    num_workers = 12
    batch_size = 200
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(DEVICE)
    NUM_EPOCHS = 75
    show = True


    for fold in [0]:
        train_dataset = RNADataset(df, mode='train', fold=fold, nfolds=nfolds)
        train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)

        test_dataset = RNADataset(df, mode='eval', fold=fold, n_folds=nfolds)
        test_loader = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)

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

        if show:
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

        # torch.save(model.state_dict(), 'model_' + str(fold) + '.pth')
        torch.save(model.state_dict(), 'model_' + 'e75-lrneg4' + '.pth')


# rna_dataset = RNADataset(df)
# X_train, X_test, y_train, y_test = train_test_split(df[''], np.concatenate([rna_dataset.react_2A3, rna_dataset.react_DMS], axis=-1), test_size=0.20, random_state=42)