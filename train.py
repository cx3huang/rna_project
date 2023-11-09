from model import RNAModel, loss
from dataset import RNADataLoader, RNADataset

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/clean_train_data.csv')

nfolds = 4
num_workers = 5
LEARNING_RATE = 1e-4
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 5

def train(model, train_loader, loss_fn, optimizer, epoch=-1):
    total_loss = 0
    all_predictions = []
    all_targets = []
    loss_history = []

    model = model.to(DEVICE)
    model.train()  # Set model in training mode
    for i, (inputs, targets, mask) in enumerate(train_loader):
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



for fold in [0]:
    train_dataset = RNADataset(df, mode='train', fold=fold, nfolds=nfolds)
    train_loader = RNADataLoader(DataLoader(train_dataset, num_workers=num_workers, batch_size=32), device=device)

    test_dataset = RNADataset(df, mode='eval', fold=fold, n_folds=nfolds)
    test_loader = RNADataLoader(DataLoader(test_dataset, num_workers=num_workers, batch_size=32), device=device)

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

    model.eval()
    with torch.no_grad():
        (inputs, targets, mask) = next(iter(test_loader))
        outputs = model(inputs, mask)

    i = 0
    m = mask[i, :].cpu().numpy()
    tgt = targets[i, :].cpu().numpy()
    y = outputs[i, :].cpu().numpy()

    plt.plot(tgt[~m], label="Target")
    plt.plot(y[~m], label="Prediction")
    plt.legend()
    plt.title("Model prediction after training")
    plt.show()

    torch.save(model.state_dict(), 'model_' + str(fold) + '.pth')