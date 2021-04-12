import numpy as np
import torch
from utils import ArielMLDataset, ChallengeMetric, Baseline, simple_transform
from torch.utils.data.dataloader import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam


# paths to data dirs
lc_train_path = ...
params_train_path = ...
lc_test_path = ...

# training parameters
train_size = 16
val_size = 16
epochs = 3

# hyper-parameters
H1 = 1024
H2 = 256

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Training
    dataset_train = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=0,
                                   max_size=train_size, transform=simple_transform, device=device)
    # Validation
    dataset_val = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=train_size,
                                 max_size=val_size, transform=simple_transform, device=device)

    # Loaders
    batch_size = int(train_size / 4)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size)
    loader_eval = DataLoader(dataset_eval, batch_size=batch_size)

    # Define baseline model
    baseline = Baseline(H1=H1, H2=H2).double().to(device)

    # Define Loss, metric and optimizer
    loss_function = MSELoss()
    challenge_metric = ChallengeMetric()
    opt = Adam(baseline.parameters())

    # Lists to record train and val scores
    train_losses = []
    val_losses = []
    val_scores = []
    best_val_score = 0.

    for epoch in range(epochs):
        print("epoch", epoch)
        train_loss = 0
        val_loss = 0
        val_score = 0
        baseline.train()
        for k, item in enumerate(loader_train):
            pred = baseline(item['lc'])
            loss = loss_function(item['target'], pred)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss = train_loss / len(loader_train)
        baseline.eval()
        for k, item in enumerate(loader_val):
            pred = baseline(item['lc'])
            loss = loss_function(item['target'], pred)
            score = challenge_metric.score(item['target'], pred)
            val_loss += loss.item()
            val_score += score.item()
        val_loss /= len(loader_val)
        val_score /= len(loader_val)
        print('Training loss', round(train_loss, 6))
        print('Val loss', round(val_loss, 6))
        print('Val score', round(val_score, 2))

        train_losses += [train_loss]
        val_losses += [val_loss]
        val_scores += [val_score]

    np.savetxt('outputs/train_losses.txt', np.array(train_losses))
    np.savetxt('outputs/val_losses.txt', np.array(val_losses))
    np.savetxt('outputs/val_scores.txt', np.array(val_scores))
    torch.save(baseline, 'outputs/model_state.pt')