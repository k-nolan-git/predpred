import torch
import torch.utils.data as data_utils
from torch import nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathvalidate import sanitize_filepath
import copy
import time
from sklearn.metrics import root_mean_squared_error as rmse, mean_squared_error as mse, mean_absolute_percentage_error as mape, root_mean_squared_log_error as rmsle

import warnings

class lstm(nn.Module):
    def __init__(self, lstm_hs=20, lstm_nl=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hs,
                            num_layers=lstm_nl, batch_first=True)
        self.linear = nn.Linear(lstm_hs, 1)
        self.double()
    def forward(self, x):
        # LSMT input shape- (batch_size, sequence_length, input_size)
        x, _ = self.lstm(x)
        # LSMT output shape- (batch_size, sequence_length, hidden_size)
        x = x[:, -1, :] # take only last timestep
        # Linear input shape - (batch_size, hidden_size)
        x = self.linear(x) 
        # Linear output shape - (batch_size, 1)
        return x

def create_loaders(model_series, embed_dim, train_split=0.8, test_series=None, batch_size=32):

    batch_size = int(batch_size)
    embed_dim = int(embed_dim)

    if not (train_split > 0 and train_split <= 1):
        raise Exception("Inavlid train_split: 0 < test_split <= 1")
    
    if not len(model_series.shape) == 1:
        raise Exception("model_series must be 1-dimensional, not {}".format(model_series.shape))
    
    if not (test_series is None):
        if not len(test_series.shape) == 1:
            raise Exception("test_series must be 1-dimensional, not {}".format(test_series.shape))
        
    if batch_size < 1:
        raise Exception("batch_size must be positive integer")
    if embed_dim < 1:
        raise Exception("embed_dim must be positive integer")
    
    tv_dataset = create_dataset(model_series, embed_dim)
    n_obs = len(tv_dataset)
    train_len = int(train_split*n_obs)
    val_len = n_obs - train_len

    if batch_size > train_len:
        batch_size = train_len
        warnings.warn("Batch size larger than dataset size, new batch_size = {}".format(batch_size))

    train_tensor, val_tensor= torch.utils.data.random_split(tv_dataset, [train_len, val_len])
    train_loader = data_utils.DataLoader(dataset = train_tensor, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_len != 0:
        val_loader = data_utils.DataLoader(dataset=val_tensor, batch_size=val_len, shuffle=False)

    test_loader = None
    if not (test_series is None):
        test_dataset = create_dataset(test_series, embed_dim)
        test_len = len(test_dataset)
        test_loader = data_utils.DataLoader(dataset=test_dataset, batch_size=test_len, shuffle=False)

    loader_dict = {
        "train_loader" : train_loader,
        "val_loader" : val_loader,
        "test_loader" : test_loader
    }
    return loader_dict

def create_dataset(series, embed_dim): ## 1d series -> Datset of inputs (series_len-embed_dim, embed_dim, 1) and outputs (series_len-embed_dim, 1)
    s_len = len(series)
    data_size = s_len-embed_dim
    inputs = np.empty((data_size, embed_dim, 1))
    outputs = np.empty((data_size, 1)) 
    
    for i in range(data_size):
        inputs[i] = series[i:i+embed_dim].reshape(embed_dim, 1)
        outputs[i] = series[i+embed_dim:i+embed_dim+1]
        
    assert inputs.flatten()[0] == series[0]
    assert outputs.flatten()[-1] == series[-1]
    
    return data_utils.TensorDataset(torch.tensor(inputs), torch.tensor(outputs))

def calc_loss(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    n_obs = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            b_size = len(X)
            pred = model(X)
            total_loss += loss_fn(pred, y).item() * b_size
            n_obs += b_size
    loss = total_loss/n_obs
    return loss

def calc_point_metrics(model, dataloader, prefix=""):
    
    result_dict = {}
    model.eval()

    with torch.no_grad():
        inputs = dataloader.dataset.tensors[0]
        targets = dataloader.dataset.tensors[1].detach().numpy()

        preds = model(inputs).detach().numpy()

        abs_err = np.abs(preds - targets)

        result_dict["{}_max_abs_err".format(prefix)] = np.max(abs_err)
        result_dict["{}_med_abs_err".format(prefix)] = np.median(abs_err)
        result_dict["{}_mae".format(prefix)] = np.mean(abs_err)

        result_dict["{}_mse".format(prefix)] = mse(targets, preds)
        result_dict["{}_rmse".format(prefix)] = rmse(targets, preds)
        result_dict["{}_mape".format(prefix)] = mape(targets, preds)
        # result_dict["{}_rmsle".format(prefix)] = rmsle(targets, preds)

    return result_dict

def train_loop(model, dataloader, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
    loss_val = loss.item()
    return loss_val

def train_model(model, train_loader, val_loader, epochs, patience=np.inf, loss_fn=nn.L1Loss(), opt=optim.Adam, start_lr=0.1, lr_decay_factor=1,
                if_save=False, save_folder=None, save_name=None, plot=False, verbose=True):
    
    start_time = time.time()
    timings = []

    if if_save:
        model_file = sanitize_filepath(save_folder + save_name + ".pth")

    assert min(epochs) > 0

    max_epochs = max(epochs)

    optimizer=opt(model.parameters(), lr=start_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=lr_decay_factor)

    current_lr = start_lr
    lr_steps = []
    lr_values = [start_lr]

    if not verbose:
        n = np.inf
    else:
        n = int(max_epochs/10)

    x = np.arange(1, max_epochs+1, 1)
    train_loss = np.zeros(max_epochs)
    val_loss = np.zeros(max_epochs)
    best_val_loss = np.inf
    patience_count = 0
    early_stop = False
    final_epoch = max_epochs
    best_model = copy.deepcopy(model)

    if verbose:
        print("Beginning training...")

    trained_models = []

    # while t < epochs and patience_count <= patience: 
    for t in range(max_epochs): ## loop through epochs
    
        if (t+1) % n == 0:
            print("Training Epoch " + str(t+1) + " of " + str(max_epochs))
            print("Training loss: {:.6f}".format(train_loss[t-1]))
            print("Validation loss: {:.6f}".format(val_loss[t-1]))
            print("Best validation loss: {:.6f}".format(best_val_loss))

        train_loss[t] = train_loop(model, train_loader, loss_fn, optimizer) ##cycle through one epoch of training data and update model weights
        val_loss[t] = calc_loss(model, val_loader, loss_fn) ##evaluate validation loss

        if val_loss[t] < best_val_loss:
            best_val_loss = val_loss[t]
            best_model = copy.deepcopy(model)
            patience_count = 0
        else:
            patience_count += 1
        if patience_count > patience:
            if verbose:
                print("Early stopping after epoch {}".format(t+1))

            trained_models.append(best_model)
            timings.append(time.time() - start_time)
            final_epoch = t + 1
            early_stop = True
            break
            
        scheduler.step(val_loss[t])

        if plot:
            new_lr = scheduler.optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                if verbose:
                    print("Learning rate change {} -> {} (Epoch {})".format(current_lr, new_lr, t+1))
                lr_steps.append(t+1)
                lr_values.append(new_lr)
                current_lr = new_lr

        if t + 1 in epochs:
            trained_models.append(best_model)
            timings.append(time.time() - start_time)

    if plot:
        if early_stop:
            x = x[0:final_epoch]
            train_loss = train_loss[0:final_epoch]
            val_loss = val_loss[0:final_epoch]
        plt.figure()
        plt.plot(x,train_loss, label="Train Loss")
        plt.plot(x,val_loss, label="Validation Loss")
        for step in lr_steps:
            plt.axvline(x=step, color='black', ls='--')
        plt.legend()
        plt.title("Loss During Training")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()
        plt.close()

    if len(trained_models) < len(epochs):
        diff = len(epochs) - len(trained_models)
        for i in range(diff):
            trained_models.append(best_model)
            timings.append(time.time() - start_time)

    assert len(trained_models) == len(epochs)

    final_val_loss = calc_loss(best_model, val_loader, loss_fn)
    assert round(final_val_loss, 6) == round(best_val_loss, 6)

    if verbose:
        print("Final validation loss: {:.6f}".format(final_val_loss))

    if if_save:
        if verbose:
            print("Saving model to: {}".format(model_file))
        torch.save(best_model.state_dict(), model_file)

    return trained_models, final_epoch, timings

def pred_test_dts(test_dts, model, embed_dim):
    n_series = test_dts.shape[0]
    test_len = test_dts.shape[1]
    test_pred = np.zeros(test_dts.shape)
    test_pred[:, 0:embed_dim] = test_dts[:, 0:embed_dim]
    
    for i in range(embed_dim, test_len):
        inp = torch.tensor(test_pred[:, i-embed_dim:i], dtype=torch.double)
        inp = torch.reshape(inp, (n_series, embed_dim, 1))        
        test_pred[:, i] = np.reshape(model(inp).detach().numpy(), (n_series,))
       
    return test_pred