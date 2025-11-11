import numpy as np
import time 
import torch

from torch.nn import Module
from torch.utils.data import DataLoader

from ...interfaces import BaseModel, BaseDataset
from ...tools.model_evaluation_tools import validate_model, evaluate_dataset
from ...tools.device_tools import get_device

def vanilla_training_loop(model:BaseModel, train_dset:BaseDataset, opt, lr:float, batch_size:int, epochs:int, val_dset:BaseDataset, validate:bool, verbose:bool) -> dict[str, dict[str, list]]:
    """The base training loop that is common to most models"""
    # Put Data into a DataLoader
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    
    # Put Data Standardiser onto the same device as everyone else
    DEVICE = get_device()

    train_dset.set_standardiser_device(DEVICE)

    # Initialise Optimiser
    assert isinstance(model, Module), "Error: `vanilla_training_loop` can only be used for torch-based models"
    opt = opt(model.parameters(), lr=lr)

    # Early Stopping Parameters
    grace_period = 5 # Number of iterations that need to pass with no improvement in validation accuracy
    no_improvement_count = 0
    max_clean_acc = 0
    max_strat_acc = 0

    # Training Loop
    total_time = time.time()
    train_losses = []
    if validate:
        valid_clean_accuracies = []
        valid_strat_accuracies = []
        valid_losses = []

    for epoch in range(epochs):
        t1 = time.time()
        train_losses.append([])
        batch = 1
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            l = model.loss(model, X, y)
            l.backward()
            opt.step()
            train_losses[-1].append(l.item())
            batch += 1

        #TODO: Validation evaluation should go here
        if validate:
            with torch.no_grad():
                # Call with no_grad to ensure no gradients are computed.
                clean_accuracy, strat_accuracy, valid_loss = validate_model(model, val_dset, model.loss)
                valid_clean_accuracies.append(clean_accuracy)
                valid_strat_accuracies.append(strat_accuracy)
                valid_losses.append(valid_loss)

            no_improvement_count += 1

            if clean_accuracy>max_clean_acc:
                max_clean_acc = clean_accuracy
                no_improvement_count = 0
                model.save_params()

            if strat_accuracy>max_strat_acc:
                max_strat_acc = strat_accuracy
                no_improvement_count = 0
                model.save_params()

            if no_improvement_count>=grace_period:
                if verbose:
                    print(f"Model Validation Accuracy has not improved in {no_improvement_count} epochs. Stopping training as model has converged") 
                break

        t2 = time.time()

        if not validate:
            # Don't overwrite models if we're doing validation and think we've already hit a best
            model.save_params() # Store intermediate parameter values

        if verbose:
            print(f"End of Epoch: {epoch+1}: {model.get_weights()}")
            print(f"------------- epoch {epoch+1} / {epochs} | time: {t2-t1} sec | loss: {np.mean(train_losses[-1])}")
            if validate:
                print(f"\t\t\t    validation: clean acc: {clean_accuracy} | strat acc: {strat_accuracy} | loss: {valid_loss}")
    
    if verbose:
        print(f"Total training time: {time.time()-total_time} seconds")

    out = {'train': {'data stats': evaluate_dataset(train_dset), 'train-loss': train_losses}}
    if validate:
        out['validation'] = {'data stats': evaluate_dataset(val_dset), 'valid-clean-acc': valid_clean_accuracies, 'valid-strat-acc': valid_strat_accuracies, 'valid-loss': valid_losses}

    return out