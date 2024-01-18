
import torch
import time

def train(model, iterator, optimizer, loss, device, warmup):
    epoch_loss_val_mse = 0
    epoch_loss_val_kl = 0
    
    model.train()
    for i, batch in enumerate(iterator): # batch is simply a batch of ci-matricies as a tensor as x and y are the same 
        start = time.time()
        # attention_mask, base_ids are already on device
        Y, X = batch
        
        Y = Y.to(device)
        X = X.to(device)


        optimizer.zero_grad() # clear gradients first

        predictions, posterior = model(X, sample_posterior=True)

        loss_val_mse, loss_val_kl = loss(predictions, Y, True, model)

        # during warmup only train mse loss
        if warmup:
            loss_val = loss_val_mse
        else:
            loss_val = loss_val_mse + 0.000001 * loss_val_kl
        
            
        loss_val.backward()
        optimizer.step()
        
        epoch_loss_val_mse += loss_val_mse.item()
        epoch_loss_val_kl += loss_val_kl.item()
        end = time.time()
        break

    return epoch_loss_val_mse / len(iterator), epoch_loss_val_kl / len(iterator), posterior

def evaluate(model, iterator, loss, device):
    epoch_loss_val_mse = 0
    epoch_loss_val_kl = 0
    
    model.eval()
    for i, batch in enumerate(iterator): # batch is simply a batch of ci-matricies as a tensor as x and y are the same 
        # attention_mask, base_ids are already on device
        Y, X = batch
        
        Y = Y.to(device)
        X = X.to(device)

        predictions, posterior = model(X, sample_posterior=True)

        loss_val_mse, loss_val_kl = loss(predictions, Y, True, model)
  
        epoch_loss_val_mse += loss_val_mse.item()
        epoch_loss_val_kl += loss_val_kl.item()
        break

    return epoch_loss_val_mse / len(iterator),  epoch_loss_val_kl / len(iterator)