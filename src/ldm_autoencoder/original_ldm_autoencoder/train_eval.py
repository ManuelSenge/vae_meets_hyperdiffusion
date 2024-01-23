
import torch

def train(model, iterator, acc_grad_batches, optimizer, loss, device, warmup, beta, variational, normalizing_constant=1, remove_std_zero_indices = True):
    epoch_loss_val_mse = 0
    epoch_loss_val_kl = 0

    if remove_std_zero_indices:
        padding = 21
    else:
        padding = 31

    model.train()
    optimizer.zero_grad() # clear grads
    for i, batch in enumerate(iterator): # batch is simply a batch of ci-matricies as a tensor as x and y are the same 
        # attention_mask, base_ids are already on device
        weights, weights_prev, weights_prev = batch
        weights = weights.view(weights.shape[0], 1, weights.shape[1])
        weights = torch.nn.functional.pad(weights, (0, padding)).to(device)

        predictions = model(weights)

        loss_val_mse, loss_val_kl = loss(predictions[:, :, :-padding], weights[:, :, :-padding], variational, model)
        # during warmup only train mse loss
        if warmup or not variational:
            loss_val = loss_val_mse
        else:
            loss_val = loss_val_mse + beta * loss_val_kl
            
        loss_val.backward()

        if ((i+1) % acc_grad_batches == 0) or (i + 1 == len(iterator)): 
            optimizer.step()
            optimizer.zero_grad() # clear gradients after every step

        epoch_loss_val_mse += loss_val_mse.item()
        epoch_loss_val_kl += loss_val_kl.item()


    return epoch_loss_val_mse / (len(iterator) * normalizing_constant), epoch_loss_val_kl / len(iterator)

def evaluate(model, iterator, loss, device, beta, variational, normalizing_constant=1, remove_std_zero_indices = True):
    epoch_loss_val_mse = 0
    epoch_loss_val_kl = 0

    if remove_std_zero_indices:
        padding = 21
    else:
        padding = 31
    
    model.eval()
    for i, batch in enumerate(iterator): # batch is simply a batch of ci-matricies as a tensor as x and y are the same 
        # attention_mask, base_ids are already on device
        weights, weights_prev, weights_prev = batch
        weights = weights.view(weights.shape[0], 1, weights.shape[1])
        weights = torch.nn.functional.pad(weights, (0, padding)).to(device)

        predictions = model(weights)

        loss_val_mse, loss_val_kl = loss(predictions[:, :, :-padding], weights[:, :, :-padding], variational, model)
        if not variational:
            loss_val = loss_val_mse
        else:
            loss_val = loss_val_mse + beta * loss_val_kl
  
        epoch_loss_val_mse += loss_val_mse.item()
        epoch_loss_val_kl += loss_val_kl.item()


    return epoch_loss_val_mse / (len(iterator) * normalizing_constant), epoch_loss_val_kl / len(iterator)