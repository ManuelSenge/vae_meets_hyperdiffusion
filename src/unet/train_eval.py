
import torch

def train(model, iterator, optimizer, loss, device, warmup, variational, remove_indx):
    epoch_loss_val_mse = 0
    epoch_loss_val_kl = 0

    if remove_indx:
        padding = 21
    else:
        padding = 31
    
    model.train()
    for i, batch in enumerate(iterator): # batch is simply a batch of ci-matricies as a tensor as x and y are the same 
        # attention_mask, base_ids are already on device
        weights, weights_prev, weights_prev = batch
        weights = torch.nn.functional.pad(weights, (0, padding)).to(device)
        optimizer.zero_grad() # clear gradients first

        predictions = model(weights)
        weights = weights.view(weights.shape[0], 1, weights.shape[1])
        loss_val_mse, loss_val_kl = loss(predictions[:, :, :-padding], weights[:, :, :-padding], variational)
        # during warmup only train mse loss
        if warmup or not variational:
            loss_val = loss_val_mse
        else:
            loss_val = loss_val_mse + loss_val_kl
        # print(loss_val.item())
        loss_val.backward()
        optimizer.step()
        
        epoch_loss_val_mse += loss_val_mse.item()
        epoch_loss_val_kl += loss_val_kl.item()
        break
            

    return epoch_loss_val_mse / len(iterator), epoch_loss_val_kl / len(iterator)

def evaluate(model, iterator, loss, device, variational, remove_indx):
    epoch_loss_val_mse = 0
    epoch_loss_val_kl = 0

    if remove_indx:
        padding = 21
    else:
        padding = 31
    
    model.eval()
    for i, batch in enumerate(iterator): # batch is simply a batch of ci-matricies as a tensor as x and y are the same 
        # attention_mask, base_ids are already on device
        weights, weights_prev, weights_prev = batch
        weights = torch.nn.functional.pad(weights, (0, padding)).to(device)

        predictions = model(weights)
        weights = weights.view(weights.shape[0], 1, weights.shape[1])
        loss_val_mse, loss_val_kl = loss(predictions[:, :, :-padding], weights[:, :, :-padding], variational)
        if not variational:
            loss_val = loss_val_mse
        else:
            loss_val = loss_val_mse + loss_val_kl
  
        epoch_loss_val_mse += loss_val_mse.item()
        epoch_loss_val_kl += loss_val_kl.item()
        break

    return epoch_loss_val_mse / len(iterator), epoch_loss_val_kl / len(iterator)