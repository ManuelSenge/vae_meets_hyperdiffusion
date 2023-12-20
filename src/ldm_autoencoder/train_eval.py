
import torch

def train(model, iterator, optimizer, loss, device, warmup, variational, normalizing_constant=1):
    epoch_loss_val_mse = 0
    epoch_loss_val_kl = 0
    
    model.train()
    for i, batch in enumerate(iterator): # batch is simply a batch of ci-matricies as a tensor as x and y are the same 
        # attention_mask, base_ids are already on device
        weights, weights_prev, weights_prev = batch
        weights = weights.view(weights.shape[0], 1, weights.shape[1])
        weights = torch.nn.functional.pad(weights, (0, 31)).to(device)
        optimizer.zero_grad() # clear gradients first

        predictions, posterior = model(weights)

        loss_val_mse, loss_val_kl = loss(predictions[:, :, :-31], weights[:, :, :-31], variational)
        # during warmup only train mse loss
        if warmup or not variational:
            loss_val = loss_val_mse
        else:
            loss_val = loss_val_mse + loss_val_kl
            
        loss_val.backward()
        optimizer.step()
        
        epoch_loss_val_mse += loss_val_mse.item()
        epoch_loss_val_kl += loss_val_kl.item()
            

    return epoch_loss_val_mse / (len(iterator) * normalizing_constant), epoch_loss_val_kl / len(iterator)

def evaluate(model, iterator, loss, device, variational, normalizing_constant=1):
    epoch_loss_val_mse = 0
    epoch_loss_val_kl = 0
    
    model.eval()
    for i, batch in enumerate(iterator): # batch is simply a batch of ci-matricies as a tensor as x and y are the same 
        # attention_mask, base_ids are already on device
        weights, weights_prev, weights_prev = batch
        weights = weights.view(weights.shape[0], 1, weights.shape[1])
        weights = torch.nn.functional.pad(weights, (0, 31)).to(device)

        predictions, posterior = model(weights)

        loss_val_mse, loss_val_kl = loss(predictions[:, :, :-31], weights[:, :, :-31], variational)
        if not variational:
            loss_val = loss_val_mse
        else:
            loss_val = loss_val_mse + loss_val_kl
  
        epoch_loss_val_mse += loss_val_mse.item()
        epoch_loss_val_kl += loss_val_kl.item()

    return epoch_loss_val_mse / (len(iterator) * normalizing_constant), epoch_loss_val_kl / len(iterator)