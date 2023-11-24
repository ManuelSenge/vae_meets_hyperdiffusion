
def train(model, iterator, optimizer, loss, device, warmup):
    epoch_loss_val_mse = 0
    epoch_loss_val_kl = 0
    
    model.train()
    for i, batch in enumerate(iterator): # batch is simply a batch of ci-matricies as a tensor as x and y are the same 
        # attention_mask, base_ids are already on device
        weights, weights_prev, weights_prev = batch
        weights = weights.to(device)
        optimizer.zero_grad() # clear gradients first

        predictions = model(weights)

        loss_val_mse, loss_val_kl = loss(predictions, weights)
        # during warmup only train mse loss
        if warmup:
            loss_val = loss_val_mse
        else:
            loss_val = loss_val_mse + loss_val_kl
            
        loss_val.backward()
        optimizer.step()
        epoch_loss_val_mse += loss_val_mse.item()
        epoch_loss_val_kl += loss_val_kl.item()
            

    return epoch_loss_val_mse / len(iterator), epoch_loss_val_kl / len(iterator)

def evaluate(model, iterator, loss, device):
    epoch_loss_val_mse = 0
    epoch_loss_val_kl = 0
    
    model.eval()
    for i, batch in enumerate(iterator): # batch is simply a batch of ci-matricies as a tensor as x and y are the same 
        # attention_mask, base_ids are already on device
        weights, weights_prev, weights_prev = batch
        weights = weights.to(device)

        predictions = model(weights)

        loss_val_mse, loss_val_kl = loss(predictions, weights)
        loss_val = loss_val_mse + loss_val_kl
  
        epoch_loss_val_mse += loss_val_mse.item()
        epoch_loss_val_kl += loss_val_kl.item()

    return epoch_loss_val_mse / len(iterator), epoch_loss_val_kl / len(iterator)