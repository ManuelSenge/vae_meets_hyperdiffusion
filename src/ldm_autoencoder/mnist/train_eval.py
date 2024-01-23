
import torch
import time
import torch.nn.functional as F

def train(model, iterator, optimizer, loss, device, warmup, beta, variational, conv_2d=False):
    epoch_loss_val_mse = 0
    epoch_loss_val_kl = 0
    
    model.train()
    for i, batch in enumerate(iterator): # batch is simply a batch of ci-matricies as a tensor as x and y are the same 
        start = time.time()
        # attention_mask, base_ids are already on device
        Y, X = batch
        
        Y = Y.to(device)
        X = X.to(device)

        if not conv_2d:
            X = X.view(-1, 1, 28*28)


        optimizer.zero_grad() # clear gradients first

        predictions, posteriors = model(X, sample_posterior=variational)

        loss_val_mse = loss(F.sigmoid(predictions.view(-1, 28*28)), X.view(-1, 28*28))
        loss_val_kl = posteriors.kl()
        loss_val_kl = torch.sum(loss_val_kl) / loss_val_kl.shape[0]


        # during warmup only train mse loss
        if warmup or not variational:
            loss_val = loss_val_mse
        else:
            loss_val = loss_val_mse + beta * loss_val_kl
            
        loss_val.backward()
        optimizer.step()
        
        epoch_loss_val_mse += loss_val_mse.item()
        epoch_loss_val_kl += loss_val_kl.item()
        end = time.time()
        
        
        

    return epoch_loss_val_mse / len(iterator), epoch_loss_val_kl / len(iterator)

def evaluate(model, iterator, loss, device, variational, conv_2d=False):
    epoch_loss_val_mse = 0
    epoch_loss_val_kl = 0
    
    model.eval()
    for i, batch in enumerate(iterator): # batch is simply a batch of ci-matricies as a tensor as x and y are the same 
        # attention_mask, base_ids are already on device
        Y, X = batch
        
        Y = Y.to(device)
        X = X.to(device)

        if not conv_2d:
            X = X.view(-1, 1, 28*28)

        predictions, posteriors = model(X)

        loss_val_mse = loss(F.sigmoid(predictions.view(-1, 28*28)), X.view(-1, 28*28))
        loss_val_kl = posteriors.kl()
        loss_val_kl = torch.sum(loss_val_kl) / loss_val_kl.shape[0]


        epoch_loss_val_mse += loss_val_mse.item()
        epoch_loss_val_kl += loss_val_kl.item()
        
        

    return epoch_loss_val_mse / len(iterator),  epoch_loss_val_kl / len(iterator)