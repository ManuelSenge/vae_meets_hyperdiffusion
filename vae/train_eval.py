
def train(model, iterator, optimizer, loss, device):
    epoch_loss = 0
    
    model.train()
    for i, batch in enumerate(iterator): # batch is simply a batch of ci-matricies as a tensor as x and y are the same 
        # attention_mask, base_ids are already on device
        weights, weights_prev, weights_prev = batch
        weights = weights.to(device)
        optimizer.zero_grad() # clear gradients first

        predictions = model(weights)

        loss_val = loss(predictions, weights)
    
        loss_val.backward()
        optimizer.step()
        print(loss_val.item())
        epoch_loss += loss_val.item()
            

    return epoch_loss / len(iterator)

def evaluate(model, iterator, loss, device):
    epoch_loss = 0
    
    model.eval()
    for i, batch in enumerate(iterator): # batch is simply a batch of ci-matricies as a tensor as x and y are the same 
        # attention_mask, base_ids are already on device
        weights, weights_prev, weights_prev = batch
        weights = weights.to(device)

        predictions = model(weights)

        loss_val = loss(predictions, weights)
  
        epoch_loss += loss_val.item()

    return epoch_loss / len(iterator)