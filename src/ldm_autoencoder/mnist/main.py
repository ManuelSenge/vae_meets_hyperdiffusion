from dataset import MNISTDataset
from train_eval import train, evaluate

import sys
sys.path.append('../')
from model.ldm.modules.autoencoder import AutoencoderKL
from omegaconf import OmegaConf
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import time
from utils import  epoch_time

import os
import argparse
import wandb
import random
import numpy as np
import sys
sys.path.append('../../')
from model.ldm.modules.autoencoder import AutoencoderKL
from loss import VAELoss
from functools import partial
from enum import Enum
from pytorch_lightning.loggers import WandbLogger
# need this seed for the lookup (as data is randomly shuffled)
random.seed(1234)
np.random.seed(1234)

def generate_during_training(wandb_logger, model, epoch, num_imgs, samples, device):
    if samples is None:
        for i in range(num_imgs):
            latent = model.sample2D(1)
            encoded = model.decode(latent)
            w_img = wandb.Image(encoded, caption="generated sample")
            wandb.log({"randomly generated": w_img})
    else:
        for s in samples:
            encoded, post = model(s[1].view(1, 1, 28, 28).to(device))
            encoded = encoded.detach().cpu()
            true_img = wandb.Image(s[1], caption="generated sample")
            pred_img = wandb.Image(encoded, caption="generated sample")
            wandb.log({"true": true_img})
            wandb.log({"encoded": pred_img})


class ScheduleType(Enum):
    EXP_CAPPED = 1
    CYCLIC = 2
    NONE = 3

def main():
    output_dir = '/Users/manuelsenge/Documents/TUM/Semester_3/ADL4CV/workspace/HyperDiffusion/src/ldm_autoencoder/mnist/output/'

    # this is whats actually used
    print('create model..')
    config_model = OmegaConf.load('../model/autoencoder_kl_8x8x64.yaml')

    BS = 256
    SEED = 1234
    N_EPOCHS = 800
    warmup_epochs = 0
    normalize = True
    learning_rate = 0.0002
    min_lr = learning_rate / 100
    single_sample_overfit = False
    single_sample_overfit_index = 1
    beta = 10e-4

    scheduler = None    
    generate_every_n_epochs = 1
    generate_n_meshes = 2

    if scheduler == ScheduleType.EXP_CAPPED:
        scheduler_exp_mult = 0.95
        assert 0 < scheduler_exp_mult < 1
        lr_lambda = lambda epoch: max(scheduler_exp_mult** (epoch//40), min_lr / learning_rate)
        optim_scheduler = partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=lr_lambda)
        scheduler_string = f"exp_mult_{scheduler_exp_mult}"
    elif scheduler == ScheduleType.CYCLIC:
        min_lr = 0.0001
        max_lr = 0.0003
        step_size_up = 50
        optim_scheduler = partial(torch.optim.lr_scheduler.CyclicLR, base_lr=min_lr, max_lr=max_lr, step_size_up=step_size_up, cycle_momentum=False)
        scheduler_string = f"min_lr_{min_lr}_max_lr_{max_lr}_step_size_up_{step_size_up}"
    elif scheduler is None:
        lr_lambda = lambda _: 1
        optim_scheduler = partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=lr_lambda)
        scheduler_string = "none"
    else:
        raise ValueError(f"Scheduler value {scheduler} not recognized")
    
    device = "auto"
    log_wandb = 1

    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    ae_params = config_model.model.params.ddconfig
    attention_resolutions = ae_params.attn_resolutions
    dropout = ae_params.dropout
    num_res = ae_params.num_res_blocks
    ch_mult = ae_params.ch_mult
    embed_dim = 14

    run_params_name = f'ldm_latent_{embed_dim}_attention_{attention_resolutions}_dropout_{dropout}_lr_{learning_rate}_num_res_{num_res}_ch_mult_{ch_mult}_BS_{BS}'
    
    if normalize:
        run_params_name += '_normalized'

    if single_sample_overfit:
        run_params_name = f'single_sample_overfit_index_{single_sample_overfit_index}_' + run_params_name

    if scheduler is not None:
        run_params_name += f'schedule_{scheduler}'

    if warmup_epochs == 0:
        run_params_name += 'no_warmup_'
    else:
        run_params_name += f'warmup_{warmup_epochs}_'

    output_file = f'{run_params_name}_{SEED}'
    
    use_checkpoint = False

    checkpoint_path = f"{output_dir}/cp_{output_file}.pt"
    random.seed(SEED)
    np.random.seed(SEED)


    if log_wandb:
        project = "MNIST-LDM-VAE"

        wandb.init( project=project,
                    entity="adl-cv",
                    name=run_params_name,
                    config={
                    "learning_rate": learning_rate,
                    "batch_size": BS,
                    "SEED": SEED,
                    "epochs": N_EPOCHS,
                    "normalize": normalize,
                    "ch_mult": ch_mult,
                    "single_sample_overfit": single_sample_overfit,
                    "single_sample_overfit_index": single_sample_overfit_index if single_sample_overfit else -1,
                    "dropout": dropout,
                    "num_res_block": num_res,
                    "latent": embed_dim,
                    "attn_resolutions":  attention_resolutions,
                    "lr_scheduler": scheduler_string,
                    "use_checkpoint": use_checkpoint
                    })
        wandb_logger = WandbLogger()
    train_dataset = MNISTDataset(train=True, shuffle=True)
    val_dataset = MNISTDataset(train=False, shuffle=False)

    train_dl = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=BS, shuffle=False)
    
    samples = []
    for i in range(generate_n_meshes):
        samples.append(train_dataset.__getitem__(i))
   

    loss_config = config_model.model.params.lossconfig
    ddconfig = config_model.model.params.ddconfig

    model = AutoencoderKL(ddconfig=ddconfig, lossconfig=loss_config, embed_dim=embed_dim)
    variational = ddconfig['variational']
    # loss = model.loss

    loss = VAELoss(autoencoder=None)

    # print(summary(model, (1, 36768), device="cpu"))

    model = model.to(device)
    print(f'model created..')


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = optim_scheduler(optimizer)

    # load checkpoint if necessary
    
    if use_checkpoint:
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Could not find checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        epoch = checkpoint['epoch']
        train_mse_loss = checkpoint['train_mse_loss']
        train_kl_loss = checkpoint['train_kl_loss']
        val_mse_loss = checkpoint['val_mse_loss']
        val_kl_loss = checkpoint['val_kl_loss']
        if log_wandb:
            wandb.log({ "train/mse_loss": train_mse_loss,
                        "train/kl_loss": train_kl_loss,
                        "val/mse_loss": val_mse_loss,
                        "val/kl_loss":val_kl_loss, 
                        "lr": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,}
            )

        best_val_loss = checkpoint['best_val_loss']
        start_epoch = epoch + 1
    else:
        start_epoch = 0
        best_val_loss = 100000
    
    
    print(f'start training..')

    for epoch in range(start_epoch, N_EPOCHS):
        start_time = time.time()
        train_mse_loss, train_kl_loss, posterior = train(model, train_dl, optimizer, loss, device, epoch < warmup_epochs, beta)
        val_mse_loss, val_kl_loss = evaluate(model, val_dl, loss, device)
        if log_wandb and epoch%generate_every_n_epochs==0:
            generate_during_training(wandb_logger, model, epoch, generate_n_meshes, samples, device)
            generate_during_training(wandb_logger, model, epoch, generate_n_meshes, None, device)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if val_mse_loss+val_kl_loss < best_val_loss:
            best_val_loss = val_mse_loss+val_kl_loss
            torch.save(model.state_dict(), f"{output_dir}/{output_file}.pt")
            # if log_wandb:
            #     wandb.save(f"{output_dir}/{output_file}.pt", base_path='/hyperdiffusion')

        if log_wandb:
            wandb.log({"train/mse_loss": train_mse_loss,
                        "train/kl_loss": train_kl_loss,
                        "val/mse_loss": val_mse_loss,
                        "val/kl_loss":val_kl_loss, 
                        "lr": lr_scheduler.get_last_lr()[0],
                        'best_val_loss': best_val_loss,
                        "epoch": epoch,}
            )
            

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s | LR: {lr_scheduler.get_last_lr()[0]}')
        print(f'\t Train MSE Loss: {train_mse_loss:.3f} Train KL Loss: {train_kl_loss:.3f}')
        print(f'\t Val. MSE Loss: {val_mse_loss:.3f} Val. KL Loss: {val_kl_loss:.3f}')

        f = open(f"{output_dir}/{output_file}.txt", "a")
        f.write(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n | LR: {lr_scheduler.get_last_lr()[0]}')
        f.write(f'\tTrain MSE Loss: {train_mse_loss:.3f} Train KL Loss: {train_kl_loss:.3f}\n')
        f.write(f'\tVal. MSE Loss: {val_mse_loss:.3f} Val. KL Loss: {val_kl_loss:.3f}\n')
        f.close()
        

        # save model and optimizer every 20 epochs
        if epoch % 20 == 0:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'train_mse_loss': train_mse_loss,
                        'train_kl_loss': train_kl_loss,  
                        'val_mse_loss': val_mse_loss,
                        'val_kl_loss': val_kl_loss,
                        'best_val_loss':best_val_loss,
                        }, checkpoint_path)
            
        lr_scheduler.step()


if __name__ == "__main__":
    # load param
    main()