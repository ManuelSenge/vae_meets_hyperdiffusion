#from model_linear import VariationalAutoencoder
from model_transformer_conv import VariationalAutoencoder
from loss import VAELoss
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import time
from utils import  epoch_time
from train_eval import train, evaluate
import os
import argparse
import wandb
import random
import numpy as np
import sys
sys.path.append('../')
from dataset import WeightDataset
from hd_utils import Config
from omegaconf import DictConfig
import hydra
from torchsummary import summary

# need this seed for the lookup (as data is randomly shuffled)
random.seed(1234)
np.random.seed(1234)


@hydra.main(
    version_base=None,
    config_path="../configs/diffusion_configs",
    config_name="train_plane",
)
def main(cfg: DictConfig):
    Config.config = cfg
    cfg.filter_bad_path = '../' + cfg.filter_bad_path
    print(cfg.filter_bad_path)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    device = 'cpu'
        
    output_dir = './outputs/'
    SEED = 1234
    output_file = 'output_'
    checkpoint_path = output_dir
    N_EPOCHS = 30
    learning_rate = 0.0001
    BS = 32
    lob_wandb = False
    random.seed(SEED)
    np.random.seed(SEED)

    if lob_wandb:
        wandb.init( project="VAE",
                    entity="adl-cv",
                    name=f'first_test_lr{learning_rate}',
                    group='first_test',
                    config={
                    "learning_rate": learning_rate,
                    "batch_size": BS,
                    "SEED": SEED,
                    "epochs": N_EPOCHS,
                    })

    dataset_path = os.path.join('..', Config.config["dataset_dir"], Config.config["dataset"])

    mlps_folder_train = '../' + Config.get("mlps_folder_train")
    mlp_kwargs = Config.config["mlp_config"]["params"]

    train_object_names = np.genfromtxt(
        os.path.join(dataset_path, "train_split.lst"), dtype="str"
    )
    train_object_names = set([str.split(".")[0] for str in train_object_names])

    val_object_names = np.genfromtxt(
            os.path.join(dataset_path, "val_split.lst"), dtype="str"
    )
    val_object_names = set([str.split(".")[0] for str in val_object_names])
    test_object_names = np.genfromtxt(
        os.path.join(dataset_path, "test_split.lst"), dtype="str"
    )
    test_object_names = set([str.split(".")[0] for str in test_object_names])
 

    # create dataset and dataloader
    print('create dataset...')
    train_dt = WeightDataset(
        mlps_folder_train,
        None,
        0, # model.dims hardcoded in Transformer
        mlp_kwargs,
        cfg,
        train_object_names,
    )

    train_dl = DataLoader(
        train_dt,
        batch_size=Config.get("batch_size"),
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    val_dt = WeightDataset(
        mlps_folder_train,
        None,
        0,
        mlp_kwargs,
        cfg,
        val_object_names,
    )

    val_dl = DataLoader(
        train_dt,
        batch_size=Config.get("batch_size"),
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    test_dt = WeightDataset(
        mlps_folder_train,
        None,
        0,
        mlp_kwargs,
        cfg,
        test_object_names,
    )

    test_dl = DataLoader(
        train_dt,
        batch_size=Config.get("batch_size"),
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    # create model loss and optimizer
    #model = VariationalAutoencoder(latent_dims=512, device=device)
    print('create model..')
    model = VariationalAutoencoder(input_dim=36737,
                                   latent_dims=512,
                                   device=device,
                                   enc_chans=[64, 32, 16, 1],
                                   enc_kernal_sizes=[8, 6, 3, 3],
                                   self_attention_encoder=True,
                                   self_attention_decoder=True)

    #print(summary(model, (1, 36737), device="cpu"))

    model = model.to(device)
    print(f'model created..')

    loss = VAELoss(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # load checkpoint if necessary
    '''checkpoint_path = f"{output_dir}/model_dict{SEED}.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_loss = checkpoint['loss']
        best_valid_loss = checkpoint['best_valid_loss']
    else:'''
    start_epoch = 0
    best_valid_loss = 100000
    
    
    print(f'start training..')

    for epoch in range(start_epoch, N_EPOCHS):
        start_time = time.time()
        train_loss = train(model, train_dl, optimizer, loss, device)
        valid_loss = evaluate(model, val_dl, loss, device)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if wandb:
            wandb.log({"train/loss": train_loss, "val/loss": valid_loss})

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"{output_dir}/{output_file}{SEED}.pt")

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')

        f = open(f"{output_dir}/{output_file}{SEED}.txt", "a")
        f.write(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n')
        f.write(f'\tTrain Loss: {train_loss:.3f}\n')
        f.write(f'\tVal. Loss: {valid_loss:.3f}\n')
        f.close()
        
        # save model and optimizer
        '''torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    'best_valid_loss':best_valid_loss,
                    }, checkpoint_path)'''

    # load the best model and test it on the test set
    model.load_state_dict(torch.load(f"{output_dir}/{output_file}{SEED}.pt"))
    test_loss = evaluate(model, test_dl, loss, device)

    if wandb:
        wandb.log({"test/loss": test_loss})

    print(f'Test Loss: {test_loss:.3f}')
    f = open(f"{output_dir}/{output_file}{SEED}.txt", "a")
    f.write(f'Test Loss: {test_loss:.3f}\n')
    f.close()


if __name__ == "__main__":
    main()