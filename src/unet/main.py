from model.openaimodel import UNetModel

#from model_linear import VariationalAutoencoder
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
from sampler import SingleInstanceBatchSampler
from hd_utils import Config
from omegaconf import DictConfig
import hydra
from torchsummary import summary
import datetime

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
    
    output_dir = '/hyperdiffusion/output_files'
    attention_encoder = "0000"
    attention_decoder = attention_encoder[::-1]
    BS = 32
    SEED = 1234
    N_EPOCHS = 500
    warmup_epochs = 10
    
    variational = False

    learning_rate = 0.0002
    attention_resolutions  = []
    dropout = 0.0
    num_res_blocks = 3
    channel_mult = [1, 2, 4, 8]
    single_sample_overfit = True

    device = "auto"
    lob_wandb = 1
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')


    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    run_params_name = f'{date_str}_unet_4593__lr_{learning_rate}_attention_res_{str(attention_resolutions)}_dropout_{dropout}' #f'bn_lr{learning_rate}_E{attention_encoder}_num_att_layers{num_att_layers}_enc_chans{wandb_enc_channels}_enc_kernel_sizes{wandb_enc_kernel_sizes}_warmup_epochs{warmup_epochs}'

    if single_sample_overfit:
        run_params_name = 'single_sample_overfit_' + run_params_name

    output_file = f'{run_params_name}_{SEED}'
    checkpoint_path = output_dir
    
    random.seed(SEED)
    np.random.seed(SEED)

    

    if lob_wandb:
        project = "UNet"

        wandb.init( project=project,
                    entity="adl-cv",
                    name=f'first_test_{run_params_name}',
                    group=f'first_test',
                    config={
                    "latent_attention": True,
                    "attention_resolutions": attention_resolutions,
                    "dropout": dropout,
                    "learning_rate": learning_rate,
                    "num_res_blocks":  num_res_blocks,
                    "batch_size": BS,
                    "SEED": SEED,
                    "epochs": N_EPOCHS,
                    "channel_mult": channel_mult,
                    "single_sample_overfit": single_sample_overfit,
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

    train_dl = None
    if single_sample_overfit:
        train_dl = DataLoader(
            train_dt,
            sampler = SingleInstanceBatchSampler(len(train_dt)),
            batch_size=BS,
            num_workers=1,
            pin_memory=True
        )
    else: 
        train_dl = DataLoader(
            train_dt,
            batch_size=Config.get("batch_size"),
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )

    assert train_dl is not None

    val_dt = WeightDataset(
        mlps_folder_train,
        None,
        0,
        mlp_kwargs,
        cfg,
        val_object_names,
        
    )

    val_dl = DataLoader(
        val_dt,
        batch_size=Config.get("batch_size"),
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        # drop_last=True
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
        test_dt,
        batch_size=Config.get("batch_size"),
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    # create model loss and optimizer
    #model = VariationalAutoencoder(latent_dims=512, device=device)
    print('create model..')
    model = UNetModel(image_size=36744, 
                    in_channels=1, 
                    model_channels=1, 
                    out_channels=1, 
                    num_res_blocks=num_res_blocks,
                    attention_resolutions=attention_resolutions,
                    dropout=dropout,
                    channel_mult=channel_mult,
                    num_heads=1,
                    dims=1,
                    num_head_channels=-1)
    #device = 'cpu'
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
        train_mse_loss, train_kl_loss = train(model, train_dl, optimizer, loss, device, epoch <= warmup_epochs, variational=variational)
        val_mse_loss, val_kl_loss = evaluate(model, val_dl, loss, device, variational=variational)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if lob_wandb:
            wandb.log({"train/mse_loss": train_mse_loss, "train/kl_loss": train_kl_loss,\
                "val/mse_loss": val_mse_loss, "val/kl_loss":val_kl_loss})

        if val_mse_loss+val_kl_loss < best_valid_loss:
            best_valid_loss = val_mse_loss+val_kl_loss
            torch.save(model.state_dict(), f"{output_dir}/{output_file}.pt")

            if lob_wandb:
                wandb.save(f"{output_dir}/{output_file}.pt")

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\t Train MSE Loss: {train_mse_loss:.3f} Train KL Loss: {train_kl_loss:.3f}')
        print(f'\t Val. MSE Loss: {val_mse_loss:.3f} Val. KL Loss: {val_kl_loss:.3f}')

        f = open(f"{output_dir}/{output_file}.txt", "a")
        f.write(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n')
        f.write(f'\tTrain MSE Loss: {train_mse_loss:.3f} Train KL Loss: {train_kl_loss:.3f}\n')
        f.write(f'\tVal. MSE Loss: {val_mse_loss:.3f} Val. KL Loss: {val_kl_loss:.3f}\n')
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
    model.load_state_dict(torch.load(f"{output_dir}/{output_file}.pt"))
    test_mse_loss, test_kl_loss = evaluate(model, test_dl, loss, device, variational=variational)

    if lob_wandb:
        wandb.log({"test/mse_loss": test_mse_loss, "test/kl_loss":test_kl_loss})

    print(f'Test MSE Loss: {test_mse_loss:.3f} Test KL Loss: {test_kl_loss:.3f}')
    f = open(f"{output_dir}/{output_file}.txt", "a")
    f.write(f'Test MSE Loss: {test_mse_loss:.3f} Test KL Loss: {test_kl_loss:.3f}\n')
    f.close()


if __name__ == "__main__":
    # load param
    main()