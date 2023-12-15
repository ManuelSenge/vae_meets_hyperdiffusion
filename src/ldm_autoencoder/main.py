from model.ldm.modules.autoencoder import AutoencoderKL
from omegaconf import OmegaConf
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
from model.ldm.modules.autoencoder import AutoencoderKL
from loss import VAELoss

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
    BS = 32
    SEED = 1234
    N_EPOCHS = 100
    warmup_epochs = 10
    normalize = True
    learning_rate = 0.0002
    device = "auto"
    log_wandb = 1
    variational = False # True to make VAE variational False to make it an AE
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    run_params_name = f'ldm_latent_1149_attention_[1149]_dropout_0.2_lr_0.0002_num_res_3_normalized_[1,4,8,16,32,64]' #f'bn_lr{learning_rate}_E{attention_encoder}_num_att_layers{num_att_layers}_enc_chans{wandb_enc_channels}_enc_kernel_sizes{wandb_enc_kernel_sizes}_warmup_epochs{warmup_epochs}'

    output_file = f'{run_params_name}_{SEED}'
    checkpoint_path = output_dir
    
    random.seed(SEED)
    np.random.seed(SEED)


    if log_wandb:
        project = "LDM"

        wandb.init( project=project,
                    entity="adl-cv",
                    name=f'{run_params_name}',
                    config={
                    "learning_rate": learning_rate,
                    "batch_size": BS,
                    "SEED": SEED,
                    "epochs": N_EPOCHS,
                    "normalize": True,
                    "ch_mult": [ 1,4,8,16,32,64],
                    "dropout": 0.2,
                    "num_res_block": 3,
                    "latent": 1149
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
 
    oai_coeff = None
    if normalize:
        oai_coeff_dataset = WeightDataset(
            mlps_folder_train,
            None, 
            0,
            mlp_kwargs,
            cfg=cfg
        )

        nets = [oai_coeff_dataset[i][0] for i in range(len(oai_coeff_dataset.mlp_files))]
        
        stdev = torch.stack(nets).flatten().std(unbiased=True).item()
        oai_coeff = 0.538 / stdev # openai coefficient according to G.pt
        print("OAI-Coeff Normalization: ", oai_coeff)
    normalizing_constant = oai_coeff if oai_coeff else 1

    print("OAI-Coeff Normalization: ", normalizing_constant)

    # create dataset and dataloader
    print('create dataset...')
    train_dt = WeightDataset(
        mlps_folder_train,
        None,
        0, # model.dims hardcoded in Transformer
        mlp_kwargs,
        cfg,
        train_object_names,
        normalize=normalizing_constant
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
        normalize=normalizing_constant,
    )

    val_dl = DataLoader(
        val_dt,
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
        normalize=normalizing_constant
    )

    test_dl = DataLoader(
        test_dt,
        batch_size=Config.get("batch_size"),
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    # create model loss and optimizer
    #model = VariationalAutoencoder(latent_dims=512, device=device)
    print('create model..')
    config_model = OmegaConf.load('./model/autoencoder_kl_8x8x64.yaml')
    loss_config = config_model.model.params.lossconfig
    ddconfig = config_model.model.params.ddconfig
    model = AutoencoderKL(ddconfig=ddconfig, lossconfig=loss_config, embed_dim=1149)
    # loss = model.loss

    loss = VAELoss(autoencoder=None)

    # print(summary(model, (1, 36768), device="cpu"))

    model = model.to(device)
    print(f'model created..')

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
        train_mse_loss, train_kl_loss = train(model, train_dl, optimizer, loss, device, epoch <= warmup_epochs, variational=variational, normalizing_constant=normalizing_constant)
        val_mse_loss, val_kl_loss = evaluate(model, val_dl, loss, device, variational=variational, normalizing_constant=normalizing_constant)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if log_wandb:
            wandb.log({"train/mse_loss": train_mse_loss, "train/kl_loss": train_kl_loss,\
                "val/mse_loss": val_mse_loss, "val/kl_loss":val_kl_loss})

        if val_mse_loss+val_kl_loss < best_valid_loss:
            best_valid_loss = val_mse_loss+val_kl_loss
            torch.save(model.state_dict(), f"{output_dir}/{output_file}.pt")
            if log_wandb:
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
    test_mse_loss, test_kl_loss = evaluate(model, test_dl, loss, device, variational=variational, normalizing_constant=normalizing_constant)

    if log_wandb:
        wandb.log({"test/mse_loss": test_mse_loss, "test/kl_loss":test_kl_loss})

    print(f'Test MSE Loss: {test_mse_loss:.3f} Test KL Loss: {test_kl_loss:.3f}')
    f = open(f"{output_dir}/{output_file}.txt", "a")
    f.write(f'Test MSE Loss: {test_mse_loss:.3f} Test KL Loss: {test_kl_loss:.3f}\n')
    f.close()


if __name__ == "__main__":
    # load param
    main()