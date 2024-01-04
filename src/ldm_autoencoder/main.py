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
from sampler import SingleInstanceBatchSampler
from hd_utils import Config
from omegaconf import DictConfig
import hydra
from torchsummary import summary
from model.ldm.modules.autoencoder import AutoencoderKL
from loss import VAELoss
from functools import partial
from enum import Enum

# need this seed for the lookup (as data is randomly shuffled)
random.seed(1234)
np.random.seed(1234)

class ScheduleType(Enum):
    EXP_CAPPED = 1
    CYCLIC = 2
    NONE = 3

@hydra.main(
    version_base=None,
    config_path="../configs/diffusion_configs",
    config_name="train_plane",
)
def main(cfg: DictConfig):
    Config.config = cfg
    cfg.filter_bad_path = '../' + cfg.filter_bad_path
    output_dir = '/hyperdiffusion/output_files'

    # this is whats actually used
    print('create model..')
    config_model = OmegaConf.load('./model/autoencoder_kl_8x8x64.yaml')

    BS = 32
    SEED = 1234
    N_EPOCHS = 800
    warmup_epochs = 10
    normalize = True
    learning_rate = 0.0002
    min_lr = learning_rate / 100
    single_sample_overfit = False
    single_sample_overfit_index = 1

    scheduler = ScheduleType.CYCLIC

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
    variational = False # True to make VAE variational False to make it an AE
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

    run_params_name = f'ldm_latent_1149_attention_{attention_resolutions}_dropout_{dropout}_lr_{learning_rate}_num_res_{num_res}_ch_mult_{ch_mult}' #f'bn_lr{learning_rate}_E{attention_encoder}_num_att_layers{num_att_layers}_enc_chans{wandb_enc_channels}_enc_kernel_sizes{wandb_enc_kernel_sizes}_warmup_epochs{warmup_epochs}'
    
    if normalize:
        run_params_name += '_normalized'

    if single_sample_overfit:
        run_params_name = f'single_sample_overfit_index_{single_sample_overfit_index}_' + run_params_name

    if scheduler is not None:
        run_params_name += f'schedule_{scheduler}'

    output_file = f'{run_params_name}_{SEED}'
    
    use_checkpoint = False

    checkpoint_path = f"{output_dir}/cp_{output_file}.pt"
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
                    "normalize": normalize,
                    "ch_mult": ch_mult,
                    "single_sample_overfit": single_sample_overfit,
                    "single_sample_overfit_index": single_sample_overfit_index if single_sample_overfit else -1,
                    "dropout": dropout,
                    "num_res_block": num_res,
                    "latent": 1149,
                    "attn_resolutions":  attention_resolutions,
                    "lr_scheduler": scheduler_string,
                    "use_checkpoint": use_checkpoint
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
            cfg,
            train_object_names
        )

        nets = [oai_coeff_dataset[i][0] for i in range(len(oai_coeff_dataset.mlp_files))]
        
        stdev = torch.stack(nets).flatten().std(unbiased=True).item()
        oai_coeff = 0.538 / stdev # openai coefficient according to G.pt
        print("OAI-Coeff Normalization: ", oai_coeff)
    normalizing_constant = oai_coeff if oai_coeff else 1

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

    train_dl = None
    if single_sample_overfit:
        train_dl = DataLoader(
            train_dt,
            sampler = SingleInstanceBatchSampler(single_sample_overfit_index, len(train_dt)),
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

    loss_config = config_model.model.params.lossconfig
    ddconfig = config_model.model.params.ddconfig
    model = AutoencoderKL(ddconfig=ddconfig, lossconfig=loss_config, embed_dim=1149)
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
        train_mse_loss, train_kl_loss = train(model, train_dl, optimizer, loss, device, epoch <= warmup_epochs, variational=variational, normalizing_constant=normalizing_constant)
        val_mse_loss, val_kl_loss = evaluate(model, val_dl, loss, device, variational=variational, normalizing_constant=normalizing_constant)
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