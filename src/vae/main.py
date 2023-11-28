# from model_linear import VariationalAutoencoder
from model_transformer_conv import VariationalAutoencoder
from loss import VAELoss
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import time
from utils import epoch_time
from train_eval import train, evaluate
import os
import argparse
import wandb
import random
import numpy as np
import sys

sys.path.append("../")
from dataset import WeightDataset
from hd_utils import Config
from omegaconf import DictConfig
import hydra
from torchsummary import summary
from annealing import frange_cycle_linear


SEED = 1234
# need this seed for the lookup (as data is randomly shuffled)
random.seed(SEED)
np.random.seed(SEED)


@hydra.main(
    version_base=None,
    config_path="./configs",
    config_name="train_plane_vae",
)
def main(cfg: DictConfig):
    Config.config = cfg

    attention_encoder = cfg.attention_layers_encoder
    attention_decoder = attention_encoder[::-1]
    N_EPOCHS = cfg.epochs
    beta = cfg.beta
    warmup_epochs = cfg.warmup_epochs

    learning_rate = cfg.lr
    enc_chans = cfg.enc_chans
    enc_kernel_sizes = cfg.enc_kernel_sizes
    num_att_layers = cfg.num_att_layers

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    attention_encoder_text = "_".join(
        ["1" if enc else "0" for enc in attention_encoder]
    )
    enc_channels_text = "_".join([str(enc) for enc in enc_chans])
    enc_kernel_sizes_text = "_".join([str(enc) for enc in enc_kernel_sizes])
    annealing_text = cfg.get("annealing", "no_annealing")

    variational = cfg.get("variational", True)

    run_params_name = f"bn_{annealing_text}_lr{learning_rate}_E{attention_encoder_text}_num_att_layers_{num_att_layers}_enc_chans_{enc_channels_text}_enc_kernel_sizes_{enc_kernel_sizes_text}_{variational}"

    output_file = f"{run_params_name}_{SEED}"

    random.seed(SEED)
    np.random.seed(SEED)

    if cfg.use_wandb:
        wandb.init(
            project="VAE" if variational else "AE",
            entity="adl-cv",
            name=f"first_test_{run_params_name}",
            group="conv_attention_vae",
            config=dict(cfg),
        )

    dataset_path = os.path.join(Config.config["dataset_dir"], Config.config["dataset"])

    mlps_folder_train = Config.get("mlps_folder_train")
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
    print("create dataset...")
    train_dt = WeightDataset(
        mlps_folder_train,
        None,
        0,  # model.dims hardcoded in Transformer
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
    )

    test_dl = DataLoader(
        test_dt,
        batch_size=Config.get("batch_size"),
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    iter_per_epoch = len(train_dl)

    total_annealing_iterations = iter_per_epoch * (N_EPOCHS - warmup_epochs)

    betas = None
    if annealing := cfg.get("annealing"):
        if annealing == "cycle_linear":
            n_cycle = cfg["cycle_linear"].get("n_cycle")
            ratio = cfg["cycle_linear"].get("ratio")
            betas = frange_cycle_linear(
                total_annealing_iterations, stop=beta, n_cycle=n_cycle, ratio=ratio
            )

    if betas is None:
        betas = [beta] * total_annealing_iterations

    # create model loss and optimizer
    # model = VariationalAutoencoder(latent_dims=512, device=device)
    print("create model..")
    model = VariationalAutoencoder(
        input_dim=36737,
        latent_dims=512,
        device=device,
        enc_chans=enc_chans,
        enc_kernal_sizes=enc_kernel_sizes,
        self_attention_encoder=attention_encoder,
        self_attention_decoder=attention_decoder,
        num_att_layers=num_att_layers,
        variational=variational,
    )
    # device = 'cpu'
    # print(summary(model, (1, 36737), device="cpu"))

    model = model.to(device)
    print(f"model created..")

    loss = VAELoss(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # load checkpoint if necessary
    """checkpoint_path = f"{output_dir}/model_dict{SEED}.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_loss = checkpoint['loss']
        best_valid_loss = checkpoint['best_valid_loss']
    else:"""
    start_epoch = 0
    best_valid_loss = 100000

    print(f"start training..")
    for epoch in range(start_epoch, N_EPOCHS):
        start_time = time.time()
        if epoch >= warmup_epochs:
            if epoch == warmup_epochs:
                print("Warmup over.")
            epoch_betas = betas[
                (epoch - warmup_epochs)
                * iter_per_epoch : (epoch - warmup_epochs + 1)
                * iter_per_epoch
            ]
            warmup = False
        else:
            epoch_betas = None
            warmup = True

        train_mse_loss, train_kl_loss = train(
            model,
            train_dl,
            optimizer,
            loss,
            warmup,
            epoch_betas,
            device,
            variational=variational,
        )
        val_mse_loss, val_kl_loss = evaluate(
            model, val_dl, loss, device, variational=variational
        )
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if cfg.use_wandb:
            wandb.log(
                {
                    "train/mse_loss": train_mse_loss,
                    "train/kl_loss": train_kl_loss,
                    "val/mse_loss": val_mse_loss,
                    "val/kl_loss": val_kl_loss,
                }
            )

        if val_mse_loss + val_kl_loss < best_valid_loss:
            best_valid_loss = val_mse_loss + val_kl_loss
            torch.save(
                model.state_dict(), f"{cfg.best_model_save_path}/{output_file}.pt"
            )

        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(
            f"\t Train MSE Loss: {train_mse_loss:.3f} Train KL Loss: {train_kl_loss:.3f}"
        )
        print(f"\t Val. MSE Loss: {val_mse_loss:.3f} Val. KL Loss: {val_kl_loss:.3f}")

        f = open(f"{cfg.best_model_save_path}/{output_file}.txt", "a")
        f.write(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n")
        f.write(
            f"\tTrain MSE Loss: {train_mse_loss:.3f} Train KL Loss: {train_kl_loss:.3f}\n"
        )
        f.write(
            f"\tVal. MSE Loss: {val_mse_loss:.3f} Val. KL Loss: {val_kl_loss:.3f}\n"
        )
        f.close()

        # save model and optimizer
        """torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    'best_valid_loss':best_valid_loss,
                    }, checkpoint_path)"""

    # load the best model and test it on the test set
    model.load_state_dict(torch.load(f"{cfg.best_model_save_path}/{output_file}.pt"))
    test_mse_loss, test_kl_loss = evaluate(
        model, test_dl, loss, device, variational=variational
    )

    if cfg.use_wandb:
        wandb.log({"test/mse_loss": test_mse_loss, "test/kl_loss": test_kl_loss})

    print(f"Test MSE Loss: {test_mse_loss:.3f} Test KL Loss: {test_kl_loss:.3f}")
    f = open(f"{cfg.best_model_save_path}/{output_file}.txt", "a")
    f.write(f"Test MSE Loss: {test_mse_loss:.3f} Test KL Loss: {test_kl_loss:.3f}\n")
    f.close()


if __name__ == "__main__":
    # load param
    main()
