import torch
import trimesh
import os
import copy
import wandb
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import hydra

import sys
sys.path.append('../')
import tqdm
from hd_utils import generate_mlp_from_weights, render_mesh
from model_transformer_conv import VariationalAutoencoder
from siren import sdf_meshing
from siren.experiment_scripts.test_sdf import SDFDecoder
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from dataset import WeightDataset
from hd_utils import Config
from omegaconf import DictConfig

'''
    model_type: mlp_3d
    out_size: 1
    hidden_neurons:
      - 128
      - 128
      - 128
    output_type: occ
    out_act: sigmoid
    multires: 4
    use_leaky_relu: False
    move: False'''

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


mlp_kwargs = OmegaConf.create({'model_type':'mlp_3d', 'out_size':1, 'hidden_neurons':[128, 128, 128], 'output_type':'occ', 'out_act':'sigmoid', 'multires':4, \
    'use_leaky_relu':False, 'move':False, 'device':'mps'})

i = 0
mesh_dir = "gen_meshes/VAE"

def generate_meshes(x_0s, folder_name="meshes", info="0", res=64, level=0):
        x_0s = x_0s.view(len(x_0s), -1)
        curr_weights = None
        x_0s = x_0s[:, :curr_weights]
        meshes = []
        sdfs = []
        for i, weights in enumerate(x_0s):
            mlp = generate_mlp_from_weights(weights, mlp_kwargs)
            sdf_decoder = SDFDecoder(
                mlp_kwargs.model_type,
                None,
                "nerf" if mlp_kwargs.model_type == "nerf" else "mlp",
                mlp_kwargs
            )
            sdf_decoder.model = mlp.to(device).eval()
            with torch.no_grad():
                effective_file_name = (
                    f"{folder_name}/mesh_epoch_{i}_{info}"
                    if folder_name is not None
                    else None
                )
                
            v, f, sdf = sdf_meshing.create_mesh(
                sdf_decoder,
                effective_file_name,
                N=res,
                level=level
                if mlp_kwargs.output_type in ["occ", "logits"]
                else 0,
            )
            if (
                "occ" in mlp_kwargs.output_type
                or "logits" in mlp_kwargs.output_type
            ):
                tmp = copy.deepcopy(f[:, 1])
                f[:, 1] = f[:, 2]
                f[:, 2] = tmp
            sdfs.append(sdf)
            mesh = trimesh.Trimesh(v, f)
            meshes.append(mesh)
        sdfs = torch.stack(sdfs)
        return meshes, sdfs

def generate_images_from_VAE(x_0):
    out_imgs = []
    if not os.path.isdir(mesh_dir):
        os.makedirs(f"gen_meshes/VAE")
    mesh, _ = generate_meshes(x_0.unsqueeze(0), None, res=700)
    mesh = mesh[0]
    mesh.vertices *= 2
    mesh.export(f"gen_meshes/VAE/mesh_{len(out_imgs)}.obj")

    # Scaling the chairs down so that they fit in the camera
    #if cfg.dataset == "03001627":
    #    mesh.vertices *= 0.7
    img, _ = render_mesh(mesh)
    out_imgs.append(img)
    return out_imgs


@hydra.main(
    version_base=None,
    config_path="../configs/diffusion_configs",
    config_name="train_plane",
)
def main(cfg: DictConfig):
    Config.config = cfg
    cfg.filter_bad_path = '../' + cfg.filter_bad_path
    num_imgs = 3
    model_path = '/Users/manuelsenge/Documents/TUM/Semester_3/ADL4CV/workspace/HyperDiffusion/vae/output_files/autoencoder_bn_False_lr0.0002_E0_0_1_1_num_att_layers_1_enc_chans_64_32_16_1_enc_kernel_sizes_8_6_3_3_False_1234.pt'
    attention_encoder = "0011"
    attention_decoder = attention_encoder[::-1]
    log_wandb = 1
    num_att_layers = 1
    enc_chans = [64, 32, 16, 1]
    enc_kernel_sizes = [8, 6, 3, 3]
    wandb_enc_channels = "_".join([str(enc) for enc in enc_chans])
    
    model = VariationalAutoencoder(input_dim=36737,
                                   latent_dims=512,
                                   device=device,
                                   enc_chans=enc_chans,
                                   enc_kernal_sizes=enc_kernel_sizes,
                                   self_attention_encoder=[int(elem) for elem in list(attention_encoder)],
                                   self_attention_decoder=[int(elem) for elem in list(attention_decoder)],
                                   num_att_layers=num_att_layers,
                                   variational=False)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    dataset_path = os.path.join('..', Config.config["dataset_dir"], Config.config["dataset"])
    test_object_names = np.genfromtxt(
        os.path.join(dataset_path, "test_split.lst"), dtype="str"
    )
    test_object_names = set([str.split(".")[0] for str in test_object_names])
    mlps_folder_train = '../' + Config.get("mlps_folder_train")

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
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    count = 0 

    if log_wandb:
        wandb.init(
                entity='adl-cv',
                project="AE eval",
                name=f"batch_norm_{attention_encoder}_enc_chans{wandb_enc_channels}",
                config={'attention_encoder': attention_encoder, 'num_imgs':num_imgs},
            )

        wandb_logger = WandbLogger()
        
    while count < num_imgs:
        sample, _, _ = test_dt.__getitem__(count)
        sample = sample.to(device)
        enc_sample = model(sample)
        true_img = generate_images_from_VAE(sample)
        pred_img = generate_images_from_VAE(enc_sample)
        if log_wandb:
            wandb_logger.log_image(
                    "true_renders", true_img, step=count
                )
            wandb_logger.log_image(
                    "generated_renders", pred_img, step=count
                )
        count += 1



if __name__ == "__main__":
    main()