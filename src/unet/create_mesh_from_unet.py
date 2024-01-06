import torch
import trimesh
import os
import copy
import wandb
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import hydra
from model.openaimodel import UNetModel
import sys
sys.path.append('../')
import tqdm
from hd_utils import generate_mlp_from_weights, render_mesh
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
mesh_dir = "gen_meshes/UNet"

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
        os.makedirs(f"gen_meshes/UNet")
    mesh, _ = generate_meshes(x_0.unsqueeze(0), None, res=700)
    mesh = mesh[0]
    mesh.vertices *= 2
    mesh.export(f"gen_meshes/UNet/mesh_{len(out_imgs)}.obj")

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
    model_path = '/Users/manuelsenge/Documents/TUM/Semester_3/ADL4CV/workspace/HyperDiffusion/our_unet/model/models/normalisation_2023-12-13_unet_2298__lr_0.0002_attention_res_8_16_dropout_0.2_1234.pt'
    log_wandb = 1
    
    model = UNetModel(image_size=36737+31, 
                    in_channels=1, 
                    model_channels=1, 
                    out_channels=1, 
                    num_res_blocks=3,
                    attention_resolutions=[8, 16],
                    dropout=0.0,
                    channel_mult= [1, 4, 8, 16, 32],
                    num_heads=1,
                    dims=1,
                    num_head_channels=-1)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    dataset_path = os.path.join('..', Config.config["dataset_dir"], Config.config["dataset"])
    test_object_names = np.genfromtxt(
        os.path.join(dataset_path, "test_split.lst"), dtype="str"
    )
    test_object_names = set([str.split(".")[0] for str in test_object_names])
    mlps_folder_train = '../' + Config.get("mlps_folder_train")

    train_object_names = np.genfromtxt(
        os.path.join(dataset_path, "train_split.lst"), dtype="str"
    )
    train_object_names = set([str.split(".")[0] for str in train_object_names])


    train_dt = WeightDataset(
        mlps_folder_train,
        None,
        0, # model.dims hardcoded in Transformer
        mlp_kwargs,
        cfg,
        train_object_names,
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
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    count = 0 

    if log_wandb:
        wandb.init(
                entity='adl-cv',
                project="UNet eval",
                name=f"normalised correct",
                #config={'attention_encoder': attention_encoder, 'num_imgs':num_imgs},
            )

        wandb_logger = WandbLogger()
        
    while count < num_imgs:
        sample, _, _ = test_dt.__getitem__(count)
        sample *= 0.6829066828697457
        sample = sample.to(device)
        sample_padded = torch.nn.functional.pad(sample.view((1, -1)), (0,31)).to(device)
        enc_sample = model(sample_padded)
        enc_sample = enc_sample.view((-1,))
        enc_sample = enc_sample[:-31]
        enc_sample /= 0.6829066828697457

        true_img = generate_images_from_VAE(sample)
        print('true_img')
        pred_img = generate_images_from_VAE(enc_sample)
        print('pred_img')
        if log_wandb:
            wandb_logger.log_image(
                    "true_renders", true_img, step=count
                )
            wandb_logger.log_image(
                    "generated_renders", pred_img, step=count
                )
        count += 1

def generate_during_training(model, samples, epoch, device, wandb_logger, remove_indx=None):

    if remove_indx:
        padding = 21
    else:
        padding = 31
    
    for i, sample in enumerate(samples):
        sample *= 0.6930342789347619
        #sample = sample.view(1, 1, sample.shape[0])
        sample_padded = torch.nn.functional.pad(sample.view((1, -1)),  (0,padding)).to(device)
        enc_sample = model(sample_padded)
        #mse = torch.nn.functional.mse_loss(sample_padded, enc_sample)
        enc_sample = enc_sample.view((-1,))
        enc_sample = enc_sample[:-padding]
        enc_sample /= 0.6930342789347619
        sample /= 0.6930342789347619
        device = torch.device('cpu') # do rest on cpu to ensure it fits on ram in gpu
        sample = sample.to(device)
        enc_sample = enc_sample.to(device)

        if remove_indx:
            # add the index again
            index = torch.load('/Users/manuelsenge/Documents/TUM/Semester_3/ADL4CV/workspace/HyperDiffusion/src/ldm_autoencoder/cache/indx_same_std.pt').to(device)
            values = torch.load('/Users/manuelsenge/Documents/TUM/Semester_3/ADL4CV/workspace/HyperDiffusion/src/ldm_autoencoder/cache/elements_same_std.pt').to(device)
            # add missing elements to encoded sample
            result_tensor = torch.zeros(enc_sample.size(0) + len(index), dtype=enc_sample.dtype).to(device)
            result_indices = torch.arange(result_tensor.size(0)).to(device)
            insert_mask = torch.zeros_like(result_indices, dtype=torch.bool).to(device)
            insert_mask[index] = True
            result_tensor[insert_mask] = values
            result_tensor[~insert_mask] = enc_sample
            enc_sample = result_tensor
            
            # add missing elements to real sample
            result_tensor = torch.zeros(sample.size(0) + len(index), dtype=sample.dtype).to(device)
            result_indices = torch.arange(result_tensor.size(0)).to(device)
            insert_mask = torch.zeros_like(result_indices, dtype=torch.bool).to(device)
            insert_mask[index] = True
            result_tensor[insert_mask] = values
            result_tensor[~insert_mask] = sample
            sample = result_tensor

        true_img = generate_images_from_VAE(sample)
        pred_img = generate_images_from_VAE(enc_sample)
        wandb_logger.log_image(
                "true_renders", true_img, step=epoch*10+i
            )
        
        wandb_logger.log_image(
                "generated_renders", pred_img, step=epoch*10+i
            )



if __name__ == "__main__":
    main()