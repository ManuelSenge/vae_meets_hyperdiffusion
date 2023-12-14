import torch
import trimesh
import os
import copy
import wandb
from pytorch_lightning.loggers import WandbLogger

import sys
sys.path.append('../')
import tqdm
from hd_utils import generate_mlp_from_weights, render_mesh
from model_transformer_conv import VariationalAutoencoder
from siren import sdf_meshing
from siren.experiment_scripts.test_sdf import SDFDecoder
from omegaconf import OmegaConf

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

def sample_from_VAE(model, N_dist):
    sample = N_dist.sample((1, 512))
    sample = sample.view((sample.shape[0], 1, sample.shape[1]))
    return model.decoder(sample)

if __name__ == "__main__":
    num_imgs = 3
    model_path = '/Users/manuelsenge/Documents/TUM/Semester_3/ADL4CV/workspace/HyperDiffusion/vae/output_files/bn_lr0.0002_E0011_num_att_layers1_enc_chans128_64_32_1_enc_kernel_sizes8_6_3_3_1234.pt'
    attention_encoder = "0011"
    attention_decoder = attention_encoder[::-1]
    log_wandb = 1
    num_att_layers = 1
    enc_chans = [128, 64, 32, 1]
    enc_kernel_sizes = [8, 6, 3, 3]
    wandb_enc_channels = "_".join([str(enc) for enc in enc_chans])

    model = VariationalAutoencoder(input_dim=36737,
                                   latent_dims=512,
                                   device=device,
                                   enc_chans=enc_chans,
                                   enc_kernal_sizes=enc_kernel_sizes,
                                   self_attention_encoder=[int(elem) for elem in list(attention_encoder)],
                                   self_attention_decoder=[int(elem) for elem in list(attention_decoder)],
                                   num_att_layers=num_att_layers)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    N_dist = torch.distributions.Normal(0, 1)
    N_dist.loc = N_dist.loc.to(device) # hack to get sampling on the GPU
    N_dist.scale = N_dist.scale.to(device)

    if log_wandb:
        wandb.init(
                entity='adl-cv',
                project="VAE eval",
                name=f"batch_norm_{attention_encoder}_enc_chans{wandb_enc_channels}",
                config={'attention_encoder': attention_encoder, 'num_imgs':num_imgs},
            )

        wandb_logger = WandbLogger()

    for i in range(num_imgs):
        x_0 = sample_from_VAE(model, N_dist)
        img = generate_images_from_VAE(x_0)
        if log_wandb:
            wandb_logger.log_image(
                    "generated_renders", img, step=i
                )
