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
from siren import sdf_meshing
from siren.experiment_scripts.test_sdf import SDFDecoder
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from dataset import WeightDataset
from hd_utils import Config
from omegaconf import DictConfig
from model.ldm.modules.autoencoder import AutoencoderKL

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
    model_path = '/Users/manuelsenge/Documents/TUM/Semester_3/ADL4CV/workspace/HyperDiffusion/our_LDM_Autoencoder/model/models/ldm_latent_1149_attention_[2298]_dropout_0.0_lr_0.0002_num_res_3_ch_mult_[1,+4,+8,+16,+32,+128]_normalized_1234.pt'
    log_wandb = 1

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

    config_model = OmegaConf.load('/Users/manuelsenge/Documents/TUM/Semester_3/ADL4CV/workspace/HyperDiffusion/our_LDM_Autoencoder/model/autoencoder_kl_8x8x64.yaml')
    loss_config = config_model.model.params.lossconfig
    ddconfig = config_model.model.params.ddconfig
    model = AutoencoderKL(ddconfig=ddconfig, lossconfig=loss_config, embed_dim=1149)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    count = 0 

    if log_wandb:
        wandb.init(
                entity='adl-cv',
                project="LDM-AE eval",
                name=f"attention 2298 train samples",
                #config={'attention_encoder': attention_encoder, 'num_imgs':num_imgs},
            )

        wandb_logger = WandbLogger()
        
    while count < num_imgs:
        sample, _, _ = train_dt.__getitem__(count)
        # normalize sample
        sample *= 0.6930342789347619
        sample = sample.view(1, 1, sample.shape[0])
        sample_padded = torch.nn.functional.pad(sample,  (0,31)).to(device)
        enc_sample, posterior = model(sample_padded)
        enc_sample = enc_sample.view((-1,))
        enc_sample = enc_sample[:-31]
        enc_sample /= 0.6930342789347619

        #true_img = generate_images_from_VAE(sample)
        #print('true_img')
        pred_img = generate_images_from_VAE(enc_sample)
        print('pred_img')
        if log_wandb:
            #wandb_logger.log_image(
            #        "true_renders", true_img, step=count
            #    )
            wandb_logger.log_image(
                    "generated_renders", pred_img, step=count
                )
        count += 1



if __name__ == "__main__":
    main()