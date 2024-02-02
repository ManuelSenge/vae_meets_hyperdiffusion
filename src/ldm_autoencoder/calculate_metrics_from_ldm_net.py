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
from sampler import SingleInstanceBatchSampler
from siren import sdf_meshing
from siren.experiment_scripts.test_sdf import SDFDecoder
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from dataset import WeightDataset
from hd_utils import Config
from omegaconf import DictConfig
from model.ldm.modules.autoencoder import AutoencoderKL
from model.ldm.modules.autoencoder_old import OldAutoencoderKL
from model.ldm.modules.distributions.distributions import DiagonalGaussianDistribution
import re
from calc_metrics import calc_metrics
from functools import partial

os.environ['PYOPENGL_PLATFORM'] = 'egl'
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
    'use_leaky_relu':False, 'move':False, 'device':str(device)}) #'cuda'

i = 0
mesh_dir = "../gen_meshes/ldm"

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

def generate_images_from_VAE(x_0, resolution=420):
    out_imgs = []
    if not os.path.isdir(mesh_dir):
        os.makedirs(mesh_dir)
    mesh, _ = generate_meshes(x_0.unsqueeze(0), None, res=resolution)
    mesh = mesh[0]
    mesh.vertices *= 2
    mesh.export(os.path.join(mesh_dir, f"mesh_{len(out_imgs)}.obj"))

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
    resolution=500

    model_file = 'rmv_ind_ldm_latent_517_attention_[2068, 1034, 517]_dropout_0.0_lr_0.0002_num_res_9_ch_mult_[4, 8, 16, 32, 64, 128, 256]_normalizedbeta_1_1234.pt'
    from_checkpoint =True
    use_validation = True
    variational = False
    var_sample_count = 5
    remove_std_zero_indices = True
    removed_std_indices_path = '../data/std_zero_indices_planes.pt'

    old_autoencoder = True

    if remove_std_zero_indices:
        removed_std_indices = torch.load(removed_std_indices_path)


    single_sample_overfit = model_file.startswith('single_sample_overfit')
    if from_checkpoint:
        model_file = 'cp_' + model_file
        
    model_path = os.path.join('../output_files', model_file)

    posterior_path_no_checkpoint = os.path.join('../output_files', 'posterior_' + model_file)

    single_sample_overfit_index = None
    if single_sample_overfit:
        single_sample_overfit_index_match = re.match(r'index_(\d)+', model_file)
        if single_sample_overfit_index:
            if single_sample_overfit_index_match and len(single_sample_overfit_index_match.groups() > 0):
                single_sample_overfit_index = int(single_sample_overfit_index_match.group(1))

    if single_sample_overfit_index is None:
        single_sample_overfit_index = 1

    print("Single sample overfit index:", single_sample_overfit_index)

    log_wandb = 0

    dataset_path = os.path.join('..', Config.config["dataset_dir"], Config.config["dataset"])
    dataset_pc_path = os.path.join('..', Config.config["dataset_dir"], Config.config["dataset"] + '_2048_pc')
    test_object_names = np.genfromtxt(
        os.path.join(dataset_path, "test_split.lst"), dtype="str"
    )
    test_object_names = set([str.split(".")[0] for str in test_object_names])
    mlps_folder_train = '../' + Config.get("mlps_folder_train")

    print(removed_std_indices)

    test_dt = WeightDataset(
        mlps_folder_train,
        None,
        0,
        mlp_kwargs,
        cfg,
        test_object_names,
        1,
        remove_std_zero_indices,
        removed_std_indices,
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
    val_object_names = np.genfromtxt(
            os.path.join(dataset_path, "val_split.lst"), dtype="str"
    )
    val_object_names = set([str.split(".")[0] for str in val_object_names])
    val_dt = WeightDataset(
        mlps_folder_train,
        None,
        0,
        mlp_kwargs,
        cfg,
        val_object_names,
        normalize=1,
        remove_std_zero_indices=remove_std_zero_indices,
        removed_std_indices=removed_std_indices
    )
    train_dt = WeightDataset(
        mlps_folder_train,
        None,
        0, # model.dims hardcoded in Transformer
        mlp_kwargs,
        cfg,
        train_object_names,
        1,
        remove_std_zero_indices,
        removed_std_indices,
    )

    removed_std_values = train_dt.get_removed_std_values()

    config_model = OmegaConf.load('./model/autoencoder_kl_8x8x64.yaml')
    loss_config = config_model.model.params.lossconfig
    ddconfig = config_model.model.params.ddconfig

    if old_autoencoder:
        model = OldAutoencoderKL(ddconfig=ddconfig, lossconfig=loss_config, embed_dim=517)
    else:
        model = AutoencoderKL(ddconfig=ddconfig, lossconfig=loss_config, embed_dim=517)

    model_dict = torch.load(model_path, map_location=device)
    if from_checkpoint:
        model.load_state_dict(model_dict['model_state_dict'])
        if variational:
            posterior_params =  model_dict['posterior']
            posterior = DiagonalGaussianDistribution(posterior_params, variational=True)
        print("Using model at epoch", model_dict['epoch'])
    else:
        model.load_state_dict(model_dict)
        if variational:
            posterior_params = torch.load(posterior_path_no_checkpoint)
            posterior = DiagonalGaussianDistribution(posterior_params, variational=True)

    model = model.to(device)
    
    model.eval()

    count = 0 

    if log_wandb:
        wandb.init(
                entity='adl-cv',
                project="LDM-AE eval" if not variational else "LDM_VAE eval",
                name=f"eval_{model_file}",
                group="Julian"
                #config={'attention_encoder': attention_encoder, 'num_imgs':num_imgs},
            )

        wandb_logger = WandbLogger()
    
    # num_imgs = 3
        
    padding = 21 if remove_std_zero_indices else 31
    train_img_indices = [3,4,9,10,12,13,14,16,21, 24,28] if not single_sample_overfit else [single_sample_overfit_index]
    train_img_indices=[0, 1, 2, 3, 4, 5, 6, 7 ,8, 9, 10, 11, 12]
    

    if variational:
        assert var_sample_count > 0

        train_img_indices = [i for i in range(var_sample_count)]

    def sample_from_model(original_sample, variational=False):
        if variational:
            sample = posterior.sample()[0]
            dec_sample = model.decode(sample.view(1, sample.shape[0], sample.shape[1]))
            dec_sample = dec_sample.view((-1,))
            dec_sample = dec_sample[:-padding]
            dec_sample /= 0.6930342789347619

            dec_sample = dec_sample.to(cpu_device)
        else:
            # original_sample, _, _ = dataset.__getitem__(ix)
            # normalize sample
            sample = original_sample * 0.6930342789347619
            sample = sample.view(1, 1, sample.shape[0])
            sample_padded = torch.nn.functional.pad(sample,  (0,padding)).to(device)
            dec_sample, posterior = model(sample_padded)
            dec_sample = dec_sample.view((-1,))
            dec_sample = dec_sample[:-padding]
            dec_sample /= 0.6930342789347619
            dec_sample = dec_sample.to(cpu_device)
            print("MSE_Loss:", torch.nn.functional.mse_loss(original_sample, dec_sample))
                    
        if remove_std_zero_indices:
            dec_sample = _add_removed_indices(dec_sample, removed_std_indices, removed_std_values, cpu_device)
            if not variational:
                original_sample = _add_removed_indices(original_sample, removed_std_indices, removed_std_values, cpu_device)
        return dec_sample
    
    

    sample_from_model = partial(sample_from_model, variational=variational)

    print("Using validation data" if val_dt else "Using train data")
    dataset = val_dt if use_validation else train_dt
    print("train_img_indices:",  train_img_indices)
    cpu_device = torch.device('cpu')
    metrics = calc_metrics(sample_from_model,
                            mlp_kwargs, 
                            cfg, 
                            '../output_files/meshes',
                            'train',
                            test_object_names,
                            dataset_pc_path,
                            device=device,)
    print('metrics', metrics)


def _add_removed_indices(sample, removed_indices, removed_values, device):
    result_tensor = torch.zeros(sample.size(0) + len(removed_indices), dtype=sample.dtype).to(device)
    result_indices = torch.arange(result_tensor.size(0)).to(device)
    insert_mask = torch.zeros_like(result_indices, dtype=torch.bool).to(device)
    insert_mask[removed_indices] = True
    result_tensor[insert_mask] = removed_values
    result_tensor[~insert_mask] = sample
    sample = result_tensor
    return sample

def generate_during_training(model, gen_dataset, gen_sample_indices, epoch,
                            device, wandb_logger, variational,
                            remove_std_zero_indices, removed_std_indices, removed_std_values, resolution):
    model.eval()
    padding = 21 if remove_std_zero_indices else 31

    normalizing_constant = gen_dataset.get_normalized_constant()

    print("Generating with normalizing constant ", normalizing_constant)

    cpu_device = torch.device('cpu')
    for i, sample_index in enumerate(gen_sample_indices):

        original_sample = None
        if variational:
            render_name = "random_sample"
            sample = model.posterior.sample()[0]
            sample = sample.view(1, sample.shape[0], sample.shape[1])
            dec_sample = model.decode(sample)
            dec_sample = dec_sample.view((-1,))
            dec_sample = dec_sample[:-padding]
            dec_sample /= normalizing_constant

            dec_sample = dec_sample.to(cpu_device)
        else: 
            render_name = "generated_sample"
            original_sample, _, _ = gen_dataset.__getitem__(sample_index)
            sample = original_sample.view(1, 1, original_sample.shape[0])
            sample_padded = torch.nn.functional.pad(sample,  (0,padding)).to(device)
            dec_sample, _ = model(sample_padded)
            dec_sample = dec_sample.view((-1,))

        
            dec_sample = dec_sample[:-padding].to(cpu_device)
            print(f"Generating image... (MSE-Loss: {torch.nn.functional.mse_loss(original_sample, dec_sample)})")
            dec_sample /= normalizing_constant
            dec_sample = dec_sample

        
        if remove_std_zero_indices:
            dec_sample = _add_removed_indices(dec_sample, removed_std_indices, removed_std_values, device=cpu_device)
            if not variational and epoch == 0:
                original_sample /= normalizing_constant
                original_sample = _add_removed_indices(original_sample, removed_std_indices, removed_std_values, device=cpu_device)

        pred_img = generate_images_from_VAE(dec_sample, resolution=resolution)

        wandb_logger.log_image(
                render_name, pred_img, step=epoch*10+i
        )
        if not variational and epoch == 0:
            assert original_sample is not None
            true_img = generate_images_from_VAE(original_sample, resolution=resolution)
            wandb_logger.log_image(
                    "true_renders", true_img, step=epoch*10+i
                )
            


if __name__ == "__main__":
    main()