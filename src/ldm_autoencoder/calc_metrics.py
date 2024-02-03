import torch
import trimesh
from hd_utils import (Config, calculate_fid_3d, generate_mlp_from_weights,
                      render_mesh, render_meshes)
from siren.experiment_scripts.test_sdf import SDFDecoder
from siren import sdf_meshing
import copy
import numpy as np
from scipy.spatial.transform import Rotation
import os
from tqdm import tqdm
from evaluation_metrics_3d import compute_all_metrics
from dataset import WeightDataset


# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
def distChamfer(a, b):
    """
    a: sample batch
    b: reference batch
    """
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P.min(1)[0], P.min(2)[0]

def generate_meshes(x_0s, mlp_kwargs, device, folder_name="meshes", info="0", res=64, level=0):
        current_epoch = 0
        x_0s = x_0s.view(len(x_0s), -1)
        curr_weights = Config.get("curr_weights")
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
                    f"{folder_name}/mesh_epoch_{current_epoch}_{i}_{info}"
                    if folder_name is not None
                    else None
                )
                if mlp_kwargs.move:
                    for i in range(16):
                        v, f, sdf = sdf_meshing.create_mesh(
                            sdf_decoder,
                            effective_file_name,
                            N=res,
                            level=0
                            if mlp_kwargs.output_type in ["occ", "logits"]
                            else 0,
                            time_val=i,
                        )  # 0.9
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
                else:
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

def calc_metrics(sample_method, mlp_kwargs, cfg, orig_meshes_dir, split_type, test_object_names, dataset_path, device):
    n_points = cfg.val.num_points

    remove_std_zero_indices = True
    removed_std_indices_path = '../data/std_zero_indices_planes.pt'

    if remove_std_zero_indices:
        removed_std_indices = torch.load(removed_std_indices_path)

    # test_object_names = list(test_object_names)[:10]

    dataset = WeightDataset(
        '../' + Config.get("mlps_folder_train"),
        None,
        0,
        mlp_kwargs,
        cfg,
        test_object_names,
        1,
        remove_std_zero_indices,
        removed_std_indices,
    )
    # First process ground truth shapes
    pcs = []
    for obj_name in test_object_names:
        pc = np.load(os.path.join(dataset_path, obj_name + '.obj' + ".npy"))
        pc = pc[:, :3]


        pc = torch.tensor(pc).type(torch.float32)
        if split_type == "test":
            pc = pc.type(torch.float32)
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
            pc = (pc - shift) / scale
        pcs.append(pc)
    r = Rotation.from_euler("x", 90, degrees=True)
    ref_pcs = torch.stack(pcs)

    # We are generating slightly more than ref_pcs
    number_of_samples_to_generate = int(len(ref_pcs) * cfg.test_sample_mult)

    sample_x_0s = []
    for i in tqdm(range(len(dataset))):
        if i == (len(dataset) - 1):
            sample = sample_method(dataset[i][0], final_sample=True)
        else:
            sample = sample_method(dataset[i][0])
        sample_x_0s.append(sample.detach())
    torch.cuda.empty_cache()
    sample_x_0s = torch.vstack(sample_x_0s)
    torch.save(sample_x_0s, f"{orig_meshes_dir}/prev_sample_x_0s.pth")
    print(sample_x_0s.shape)
    if cfg.dedup:
        sample_dist = torch.cdist(
            sample_x_0s,
            sample_x_0s,
            p=2,
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        sample_dist_min = sample_dist.kthvalue(k=2, dim=1)[0]
        sample_dist_min_sorted = torch.argsort(sample_dist_min, descending=True)[
            : int(len(ref_pcs) * 1.01)
        ]
        sample_x_0s = sample_x_0s[sample_dist_min_sorted]
        print(
            "sample_dist.shape, sample_x_0s.shape",
            sample_dist.shape,
            sample_x_0s.shape,
        )
    torch.save(sample_x_0s, f"{orig_meshes_dir}/sample_x_0s.pth")
    print("Sampled")

    print("Running marching cubes")
    sample_batch = []
    for x_0s in tqdm(sample_x_0s):
        mesh, _ = generate_meshes(
            x_0s.unsqueeze(0) / cfg.normalization_factor,
            mlp_kwargs,
            device,
            None,
            res=356 if split_type == "test" else 256,
            level=1.386 if split_type == "test" else 0,
        )
        mesh = mesh[0]
    
        if len(mesh.vertices) > 0:
            pc = torch.tensor(mesh.sample(n_points))
            if not cfg.mlp_config.params.move and "hyper" in cfg.method:
                pc = pc * 2
            pc = pc.type(torch.float32)
            if split_type == "test":
                pc = pc.type(torch.float32)
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = pc.flatten().std().reshape(1, 1)
                pc = (pc - shift) / scale
        else:
            print("Empty mesh")
            if split_type in ["val", "train"]:
                pc = torch.zeros_like(ref_pcs[0])
            else:
                continue
        sample_batch.append(pc)
    print("Marching cubes completed")

    print("number of samples generated:", len(sample_batch))
    sample_batch = sample_batch[: len(ref_pcs)]
    print("number of samples generated (after clipping):", len(sample_batch))
    sample_pcs = torch.stack(sample_batch)
    assert len(sample_pcs) == len(ref_pcs)
    torch.save(sample_pcs, f"{orig_meshes_dir}/samples.pth")

    print("Starting metric computation for", split_type)
    metrics = dict()
    fid = calculate_fid_3d(
        sample_pcs #.to(cfg.device)
        , ref_pcs#.to(cfg.device)
        , None
    )
    metrics = compute_all_metrics(
        sample_pcs.to(cfg.device),
        ref_pcs.to(cfg.device),
        16 if split_type == "test" else 16,
        None,
    )
    metrics["fid"] = fid.item()
    

    print("Completed metric computation for", split_type)

    return metrics
