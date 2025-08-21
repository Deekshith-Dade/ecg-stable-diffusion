from torchvision.utils import make_grid
from tqdm import tqdm
import yaml
import wandb
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import matplotlib.pyplot as plt
from utils.plot_utils import visualizeLeads_comp
from sampling.sampling_utils import diff_random_sample
from torch.utils.data import DataLoader
from dataset.celeb_dataset import CelebDataset
from dataset.dataset import getKCLTrainTestDataset
from diff_utils import drop_text_condition, drop_class_condition
from utils.text_utils import get_tokenizer_and_model, get_text_representation
from scheduler import LinearNoiseScheduler, CosineNoiseScheduler
from models.image.unet_cond_base import Unet as UnetImg
from models.unet import Unet
# from models.image.vqvae import VQVAE as VQVAEImg
from vqvae_models.ecg.vqvae import VQVAEECG
import torch.optim as optim
import torch
import numpy as np
import os
import datetime
import sys
from training import Training
sys.path.append("..")


def main():
    logtowandb = False
    config_path = "/uu/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg_latent_diff/configs/ecg_diff.yaml"
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    if "LOCAL_RANK" in os.environ:
        init_process_group(backend='nccl')
    gpu_id = int(os.environ.get("LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        gpu_id = torch.device("cpu")

    config['device'] = gpu_id
    config['logtowandb'] = logtowandb

    stats = torch.load('/uu/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg_latent_diff/data/ecg_train_stats.pt',
                       weights_only=False, map_location="cpu")
    means = stats['mean'].to(gpu_id)
    stds = stats['std'].to(gpu_id)

    config['dataset_config']['means'] = means
    config['dataset_config']['stds'] = stds
    # config['train_params']['checkpoint_path'] = "/uu/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg_latent_diff/diffusion/results/01_ecg/2025-08-01_18-32-03/checkpoint.pt"
    config['train_params']['checkpoint_path'] = None

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    if config['train_params']['checkpoint_path'] is not None:
        formatted_time = config['train_params']['checkpoint_path'].split(
            "/")[-2]
    results_folder = f"./results/01_ecg/{formatted_time}"
    config['train_params']['results_folder'] = results_folder

    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(f"{results_folder}/plots", exist_ok=True)

    if (gpu_id == 0 or gpu_id == torch.device("cpu")):
        print(config)
        print(f"Saving Results @{results_folder}")
        print(f"GPU_ID: {gpu_id}")

    if (gpu_id == 0 or gpu_id == torch.device("cpu")) and logtowandb:
        wandbrun = wandb.init(
            project="latent_ecg",
            notes=f"diffusion in latent space",
            tags=["latent", "diffusion"],
            entity="deekshith",
            reinit=True,
            config=config,
            name=f"{"ecg_latent"}_{formatted_time}",
            resume="allow",
            # id="qw0mzged"
        )

    trainer = Training(config)
    trainer.train()

    if (gpu_id == 0 or gpu_id == torch.device("cpu")) and logtowandb:
        wandbrun.finish()

    if "LOCAL_RANK" in os.environ:
        destroy_process_group()


if __name__ == "__main__":
    main()
