# from torchvision.utils import make_grid
# import torchvision
# from utils.plot_utils import visualizeLeads_comp
from models.image.discriminator import Discriminator as DiscriminatorImg
from models.image.vqvae import VQVAEImg
from models.ecg.vqvae import VQVAEECG
from models.ecg.discriminator import ECG_SpatioTemporalNet
from models.ecg.network_params import spatioTemporalParams_v4
from dataset.celeb_dataset import CelebDataset
from dataset import dataset
# import matplotlib.pyplot as plt
import wandb
# from tqdm.auto import tqdm
from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
import argparse
import torch.optim as optim
import torch.nn as nn
import torch
import yaml
import os
import datetime

import sys

from vq_vae.VQVAE_Training import VQVAETraining
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# recon_criterion = nn.MSELoss()


parser = argparse.ArgumentParser()

timestamp = ""

parser.add_argument("--batch_size", type=int, default=42)
parser.add_argument("--n_epochs", type=int, default=450)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--log_interval", type=int, default=1)
parser.add_argument("--scale_training_size", type=float, default=0.1)
parser.add_argument("--save_every", type=int, default=1,
                    help="Save model every n epochs")
parser.add_argument("--logtowandb", action='store_true',
                    default=False, help="Log to wandb")
parser.add_argument("--train_ecgs", action='store_true',
                    default=False, help="Train on ECGs")
parser.add_argument("--learning_rate_disc", type=float, default=1e-4)
parser.add_argument("--beta", type=float, default=0.25)
parser.add_argument("--disc_epoch_start", type=int, default=25)
parser.add_argument("--disc_loss_weight", type=float, default=0.5)


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# train_dataset, val_dataset = dataset.get_datasets(args.scale_training_size)


def ddp_setup():
    init_process_group(backend='nccl')


def main():
    # Initialize distributed training if running with torchrun
    if "LOCAL_RANK" in os.environ:
        ddp_setup()

    gpu_id = int(os.environ.get("LOCAL_RANK", 0))
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    args.vqvae_checkpoint = "/uu/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg_latent_diff/vq_vae/results/ecgs/vqvae_2025-07-18_18-23-09/checkpoint.pt"

    train_ecgs = args.train_ecgs
    if train_ecgs:
        print(f"Loading mean and stds for the dataset")
        stats = torch.load(
            '/uu/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg_latent_diff/data/ecg_train_stats.pt', weights_only=True, map_location=device)
        mean = stats['mean']
        std = stats['std']
        print(f"Means: {mean}, Stds: {std}")
        config_path = "/uu/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg_latent_diff/configs/ecg_diff.yaml"
        train_dataset, val_dataset = dataset.get_datasets(
            args.scale_training_size)

    else:
        config_path = "/uu/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg_latent_diff/configs/im_diff.yaml"
        stats = None
        mean = None
        std = None
        train_dataset = CelebDataset(split='train')

    with open(config_path, 'r') as f:
        try:
            model_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    print(model_config)

    config = dict(
        means=mean,
        stds=std,
        args=args,
        autoencoder_config=model_config['autoencoder_config']
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=8, pin_memory=True,
                                  drop_last=True,
                                  sampler=DistributedSampler(train_dataset, drop_last=True, shuffle=True) if "LOCAL_RANK" in os.environ else None)

    if train_ecgs:
        model = VQVAEECG(model_config=model_config['autoencoder_config'])
        print("Setting up DISCRIMINATOR")
        firstLayerParams = dict(
            in_channels=1, out_channels=32, bias=True, kernel_size=(1, 7), maxPoolKernel=7)
        lastLayerParams = dict(maxPoolSize=(8, 1))
        discriminatorParams = {'temporalResidualBlockParams': spatioTemporalParams_v4['temporalResidualBlockParams'],
                               'spatialResidualBlockParams': spatioTemporalParams_v4['spatialResidualBlockParams'],
                               'integrationMethod': 'concat', 'problemType': 'BCELogits', 'firstLayerParams': firstLayerParams, 'lastLayerParams': lastLayerParams}

        discriminator = ECG_SpatioTemporalNet(**discriminatorParams)
        optimizer_disc = optim.AdamW(
            discriminator.parameters(), lr=args.learning_rate_disc, betas=(0.5, 0.999))

    else:
        model = VQVAEImg(
            im_channels=3, model_config=model_config['autoencoder_config'])
        discriminator = DiscriminatorImg(im_channels=3)
        optimizer_disc = optim.AdamW(
            discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    discriminator_setup = {
        'discriminator': discriminator,
        'optimizer': optimizer_disc
    }
    model.train()

    results_folder = f'./results/{"ecgs" if train_ecgs else "imgs"}/vqvae_{formatted_time}'
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(f"{results_folder}/plots", exist_ok=True)

    if gpu_id == 0 and args.logtowandb:
        wandbrun = wandb.init(
            project="ecg_vqvae" if train_ecgs else "img_vqvae",
            notes=f"A UNET and 1M dataset" if not train_ecgs else "A VQVAE and 30K Images",
            tags=["vqvae", "bigdataset"] if not train_ecgs else [
                "vqvae", "images"],
            entity="deekshith",
            reinit=True,
            config=config,
            name=f"{"vqvae"}_{formatted_time}"
        )

    trainer = VQVAETraining(
        args=args,
        model=model,
        optimizer=optimizer,
        dataloader=train_dataloader,
        results_folder=results_folder,
        stats=stats,
        discriminator_setup=discriminator_setup,
        checkpoint_path=f"{results_folder}/checkpoint.pt" if os.path.exists(
            f"{results_folder}/checkpoint.pt") else None
    )
    trainer.train()

    if gpu_id == 0 and args.logtowandb:
        wandbrun.finish()

    # Clean up distributed training if it was initialized
    if "LOCAL_RANK" in os.environ:
        destroy_process_group()


if __name__ == "__main__":
    main()
