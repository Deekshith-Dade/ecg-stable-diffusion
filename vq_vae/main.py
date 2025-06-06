import os
import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from tqdm.auto import tqdm
import wandb
import matplotlib.pyplot as plt

from data import dataset
from vq_vae.vqvae import VQVAE
from utils.plot_utils import visualizeLeads_comp


parser = argparse.ArgumentParser()

timestamp = ""

parser.add_argument("--batch_size", type=int, default=48)
parser.add_argument("--n_epochs", type=int, default=250)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=1)
parser.add_argument("--save_every", type=int, default=10, help="Save model every n epochs")
parser.add_argument("--logtowandb", action='store_true', default=False, help="Log to wandb")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
train_dataset, val_dataset = dataset.get_datasets(scale_training_size=0.1)


def ddp_setup():
    init_process_group(backend='nccl')

def calculate_mean_std(dataset):
    print("Calculating Means and Stds of Dataset")
    
    means = torch.zeros(8)
    stds = torch.zeros(8)
    n_samples = 0

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    for batch in tqdm(dataloader, desc="Calculating mean & std"):
        data = batch['image']  # Shape: [batch_size, 1, 8, 2500]
        batch_size = data.size(0)
        data = data.squeeze(1)  # Shape: [batch_size, 8, 2500]
        
        means += data.sum(dim=(0, 2))  # Sum over batch and time dimensions
        stds += (data ** 2).sum(dim=(0, 2))  # Sum of squares
        n_samples += batch_size * data.size(2)  # batch_size * 2500
    
    means /= n_samples  # Final mean
    stds = torch.sqrt((stds / n_samples) - means ** 2 + 1e-6)  # Variance formula

    print(f"Mean per lead: {means}")
    print(f"Std per lead: {stds}")
    
    means = means.to(device)
    stds = stds.to(device)
    return means, stds

def train(model, optimizer, dataloader, means, stds, results_folder):
    
    # Check if we're running with distributed training
    use_ddp = "LOCAL_RANK" in os.environ
    
    if use_ddp:
        gpu_id = int(os.environ["LOCAL_RANK"])
        model = model.to(gpu_id)
    else:
        gpu_id = 0
        model = model.to(device)
    
    results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
    'embedding_loss': [],
    }
    
    
    epochs_run = 0
    if os.path.exists(f"{results_folder}/checkpoint.pt"):
        checkpoint = torch.load(f"{results_folder}/checkpoint.pt", map_location=device)
        # Load state dict - handle both DDP and non-DDP saved models
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs_run = checkpoint['epoch']
        results['n_updates'] = checkpoint['n_updates']
        print(f"Resuming training from epoch {epochs_run} with {results['n_updates']} updates.")

    if use_ddp:
        model = DDP(model, device_ids=[gpu_id])
        means = means.to(gpu_id)
        stds = stds.to(gpu_id)
    
    for epoch in range(epochs_run, args.n_epochs):
        print(f"Training step {epoch+1}/{args.n_epochs}", end='\r')
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            # Use the correct device (GPU ID for DDP, or global device for single GPU)
            target_device = gpu_id if use_ddp else device
            x = batch['image'].unsqueeze(1).to(target_device)
            # import pdb; pdb.set_trace()  
            x = (x - means) / stds
            
            optimizer.zero_grad()
            
            embedding_loss, x_hat, perplexity = model(x)
            recon_loss = torch.mean((x_hat - x) ** 2 / (stds ** 2))
            loss = recon_loss + embedding_loss
            
            loss.backward()
            optimizer.step()
            
            results['recon_errors'].append(recon_loss.cpu().detach().numpy())
            results['perplexities'].append(perplexity.cpu().detach().numpy())
            results['loss_vals'].append(loss.cpu().detach().numpy())
            results["n_updates"] = epoch * len(dataloader) + batch_idx + 1
            results["embedding_loss"].append(embedding_loss.cpu().detach().numpy())


        if (gpu_id == 0) and (epoch % args.save_every == 0 or epoch == args.n_epochs - 1):
            print(f"Saving checkpoint at epoch {epoch}")
            # Handle both DDP and non-DDP cases for state_dict
            model_state_dict = model.module.state_dict() if use_ddp else model.state_dict()
            checkpoint = {
                'epoch': epoch,
                'n_updates': results['n_updates'],
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, f"{results_folder}/checkpoint.pt")


        if (gpu_id == 0) and (epoch % args.log_interval == 0 or epoch == args.n_epochs - 1):
            training_log = dict(
                    step = results['n_updates'],
                    recon_error = np.mean(results['recon_errors'][-args.log_interval * len(dataloader):]),
                    loss = np.mean(results['loss_vals'][-args.log_interval * len(dataloader):]),
                    perplexity = np.mean(results['perplexities'][-args.log_interval * len(dataloader):]),
                    embedding_loss = np.mean(results['embedding_loss'][-args.log_interval * len(dataloader):]),
                )

            print(f"Step {results['n_updates']}, Recon Error: {training_log['recon_error']}, Loss: {training_log['loss']}, Perplexity: {training_log['perplexity']}, Embedding Loss: {training_log['embedding_loss']}")

            if args.logtowandb:
                x = x * stds + means
                x_hat = x_hat * stds + means

                fig1 = visualizeLeads_comp(x[0].squeeze().detach().cpu(), "comparison 1", x_hat[0].squeeze().detach().cpu(), f"{results_folder}/plots/{results['n_updates']}_fig1.png")
                plt.close()
                fig2 = visualizeLeads_comp(x[1].squeeze().detach().cpu(), "comparison 2", x_hat[1].squeeze().detach().cpu(), f"{results_folder}/plots/{results['n_updates']}_fig2.png")
                plt.close()
                
                training_log['fig1'] = fig1
                training_log['fig2'] = fig2

                wandb.log(training_log)


def main():
    # Initialize distributed training if running with torchrun
    if "LOCAL_RANK" in os.environ:
        ddp_setup()
    
    gpu_id = int(os.environ.get("LOCAL_RANK", 0))
    print(f"Calculating mean and stds for the dataset")
    # means, stds = calculate_mean_std(train_dataset)
    # means = means.reshape(1, 1, -1, 1)
    # stds = stds.reshape(1, 1, -1, 1)
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    stats = torch.load('/uu/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg_latent_diff/data/ecg_train_stats.pt', weights_only=True)
    means = stats['mean'].to(device)
    stds = stats['std'].to(device)
    print(f"Means: {means}, Stds: {stds}")
    # torch.save({'mean': means, 'std': stds}, '/uu/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg_latent_diff/data/ecg_train_stats.pt')

    config = dict(
        means = means,
        stds = stds,
        args = args
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                  shuffle=False, num_workers=8, pin_memory=True, 
                                  drop_last=True,
                                  sampler=DistributedSampler(train_dataset, drop_last=True, shuffle=True) if "LOCAL_RANK" in os.environ else None)
    
    model = VQVAE()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

    model.train()

    
    results_folder = f'./results/vqvae_{formatted_time}'
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(f"{results_folder}/plots", exist_ok=True)
    
    if gpu_id == 0 and args.logtowandb:
        wandbrun = wandb.init(
            project="ecg_vqvae",
            notes = f"A simple trial of vqvae with small encoder",
            tags = ["vqvae", "bigdataset"],
            entity="deekshith",
            reinit=True,
            config=config,
            name=f"{"vqvae"}_{datetime.datetime.now()}"
        )
        
    train(model, optimizer, train_dataloader, means, stds, results_folder)

    if gpu_id == 0 and args.logtowandb:
        wandbrun.finish()
    
    # Clean up distributed training if it was initialized
    if "LOCAL_RANK" in os.environ:
        destroy_process_group()


if __name__ == "__main__":
    main()