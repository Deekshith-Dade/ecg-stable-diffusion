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
from tqdm.auto import tqdm
import wandb
import matplotlib.pyplot as plt

from data import dataset
from vq_vae.vqvae import VQVAE
from utils.plot_utils import visualizeLeads_comp


parser = argparse.ArgumentParser()

timestamp = ""

parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--n_epochs", type=int, default=250)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=256)
parser.add_argument("--beta", type=float, default=0.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=5)
parser.add_argument("--logtowandb", action='store_true', default=False, help="Log to wandb")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "device")
print(device)
train_dataset, val_dataset = dataset.get_datasets(scale_training_size=0.25)

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

model = VQVAE(n_embeddings=args.n_embeddings, embedding_dim=args.embedding_dim, beta=args.beta)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

model.train()



def train(dataloader, means, stds, results_folder):
    results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': []
}
    for epoch in range(args.n_epochs):
        print(f"Training step {epoch+1}/{args.n_epochs}", end='\r')
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            x = batch['image'].unsqueeze(1).to(device)
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

        if epoch % args.log_interval == 0 or epoch == args.n_epochs - 1:
            training_log = dict(
                    step = results['n_updates'],
                    recon_error = np.mean(results['recon_errors'][-args.log_interval:]),
                    loss = np.mean(results['loss_vals'][-args.log_interval:]),
                    perplexity = np.mean(results['perplexities'][-args.log_interval:]),
                )
                
            print(f"Step {results['n_updates']}, Recon Error: {training_log['recon_error']}, Loss: {training_log['loss']}, Perplexity: {training_log['perplexity']}")
            
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
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    
    
    results_folder = f'./results/vqvae_{formatted_time}'
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(f"{results_folder}/plots", exist_ok=True)
    
    if args.logtowandb:
        wandbrun = wandb.init(
            project="ecg_vqvae",
            notes = f"A simple trial of vqvae with small encoder",
            tags = ["vqvae", "bigdataset"],
            entity="deekshith",
            reinit=True,
            config=config,
            name=f"{"vqvae"}_{datetime.datetime.now()}"
        )
        
    train(train_dataloader, means, stds, results_folder)
    
    if args.logtowandb:
        wandbrun.finish()


if __name__ == "__main__":
    main()