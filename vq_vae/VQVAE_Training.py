from vq_vae.models.vqvae_base import VQVAEBase


import torch
from typing import Optional, Dict, Any
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import wandb
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from utils.plot_utils import visualizeLeads_comp


class VQVAETraining:
    def __init__(self,
                 args: argparse.Namespace,
                 model: VQVAEBase,
                 optimizer: torch.optim.Optimizer,
                 dataloader: torch.utils.data.DataLoader,
                 test_dataset: torch.utils.data.Dataset,
                 results_folder: str,
                 stats: Optional[dict] = None,
                 model_config=None,
                 discriminator_setup: Optional[Dict[str, Any]] = None,
                 checkpoint_path: Optional[str] = None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.test_dataset = test_dataset
        self.results_folder = results_folder
        self.discriminator: Optional[nn.Module] = discriminator_setup[
            'discriminator'] if discriminator_setup is not None else None
        self.optimizer_disc: Optional[torch.optim.Optimizer] = discriminator_setup[
            'optimizer'] if discriminator_setup is not None else None
        self.means = stats['mean'] if stats is not None else None
        self.stds = stats['std'] if stats is not None else None
        self.model_config = model_config

        self.use_ddp = "LOCAL_RANK" in os.environ
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if self.use_ddp:
            self.gpu_id = int(os.environ["LOCAL_RANK"])
            self.model = self.model.to(self.gpu_id)
            if self.discriminator:
                self.discriminator = self.discriminator.to(self.gpu_id)
        else:
            self.gpu_id = 0
            self.model = self.model.to(self.device)
            if self.discriminator:
                self.discriminator = self.discriminator.to(self.device)

        self.results = {
            'n_updates': 0,
            'recon_errors': [],
            'loss_vals': [],
            'perplexities': [],
            'embedding_loss': [],
            'min_encoding_indices': [],
            'vae_gen_loss': [],
            'disc_losses': [],
        }

        self.beta = args.beta
        self.disc_loss_weight = args.disc_loss_weight
        self.disc_epoch_start = args.disc_epoch_start
        self.disc_criterion = torch.nn.BCEWithLogitsLoss()
        self.recon_criterion = torch.nn.MSELoss()
        self.curr_epoch = 0

        # ToDO: load checkpoint
        if args.vqvae_checkpoint:
            print(f"Loading checkpoint from {args.vqvae_checkpoint}")
            checkpoint = torch.load(
                args.vqvae_checkpoint, map_location=self.device)
            self.results['n_updates'] = checkpoint['n_updates']
            self.curr_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.discriminator:
                self.discriminator.load_state_dict(
                    checkpoint['discriminator_state_dict'])
                if self.optimizer_disc:
                    self.optimizer_disc.load_state_dict(
                        checkpoint['optimizer_disc_state_dict'])

        self.target_device = self.gpu_id if self.use_ddp else self.device
        if self.use_ddp:
            self.model = DDP(self.model, device_ids=[self.gpu_id])

            # Move optimizer state to the correct device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.gpu_id)

            if self.discriminator and self.optimizer_disc:
                # Move discriminator optimizer state to the correct device
                for state in self.optimizer_disc.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.gpu_id)

            if self.means is not None:
                self.means = self.means.to(self.gpu_id)
            if self.stds is not None:
                self.stds = self.stds.to(self.gpu_id)

        self.best_avg_perplexity = float('-inf')

    def train(self):

        for epoch in range(self.curr_epoch, self.args.n_epochs):
            print(f"Training step {epoch+1}/{self.args.n_epochs}", end='\r')
            self.model.train()

            if isinstance(self.dataloader.sampler, DistributedSampler):
                self.dataloader.sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc=f"Epoch {epoch}")):
                x: torch.Tensor = batch['image'].to(self.target_device)
                if self.means is not None and self.stds is not None:
                    x = x.unsqueeze(1).to(self.target_device)
                    x = (x - self.means) / self.stds

                self.optimizer.zero_grad()

                forward_output = self.model(x)
                x_hat = forward_output.x_hat
                z_q = forward_output.z_q
                perplexity = forward_output.perplexity
                quantize_losses = forward_output.quantize_losses
                encoding_indices = forward_output.encoding_indices

                recon_loss = self.recon_criterion(x_hat, x)
                embedding_loss = quantize_losses['commitment_loss']
                loss = recon_loss + embedding_loss

                # Discriminator on Generator
                if epoch >= self.disc_epoch_start and self.discriminator:
                    disc_fake_pred = self.discriminator(x_hat)
                    disc_fake_loss = self.disc_criterion(disc_fake_pred, torch.ones(
                        disc_fake_pred.shape, device=self.target_device))

                    loss += self.disc_loss_weight * disc_fake_loss
                    self.results['vae_gen_loss'].append(
                        disc_fake_loss.cpu().detach().numpy())

                loss.backward()
                self.optimizer.step()

                # Discriminator
                if epoch >= self.disc_epoch_start and self.discriminator:
                    if self.optimizer_disc:
                        self.optimizer_disc.zero_grad()

                    fake = x_hat.detach()
                    disc_fake_pred = self.discriminator(fake)
                    disc_real_pred = self.discriminator(x)
                    disc_fake_loss = self.disc_criterion(disc_fake_pred, torch.zeros(
                        disc_fake_pred.shape, device=self.target_device))
                    disc_real_loss = self.disc_criterion(disc_real_pred, torch.ones(
                        disc_real_pred.shape, device=self.target_device))

                    disc_loss = self.disc_loss_weight * \
                        (disc_fake_loss + disc_real_loss)
                    self.results['disc_losses'].append(
                        disc_loss.cpu().detach().numpy())

                    disc_loss.backward()
                    if self.optimizer_disc:
                        self.optimizer_disc.step()

                self.results['recon_errors'].append(
                    recon_loss.cpu().detach().numpy())
                self.results['perplexities'].append(
                    perplexity.cpu().detach().numpy())
                self.results['loss_vals'].append(loss.cpu().detach().numpy())
                self.results['n_updates'] = epoch * \
                    len(self.dataloader) + batch_idx
                self.results['embedding_loss'].append(
                    embedding_loss.cpu().detach().numpy())
                self.results['min_encoding_indices'].append(
                    encoding_indices.cpu().detach().numpy())

            # Logging and Saving
            # Saving Checkpoint
            if self.gpu_id == 0 and (epoch % self.args.save_every == 0 or epoch == self.args.n_epochs - 1):
                self._save_checkpoint(epoch, 5)

            # Logging
            training_log = {}
            if self.gpu_id == 0 and (epoch % self.args.log_interval == 0 or epoch == self.args.n_epochs - 1):
                training_log['step'] = self.results['n_updates']
                training_log['recon_error'] = np.mean(
                    self.results['recon_errors'][-self.args.log_interval * len(self.dataloader):])
                training_log['loss'] = np.mean(
                    self.results['loss_vals'][-self.args.log_interval * len(self.dataloader):])
                training_log['perplexity'] = np.mean(
                    self.results['perplexities'][-self.args.log_interval * len(self.dataloader):])
                training_log['embedding_loss'] = np.mean(
                    self.results['embedding_loss'][-self.args.log_interval * len(self.dataloader):])

                disc_str_loss = None
                if epoch >= self.disc_epoch_start and self.discriminator:
                    training_log['vae_gen_loss'] = np.mean(
                        self.results['vae_gen_loss'][-self.args.log_interval * len(self.dataloader):])
                    training_log['disc_losses'] = np.mean(
                        self.results['disc_losses'][-self.args.log_interval * len(self.dataloader):])
                    disc_str_loss = f"VAE Gen Loss: {training_log['vae_gen_loss']}, Disc Losses: {training_log['disc_losses']}"

                print(f"Step {self.results['n_updates']}, Recon Error: {training_log['recon_error']}, Loss: {training_log['loss']}, Perplexity: {training_log['perplexity']}, Embedding Loss: {training_log['embedding_loss']}, Discriminator Loss: {disc_str_loss}")

                if self.args.logtowandb:

                    # Codebook Usage Heatmap
                    codebook_size: int = self.model.codebook_size if hasattr(
                        self.model, 'codebook_size') else self.model.module.codebook_size  # type: ignore
                    recent_indices = self.results['min_encoding_indices'][-self.args.log_interval * len(
                        self.dataloader):]
                    flattened_indices = torch.cat(
                        [torch.as_tensor(x).view(-1) for x in recent_indices])
                    fig = self._plot_codebook_usage_heatmap(
                        flattened_indices, codebook_size)
                    training_log['codebook_usage_heatmap'] = fig

                    # Plotting Samples
                    if not self.args.train_ecgs:
                        img = self._prepare_images(x, x_hat)
                        training_log['fig1'] = img
                    else:
                        x = x * self.stds + self.means if self.stds is not None else x
                        x_hat = x_hat * self.stds + self.means if self.stds is not None else x_hat

                        fig1 = visualizeLeads_comp(x[0].squeeze().detach().cpu(), "Train comparison 1", x_hat[0].squeeze(
                        ).detach().cpu(), f"{self.results_folder}/plots/{self.results['n_updates']}_fig1.png")
                        plt.close()
                        fig2 = visualizeLeads_comp(x[1].squeeze().detach().cpu(), "Train comparison 2", x_hat[1].squeeze(
                        ).detach().cpu(), f"{self.results_folder}/plots/{self.results['n_updates']}_fig2.png")
                        plt.close()
                        training_log['fig1'] = fig1
                        training_log['fig2'] = fig2

                        # Plotting Test Images
                        # idxs = torch.randint(0, len(self.test_dataset), (2,))
                        # inputs = torch.stack(
                        #     [self.test_dataset[ind]['image'].unsqueeze(0) for ind in idxs], dim=0).to(self.device)
                        # norm_inputs = (inputs - self.means) / self.stds
                        # with torch.no_grad():
                        #     outputs = self.model(norm_inputs)
                        # x_hat = outputs.x_hat
                        # x_hat = x_hat * self.stds + self.means

                        # test_fig1 = visualizeLeads_comp(inputs[0].squeeze().detach().cpu(), "Test comparison 1", x_hat[0].squeeze(
                        # ).detach().cpu(), f"{self.results_folder}/plots/{self.results['n_updates']}_test_fig1.png")
                        # plt.close()
                        # test_fig2 = visualizeLeads_comp(inputs[1].squeeze().detach().cpu(), "Test comparison 2", x_hat[1].squeeze(
                        # ).detach().cpu(), f"{self.results_folder}/plots/{self.results['n_updates']}_test_fig2.png")
                        # plt.close()
                        # training_log['test_fig1'] = test_fig1
                        # training_log['test_fig2'] = test_fig2
                        # print("Plots Loaded to Training Dict")

                    wandb.log(training_log)

    def _save_checkpoint(self, epoch: int, save_interval: int = 5):
        print(f"Saving checkpoint at epoch {epoch}")
        if self.use_ddp:
            model_state_dict = self.model.module.state_dict()  # type: ignore
        else:
            model_state_dict = self.model.state_dict()  # type: ignore
        model_state_dict_disc = self.discriminator.state_dict() if self.discriminator else None

        checkpoint = {
            'epoch': epoch,
            'n_updates': self.results['n_updates'],
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'discriminator_state_dict': model_state_dict_disc,
            'optimizer_disc_state_dict': self.optimizer_disc.state_dict() if self.optimizer_disc else None,
            'model_config': self.model_config,
        }
        torch.save(
            checkpoint, f"{self.results_folder}/checkpoint.pt")

        if save_interval > 0 and epoch % save_interval == 0:
            torch.save(
                checkpoint, f"{self.results_folder}/checkpoint_{epoch}.pt")

    def _prepare_images(self, x: torch.Tensor, x_hat: torch.Tensor, n_samples: int = 8):
        x = x.cpu().detach()
        sample_size = min(n_samples, x.shape[0])
        save_output = torch.clamp(x_hat[:sample_size], -1., 1.).detach().cpu()
        save_output = ((save_output + 1) / 2)
        save_input = ((x[:sample_size] + 1) / 2).detach().cpu()

        grid = make_grid(
            torch.cat([save_input, save_output], dim=0), nrow=sample_size)
        img = wandb.Image(grid)
        return img

    def _plot_codebook_usage_heatmap(self, flattened_indices: torch.Tensor, codebook_size: int):
        usage = torch.bincount(
            flattened_indices, minlength=codebook_size).float()  # type: ignore
        usage_ratio = usage / usage.sum()

        fig, axs = plt.subplots(1, 1, figsize=(10, 1))
        im = axs.imshow(usage_ratio.view(1, -1), cmap="viridis", aspect="auto")

        axs.set_title("Codebook Usage Heatmap")
        axs.set_xlabel("Codebook Index")
        axs.set_yticks([])

        cbar = fig.colorbar(im, ax=axs, orientation="vertical", pad=0.01)
        cbar.set_label("Usage Ratio")
        img = wandb.Image(fig)
        plt.close()
        return img
