import os
import numpy as np
import torch

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import torch.optim as optim

from dataset.dataset import getKCLTrainTestDataset
from diffusion.scheduler import LinearNoiseScheduler, CosineNoiseScheduler
from utils.plot_utils import visualizeLeads_comp
from utils.text_utils import get_text_representation, get_tokenizer_and_model

from models.unet import Unet
from vqvae_models.ecg.vqvae import VQVAEECG
from diff_utils import drop_text_condition, drop_class_condition

from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt


class Training:
    def __init__(self, config):
        self.use_ddp = "LOCAL_RANK" in os.environ

        if self.use_ddp:
            self.gpu_id = int(os.environ["LOCAL_RANK"])
        else:
            self.gpu_id = 0
        self.config = config

        self.device = torch.device(
            f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.diffusion_config = config['diffusion_params']
        self.dataset_config = config['dataset_config']
        self.diffusion_model_config = config['ldm_config']
        self.autoencoder_model_config = config['autoencoder_config']
        self.train_config = config['train_params']
        self.means = self.dataset_config['means']
        self.stds = self.dataset_config['stds']
        self.logtowandb = config['logtowandb']

        # Noise Scheduler
        if self.train_config['noise_schedule'] == "cosine":
            self.scheduler = CosineNoiseScheduler(
                num_timesteps=self.diffusion_config['num_timesteps'])
        else:
            self.scheduler = LinearNoiseScheduler(num_timesteps=self.diffusion_config['num_timesteps'],
                                                  beta_start=self.diffusion_config['beta_start'],
                                                  beta_end=self.diffusion_config['beta_end'])

        # Condition Related
        self.text_tokenizer = None
        self.text_model = None
        self.empty_text_embed = None
        self.condition_types = []
        self.condition_config = self.diffusion_model_config.get(
            "condition_config", None)
        if self.condition_config is not None:
            assert 'condition_types' in self.condition_config, "condition types missing in condition config"
            self.condition_types = self.condition_config['condition_types']
            if 'text' in self.condition_types:
                with torch.no_grad():
                    self.text_tokenizer, self.text_model = get_tokenizer_and_model(
                        self.condition_config['text_condition_config']['text_embed_model'], device=self.device
                    )
                    self.empty_text_embed = get_text_representation(
                        [''], self.text_tokenizer, self.text_model, self.device
                    )

        self.im_dataset, _ = getKCLTrainTestDataset(self.dataset_config)
        self.sampler = DistributedSampler(
            self.im_dataset, drop_last=True, shuffle=True) if self.use_ddp else None
        self.dataloader = DataLoader(self.im_dataset, batch_size=self.train_config['batch_size'], shuffle=False, drop_last=True,
                                     pin_memory=True, num_workers=4, sampler=self.sampler)

        self.model = Unet(channels=self.diffusion_model_config['in_channels'],
                          model_config=self.diffusion_model_config).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.train_config['ldm_lr'])
        self.start_epoch = 0
        self.vqvae_mean = None
        self.vqvae_std = None

        # load previous vqvae model if available
        self.vae = VQVAEECG(
            model_config=self.autoencoder_model_config).to(self.device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        if os.path.exists(self.train_config['vqvae_autoencoder_ckpt_name']):
            self.vae.load_state_dict(torch.load(
                self.train_config['vqvae_autoencoder_ckpt_name'], map_location=self.device, weights_only=False
            )['model_state_dict'])
            print("Loaded VQVAE Checkpoint")
        else:
            raise Exception('VAE Checkpoint not Found')

        # calculating latent means and stds
        self.calculate_latent_mean_std_ddp()
        if self.use_ddp:
            dist.barrier()

        # Loading LDM if available
        if self.train_config['checkpoint_path'] is not None:
            if os.path.exists(self.train_config['checkpoint_path']):
                checkpoint = torch.load(
                    self.train_config['checkpoint_path'], map_location=self.device, weights_only=False
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                print('Loaded Model and Optimizer Checkpoint')
            else:
                print('Model Not found at the given path')
                return
        else:
            print('Starting Without checkpoint')

        if self.use_ddp:
            self.model = DDP(self.model, device_ids=[self.device])
            self.means = self.means.to(self.device)
            self.stds = self.stds.to(self.device)

        self.model.train()

        self.num_epochs = self.train_config['epochs']
        self.criterion = torch.nn.functional.mse_loss

    def calculate_latent_mean_std(self):
        self.vae.eval()

        latents_sum = None
        latents_sq_sum = None
        count = 0
        with torch.no_grad():
            for batch in tqdm(self.normal_dataloader, desc="Calculating Latent Mean and Std", disable=(self.gpu_id != 0)):

                x = batch['image'].to(self.device)
                encoder_output = self.vae.encode(x)
                latents = encoder_output.quantize_output.z_q

                b, c = latents.shape[0], latents.shape[1]
                latents_flat = latents.view(b, c, -1)

                if latents_sum is None:
                    latents_sum = latents_flat.sum(dim=(0, 2))
                    latents_sq_sum = (latents_flat ** 2).sum(dim=(0, 2))
                else:
                    latents_sum += latents_flat.sum(dim=(0, 2))
                    latents_sq_sum += (latents_flat ** 2).sum(dim=(0, 2))
                count += b * latents_flat.shape[2]

        mean = latents_sum / count
        var = (latents_sq_sum / count) - (mean ** 2)
        std = torch.sqrt(var)

        self.vqvae_mean = mean.view(1, -1, 1, 1).to(self.device)
        self.vqvae_std = std.view(1, -1, 1, 1).to(self.device)

        print("VQ-VAE Latent Mean:", self.vqvae_mean)
        print("VQ-VAE Latent Std:", self.vqvae_std)

    def calculate_latent_mean_std_ddp(self):
        self.vae.eval()

        use_ddp = getattr(self, "use_ddp", False)
        ddp_active = use_ddp and dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if ddp_active else 0

        local_sum = None
        local_sqsum = None
        local_count = 0
        spatial_ndims = None

        with torch.no_grad():
            iterator = self.dataloader
            if rank == 0:
                pbar = tqdm(iterator, desc="Calculating Latent Mean and Std")
            else:
                pbar = iterator

            for batch in pbar:
                x = batch['image'].to(self.device, non_blocking=True)

                enc_out = self.vae.encode(x)

                latents = enc_out.quantize_output.z_q

                B, C = latents.shape[:2]
                spatial_ndims = latents.dim() - 2
                latents_flat = latents.view(B, C, -1)

                s = latents_flat.sum(dim=(0, 2))
                q = (latents_flat ** 2).sum(dim=(0, 2))
                n = B * latents_flat.shape[2]

                if local_sum is None:
                    local_sum = s
                    local_sqsum = q
                else:
                    local_sum += s
                    local_sqsum += q
                local_count += n

        if local_sum is None:
            raise RuntimeError("No data was processed. Check dataloader")

        sum_t = local_sum.to(self.device, dtype=torch.float64)
        sqsum_t = local_sqsum.to(self.device, dtype=torch.float64)
        cnt_t = torch.tensor(
            [local_count], device=self.device, dtype=torch.long)

        if ddp_active:
            dist.all_reduce(sum_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(sqsum_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnt_t, op=dist.ReduceOp.SUM)

        total_count = cnt_t.item()
        mean = sum_t / total_count

        var = (sqsum_t / total_count) - mean ** 2
        std = torch.sqrt(torch.clamp(var, min=1e-12))

        final_shape = (1, -1) + (1, ) * \
            (spatial_ndims if spatial_ndims is not None else 2)
        self.vqvae_mean = mean.view(*final_shape).to(self.device, x.dtype)
        self.vqvae_std = std.view(*final_shape).to(self.device, x.dtype)

        if rank == 0:
            print(
                f"VQ-VAE Latent Mean (first 5 channels): {self.vqvae_mean.flatten()[:5]}")
            print(
                f"VQ-VAE Latent Std (first 5 channels): {self.vqvae_std.flatten()[:5]}")

    def normalize_latents(self, latents):
        return (latents - self.vqvae_mean) / self.vqvae_std

    def denormalize_latents(self, latents):
        return latents * self.vqvae_std + self.vqvae_mean

    def diff_random_sample(self):
        self.model.eval()
        self.vae.eval()

        channels = self.autoencoder_model_config['embedding_dim']
        n_examples = 2
        xt = torch.randn(n_examples, channels, 8, 312).to(self.device)

        uncond_input = {}
        cond_input = {}
        text_prompt = None
        if 'text' in self.condition_types:
            text_prompt = ["Normal signs of Hyperkalemia",
                           "Severe Stage Hyperkalemia"]
            empty_prompt = [''] * len(text_prompt)
            text_prompt_embed = get_text_representation(
                text_prompt, self.text_tokenizer, self.text_model, self.device)
            empty_text_embed = get_text_representation(
                empty_prompt, self.text_tokenizer, self.text_model, self.device)
            assert empty_text_embed.shape == text_prompt_embed.shape
            uncond_input['text'] = empty_text_embed
            cond_input['text'] = text_prompt_embed

        if 'class' in self.condition_types:
            if text_prompt is None:
                text_prompt = ['class 0', 'class 1']
            class_condition = torch.nn.functional.one_hot(
                torch.tensor([0, 1]),
                2
            ).to(self.device)
            cond_input['class'] = class_condition
            uncond_input['class'] = cond_input['class'] * 0

        cf_guidance_scale = self.train_config.get('cf_guidance_scale', 2.0)

        # Sampling
        with torch.no_grad():
            for i in tqdm(reversed(range(self.diffusion_config['num_timesteps'])), disable=(self.gpu_id != 0)):
                t = (torch.ones((xt.shape[0],)) * i).long().to(self.device)
                noise_pred_cond = self.model(xt, t, cond_input)

                if cf_guidance_scale > 1:
                    noise_pred_uncond = self.model(xt, t, uncond_input)
                    noise_pred = noise_pred_uncond + cf_guidance_scale * \
                        (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                xt, x0_pred = self.scheduler.sample_prev_timestep(
                    xt, noise_pred, torch.as_tensor(i).to(self.device)
                )

                if i == 0:
                    decoder_output = self.vae.decode(
                        self.denormalize_latents(x0_pred))
                    ims = decoder_output.x_hat
                else:
                    ims = x0_pred

        ims = ims * self.stds + self.means

        return ims, text_prompt

    def train(self):

        for epoch_idx in range(self.start_epoch, self.num_epochs):
            self.model.train()

            losses = []
            perplexities = []

            for data in tqdm(self.dataloader, disable=(self.gpu_id != 0)):
                cond_input = None
                if self.condition_config is not None:
                    im, cond_input = data['image'], data['cond_inputs']
                else:
                    im = data

                self.optimizer.zero_grad()

                im = im.float().to(self.device)

                with torch.no_grad():
                    im = (im - self.means) / self.stds

                    encoder_output = self.vae.encode(im)
                    im = encoder_output.quantize_output.z_q
                    im = self.normalize_latents(im)
                    perplexity = encoder_output.quantize_output.perplexity
                    perplexities.append(perplexity.item())

                #### Preparing Condition Inputs ####
                if 'text' in self.condition_types:
                    with torch.no_grad():
                        assert 'text' in cond_input, 'Conditioning Type Text but no text condition input present'
                        text_condition = get_text_representation(cond_input['text'],
                                                                 self.text_tokenizer,
                                                                 self.text_model,
                                                                 self.device)
                        text_drop_prob = self.condition_config['text_condition_config'].get(
                            'cond_drop_prob', 0.)
                        text_condition = drop_text_condition(
                            text_condition, im, self.empty_text_embed, text_drop_prob)
                        cond_input['text'] = text_condition

                if 'class' in self.condition_types:
                    assert 'class' in cond_input, 'Conditioning Type class but no class conditioning input present'
                    class_condition = torch.nn.functional.one_hot(
                        cond_input['class'],
                        self.condition_config['class_condition_config']['num_classes']
                    ).to(self.device)
                    class_drop_prob = self.condition_config['class_condition_config'].get(
                        'cond_drop_prob', 0.)
                    cond_input['class'] = drop_class_condition(
                        class_condition, class_drop_prob, im)
                ### Condition Inputs Prepared ###

                noise = torch.randn_like(im).to(self.device)

                t = torch.randint(
                    0, self.diffusion_config['num_timesteps'], (im.shape[0],)).to(self.device)
                noisy_im = self.scheduler.add_noise(im, noise, t)
                noise_pred = self.model(noisy_im, t, cond_input=cond_input)
                loss = self.criterion(noise_pred, noise)
                losses.append(loss.item())

                loss.backward()
                self.optimizer.step()

            # Post Epoch Logging
            training_log = dict(
                step=epoch_idx,
                loss=np.mean(losses),
                perplexity=np.mean(perplexities) if perplexities else 0
            )

            if (self.gpu_id == 0 or self.device == torch.device("cpu")) and self.logtowandb and epoch_idx % 1 == 0:
                print(
                    f'Finished epoch: {epoch_idx + 1} | Loss: {np.mean(losses):.4f} | Perplexity : {training_log['perplexity']:.4f}'
                )
                ims, text_prompts = self.diff_random_sample()

                fig1 = visualizeLeads_comp(ims[0].squeeze().detach().cpu(), text_prompts[0],
                                           ims[0].squeeze().detach().cpu(), f"{self.train_config['results_folder']}/plots/{epoch_idx}_fig1.png")
                plt.close()
                fig2 = visualizeLeads_comp(ims[1].squeeze().detach().cpu(), text_prompts[1],
                                           ims[1].squeeze().detach().cpu(), f"{self.train_config['results_folder']}/plots/{epoch_idx}_fig1.png")
                plt.close()
                training_log['fig1'] = fig1
                training_log['fig2'] = fig2

                wandb.log(training_log)
                checkpoint = {
                    'model_state_dict': self.model.module.state_dict() if self.use_ddp else self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch_idx,
                    'loss': np.mean(losses),
                    'config': self.config,
                    'vqvae_mean': self.vqvae_mean,
                    'vqvae_std': self.vqvae_std,
                }

                torch.save(
                    checkpoint, f"{self.train_config['results_folder']}/checkpoint.pt")
                if epoch_idx % 10 == 0:
                    torch.save(
                        checkpoint, f"{self.train_config['results_folder']}/checkpoint_{epoch_idx}.pt")

        print('Done Training.......')
