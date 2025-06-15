import os
import datetime
import sys
sys.path.append("..")  

import numpy as np
import torch
import torch.optim as optim
from vq_vae.vqvae import VQVAE
from models.unet import Unet
from linear_noise_schedule import LinearNoiseScheduler
from utils.text_utils import get_tokenizer_and_model, get_text_representation
from diff_utils import drop_text_condition, drop_class_condition
from data.dataset import getKCLTrainTestDataset
from torch.utils.data import DataLoader
from sampling.sampling_utils import diff_random_sample
from utils.plot_utils import visualizeLeads_comp
import matplotlib.pyplot as plt

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import wandb
import yaml
from tqdm import tqdm

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
logtowandb = True

def train(config):
    use_ddp = "LOCAL_RANK" in os.environ
    
    if use_ddp:
        gpu_id = int(os.environ["LOCAL_RANK"])
        
    else:
        gpu_id = 0
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_config']
    diffusion_model_config = config['ldm_config']
    autoencoder_model_config = config['autoencoder_config']
    train_config = config['train_params']
    means = dataset_config['means']
    stds = dataset_config['stds']
    
    #### Noise Scheduler #####
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    
    # Condition Related Components
    text_tokenizer = None
    text_model = None
    empty_text_embed = None
    condition_types = []
    condition_config = diffusion_model_config.get("condition_config", None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, "condition types missing in condition config"
        condition_types = condition_config['condition_types']
        if 'text' in condition_types:
            with torch.no_grad():
                text_tokenizer, text_model = get_tokenizer_and_model(
                    condition_config['text_condition_config']['text_embed_model'], device=device)
                empty_text_embed = get_text_representation([''], text_tokenizer, text_model, device)
    
    # Load Dataset and stuff
    dataset, val_dataset  = getKCLTrainTestDataset(dataset_config)
    dataloader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=False, drop_last=True,
                            pin_memory=True, sampler=DistributedSampler(dataset, drop_last=True, shuffle=True) if use_ddp else None)
    
    model = Unet(channels=diffusion_model_config['in_channels'], model_config=diffusion_model_config).to(device)
    
    if use_ddp:
        model = DDP(model, device_ids=[device])
        means = means.to(device)
        stds = stds.to(device)
    
    model.train()
    
    vae = None
    if not dataset.use_latents:
        print('Loading vqvae as latents not present')
        vae = VQVAE(model_config=autoencoder_model_config).to(device)
        vae.eval()
        
        if os.path.exists(train_config['vqvae_autoencoder_ckpt_name']):
            vae.load_state_dict(torch.load(train_config['vqvae_autoencoder_ckpt_name'], weights_only=True, map_location=device)['model_state_dict'])
        else:
            raise Exception('VAE checkpoint not found')
        
    num_epochs = train_config['epochs']
    optimizer = optim.Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()
    
    if not dataset.use_latents:
        assert vae is not None
        for param in vae.parameters():
            param.requires_grad = False
    
    
    for epoch_idx in range(num_epochs):
        model.train()
        
        losses = []
        for data in tqdm(dataloader):
            cond_input = None
            if condition_config is not None:
                im, cond_input = data['image'], data['cond_inputs']
            else:
                im = data['image']
            optimizer.zero_grad()
            im = im.float().to(device)
            if not dataset.use_latents:
                with torch.no_grad():
                    im = (im - means) / stds
                    _, im, _ = vae.encode(im)
            
            ##### Conditional Inputs #####
            if 'text' in condition_types:
                with torch.no_grad():
                    assert 'text' in cond_input, 'Conditioning Type Text but no text conditing input present'
                    text_condition = get_text_representation(cond_input['text'],
                                                             text_tokenizer,
                                                             text_model,
                                                             device)
                    text_drop_prob = condition_config['text_condition_config'].get('cond_drop_prob', 0.)
                    text_condition = drop_text_condition(text_condition, im, empty_text_embed, text_drop_prob)
                    cond_input['text'] = text_condition
            
            if 'class' in condition_types:
                assert 'class' in cond_input, 'Conditioning Type Class but no class conditioning input present'
                class_condition = torch.nn.functional.one_hot(
                    cond_input['class'],
                    condition_config['class_condition_config']['num_classes']
                ).to(device)
                class_drop_prob = condition_config['class_condition_config'].get('cond_drop_prob', 0.)
                
                cond_input['class'] = drop_class_condition(class_condition, class_drop_prob, im)
            
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t, cond_input=cond_input)
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        
        training_log = dict(
                step = epoch_idx,
                loss = np.mean(losses)
        )
            
        
        if (gpu_id == 0 or device == torch.device("cpu")) and logtowandb and epoch_idx % 1 == 0:
            print(f'Finished epoch: {epoch_idx + 1} | Loss : {np.mean(losses):.4f}')
            ims, text_prompts = diff_random_sample(model, vae, scheduler, train_config,
                                     diffusion_model_config, autoencoder_model_config,
                                     diffusion_config, dataset_config, text_tokenizer,
                                     text_model, device=gpu_id)
            
            fig1 = visualizeLeads_comp(ims[0].squeeze().detach().cpu(), text_prompts[0], ims[0].squeeze().detach().cpu(), f"{train_config['results_folder']}/plots/{epoch_idx}_fig1.png")
            plt.close()
            fig2 = visualizeLeads_comp(ims[1].squeeze().detach().cpu(), text_prompts[1], ims[1].squeeze().detach().cpu(), f"{train_config['results_folder']}/plots/{epoch_idx}_fig2.png")
            plt.close()
            training_log['fig1'] = fig1
            training_log['fig2'] = fig2
            
            wandb.log(training_log)
            torch.save(model.state_dict(), f"{train_config['results_folder']}/checkpoint.pt")
    
    print('Done Training ...')


def main():
    config_path = "/uu/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg_latent_diff/configs/diff.yaml"
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
    
    stats = torch.load('/uu/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg_latent_diff/data/ecg_train_stats.pt', weights_only=True, map_location="cpu")
    means = stats['mean'].to(gpu_id)
    stds = stats['std'].to(gpu_id)
    
    
    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = f"./results/latent_{formatted_time}"
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(f"{results_folder}/plots", exist_ok=True)
    config['train_params']['results_folder'] = results_folder
    config['dataset_config']['means'] = means
    config['dataset_config']['stds'] = stds
    
    
    if (gpu_id == 0 or gpu_id == torch.device("cpu")):
        print(config)
        print(f"Saving Results @{results_folder}")
        print(f"GPU_ID: {gpu_id}")
    
    if (gpu_id == 0 or gpu_id == torch.device("cpu")) and logtowandb:
        wandbrun = wandb.init(
            project = "latent_ecg",
            notes = f"diffusion in latent space",
            tags= ["latent", "diffusion"],
            entity="deekshith",
            reinit=True,
            config=config,
            name=f"{"latent"}_{formatted_time}"
        )
    
    train(config)       
    if (gpu_id == 0 or gpu_id == torch.device("cpu")) and logtowandb:
        wandbrun.finish() 
    
    if "LOCAL_RANK" in os.environ:
        destroy_process_group()

if __name__ == "__main__":
    main()

