import os
import yaml
import torch
from scheduler import LinearNoiseScheduler, CosineNoiseScheduler
from utils.text_utils import get_text_representation, get_tokenizer_and_model
from functools import partial
from models.unet import Unet
from models.vqvae_models.ecg.vqvae import VQVAEECG


class CounterfactualMachine:
    def __init__(self, config_path, stats_path, datasetFunc, ldm_checkpoint_path, vae_checkpoint_path):
        self.ldm_checkpoint_path = ldm_checkpoint_path
        self.vae_checkpoint_path = vae_checkpoint_path
        self.config_path = config_path
        self.config = None
        with open(self.config_path, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
                return
        self.config = config

        # load means stds
        stats = torch.load(stats_path, weights_only=False, map_location="cpu")
        self.means = stats['means']
        self.stds = stats['stds']

        # Gather Configs
        self.diffusion_config = config['diffusion_params']
        self.dataset_config = config['dataset_config']
        self.diffusion_model_config = config['ldm_config']
        self.autoencoder_model_config = config['autoencoder_config']

        # Noise Scheduler
        self.scheduler = None
        self._prepare_scheduler()

        # Condition Related
        self.text_tokenizer = None
        self.text_model = None
        self.empty_text_embed = None
        self.condition_types = []
        self._prepare_conditioning()

        # Dataset related things
        self.train_dataset = None
        self.val_dataset = None
        self._load_dataset(self, partial(datasetFunc, self.dataset_config))

        # Loading Models
        self.ldm = None
        self.vqvae = None
        self._prepare_models()

    def _prepare_scheduler(self):
        if self.train_config['noise_schedule'] == "cosine":
            self.scheduler = CosineNoiseScheduler(
                num_timesteps=self.diffusion_config['num_timesteps'])
        else:
            self.scheduler = LinearNoiseScheduler(num_timesteps=self.diffusion_config['num_timesteps'],
                                                  beta_start=self.diffusion_config['beta_start'],
                                                  beta_end=self.diffusion_config['beta_end'])

    def _prepare_conditioning(self):
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

    def _prepare_models(self):
        self.ldm = Unet(
            channels=self.diffusion_model_config['in_channels'], model_config=self.diffusion_model_config)
        self.vae = VQVAEECG(model_config=self.autoencoder_model_config)

        assert self.ldm_checkpoint_path is not None and self.vae_checkpoint_path is not None, "Checkpoint path not provided"
        assert os.path.exists(self.ldm_checkpoint_path) and os.path.exists(
            self.vae_checkpoint_path)

        ldm_checkpoint = torch.load(
            self.ldm_checkpoint_path, map_location="cpu", weights_only=False)
        vae_checkpoint = torch.load(
            self.vae_checkpoint_path, map_location="cpu", weights_only=False)

        self.ldm.load_state_dict(ldm_checkpoint['model_state_dict'])
        self.ldm.eval()
        self.vae.load_state_dict(vae_checkpoint['model_state_dict'])
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

    def _load_dataset(self, getDatasetFunc):
        self.train_dataset, self.val_dataset = getDatasetFunc()

    def _normalize_ecg(self):
        pass

    def _generate_counterfactual(self):
        pass
