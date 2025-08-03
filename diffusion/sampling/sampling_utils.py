import torch
from utils.text_utils import get_text_representation
from tqdm import tqdm


def diff_random_sample(model, vae, scheduler, train_config, diffusion_model_config,
                       autoencoder_model_config, diffusion_config, dataset_config,
                       text_tokenizer, text_model, device):

    model.eval()
    vae.eval()
    channels = autoencoder_model_config['embedding_dim']
    # channels = autoencoder_model_config['z_channels']
    n_examples = 2
    xt = torch.randn(n_examples,
                     channels,
                     8, 312).to(device)
    # xt = torch.randn(n_examples, channels, 32, 32).to(device)

    # text_prompt = ['She is a woman with blond hair. She is wearing lipstick.'] * (n_examples // 2)
    # text_prompt += ['He is a man with black hair. He has a moustache.'] * (n_examples // 2)
    # text_prompt = ["Normal Signs of Hyperkalemia", "Severe Stage Hyperkalemia"]
    # empty_prompt = [''] * len(text_prompt)
    # text_prompt_embed = get_text_representation(text_prompt,
    #                                             text_tokenizer,
    #                                             text_model,
    #                                             device)
    # empty_text_embed = get_text_representation(empty_prompt,
    #                                            text_tokenizer,
    #                                            text_model,
    #                                            device)
    # assert empty_text_embed.shape == text_prompt_embed.shape

    # uncond_input = {
    #     'text': empty_text_embed
    # }

    # cond_input = {
    #     'text': text_prompt_embed
    # }

    text_prompt = ["class 0", "class 1"]
    class_condition = torch.nn.functional.one_hot(
        torch.tensor([0, 1]),
        2
    ).to(device)
    cond_input = {
        'class': class_condition
    }
    uncond_input = {
        'class': cond_input['class'] * 0
    }
    cf_guidance_scale = train_config.get('cf_guidance_scale', 2.0)

    # Sampling
    with torch.no_grad():
        for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
            t = (torch.ones((xt.shape[0],)) * i).long().to(device)
            noise_pred_cond = model(xt, t, cond_input)

            if cf_guidance_scale > 1:
                noise_pred_uncond = model(xt, t, uncond_input)
                noise_pred = noise_pred_uncond + cf_guidance_scale * \
                    (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond

            xt, x0_pred = scheduler.sample_prev_timestep(
                xt, noise_pred, torch.as_tensor(i).to(device))

            if i == 0:
                decoder_output = vae.decode(x0_pred / 5.0)
                ims = decoder_output.x_hat
            else:
                ims = x0_pred

    means = dataset_config['means'].to(ims.device)
    stds = dataset_config['stds'].to(ims.device)

    ims = ims * stds + means

    # ims = torch.clamp(ims, -1., 1.).detach().cpu()
    # ims = (ims + 1) / 2

    return ims, text_prompt
