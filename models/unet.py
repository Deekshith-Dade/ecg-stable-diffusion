import torch.nn as nn
import torch
from einops import einsum
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.components import (SinusoidalPosEmb, 
                               ResnetBlock,
                               Residual,
                               PreNorm,
                               Attention,
                               CrossAttention,
                               Downsample,
                               Upsample)

class Unet(nn.Module):
    def __init__(
        self,
        channels,
        model_config
    ):
        super().__init__()
        dim = model_config["dim"]
        init_dim = model_config["init_dim"]
        dim_mults = model_config["dim_mults"]
        attn_dim_head = model_config["attn_dim_head"]
        attn_heads = model_config["attn_heads"]
        
        self.class_cond = False
        self.text_cond = False
        self.text_embed_dim = None
        self.condition_config = model_config["condition_config"]
        if self.condition_config is not None:
            assert 'condition_types' in self.condition_config, "conditon type not provided"
            condition_types = self.condition_config['condition_types']
            if 'class' in condition_types:
                self.class_cond = True
                self.num_classes = self.condition_config['class_condition_config']['num_classes']
            if 'text' in condition_types:
                self.text_cond = True
                self.text_embed_dim = self.condition_config['text_condition_config']['text_embed_dim']
        self.cond = self.class_cond or self.text_cond
        
        self.channels = channels
        input_channels = channels
        
        self.init_dim = init_dim
        self.init_conv = nn.Conv2d(input_channels, init_dim, (1,7), padding=(0,3))
        
        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        self.time_dim = dim * 4
        
        sinus_pos_emb = SinusoidalPosEmb(dim)
        
        self.time_mlp = nn.Sequential(
            sinus_pos_emb,
            nn.Linear(dim, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim)
        )
        
        if self.class_cond:
            self.classes_emb = nn.Embedding(self.num_classes, dim)
            self.classes_dim = dim * 4
            self.classes_mlp = nn.Sequential(
                nn.Linear(dim, self.classes_dim),
                nn.GELU(),
                nn.Linear(self.classes_dim, self.classes_dim)
            )
        
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim=self.time_dim, classes_emb_dim=self.classes_dim if self.class_cond else None),
                ResnetBlock(dim_in, dim_in, time_emb_dim=self.time_dim, classes_emb_dim=self.classes_dim if self.class_cond else None),
                Residual(PreNorm(dim_in, Attention(dim_in, heads=attn_heads, dim_head=attn_dim_head))),
                Residual(PreNorm(dim_in, CrossAttention(dim_in, context_dim=self.text_embed_dim, heads=attn_heads, dim_head=attn_dim_head ))) if self.text_cond else nn.Identity(),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, (1, 3), padding=(0, 1))
            ]))
        
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = self.time_dim, classes_emb_dim = self.classes_dim if self.class_cond else None)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = attn_heads)))
        self.mid_xattn = Residual(PreNorm(mid_dim, CrossAttention(mid_dim, context_dim=self.text_embed_dim, dim_head = attn_dim_head, heads = attn_heads))) if self.text_cond else nn.Identity()
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = self.time_dim, classes_emb_dim = self.classes_dim if self.class_cond else None)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = self.time_dim, classes_emb_dim = self.classes_dim if self.class_cond else None),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = self.time_dim, classes_emb_dim = self.classes_dim if self.class_cond else None),
            Residual(PreNorm(dim_out, Attention(dim_out, heads=attn_heads, dim_head=attn_dim_head))),
                Residual(PreNorm(dim_out, CrossAttention(dim_out, context_dim=self.text_embed_dim, heads=attn_heads, dim_head=attn_dim_head ))) if self.text_cond else nn.Identity(),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, (1, 3), padding = (0, 1))
            ]))
        
        self.out_dim = channels
        
        self.final_res_block = ResnetBlock(init_dim * 2, init_dim, time_emb_dim=self.time_dim, classes_emb_dim=self.classes_dim if self.class_cond else None)
        self.norm_out = nn.GroupNorm(int(self.init_dim / 4), self.init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)
    
    def forward(
        self,
        x,
        time,
        cond_input=None,
    ):
        
        if self.cond:
            assert cond_input is not None, "Model initialized with conditioning so cond_input cannot be None"
        
        batch, device = x.shape[0], x.device
        if len(x.shape) != 4:
            raise ValueError(f"Expected input tensor to have 4 dimensions (batch, channels, height, width), but got {len(x.shape)} dimensions with shape {x.shape}")
        
        if time.shape != (batch,):
            time = time.reshape(batch)
        
        t_emb = self.time_mlp(time)
        
        class_emb = None
        if self.class_cond:
            assert 'class' in cond_input, "Model initialized with class conditioning but cond_input has no text information"
            class_emb = einsum(cond_input['class'].float(), self.classes_emb.weight, 'b n, n d -> b d')
            class_emb = self.classes_mlp(class_emb)
             # t_emb += class_emb class embedding can be added two ways either pass it to the block or add to t_emb itself
        
        context=None
        if self.text_cond:
            assert 'text' in cond_input, "Model initialized with text conditioning but cond_input has no text information"
            context = cond_input['text']
        
        
        x = self.init_conv(x)
        r = x.clone()
        
        h = []
        for block1, block2, attn, xattn, downsample in self.downs:
            x = block1(x, t_emb, class_emb) 
            h.append(x)
            x = block2(x, t_emb, class_emb)
            x = attn(x)
            x = xattn(x, context=context) if self.text_cond else xattn(x)
            h.append(x)
            
            x = downsample(x)
        
        x = self.mid_block1(x, t_emb, class_emb)
        x = self.mid_attn(x)
        x = self.mid_xattn(x, context=context) if self.text_cond else self.mid_xattn(x)
        x = self.mid_block2(x, t_emb, class_emb)
        
        for block1, block2, attn, xattn, upsample in self.ups:
            hid = h.pop()
            x = torch.cat((x, hid), dim = 1)
            x = block1(x, t_emb, class_emb)
            
            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t_emb, class_emb)
            x = attn(x)
            x = xattn(x, context=context) if self.text_cond else xattn(x)
            
            x = upsample(x)
        
        x = torch.cat((x, r), dim = 1)
        
        x = self.final_res_block(x, t_emb, class_emb)
        x = self.norm_out(x)  
        return self.final_conv(x)    


if __name__ == "__main__":
    
    import yaml
    
    path = "/uu/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg_latent_diff/configs/diff.yaml"
    with open(path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    
    x = torch.randn(4, 8, 8, 312)
    t = torch.randint(0, 1000, (4,))
    cond_input = {
        "class": torch.randint(0, 2, (4,)),
        "text": torch.randn(4, 77, 768)
    }
    
    
    
    class_condition = torch.nn.functional.one_hot(
                    cond_input['class'],
                    2)
    cond_input['class'] = class_condition
    
    model = Unet(channels=config['ldm_config']['in_channels'], model_config=config['ldm_config'])
    
    out = model(x, t, cond_input=cond_input)
    
    print(f"Output Shape: {out.shape}")