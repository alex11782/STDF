import torch
import torch.nn as nn
from transformers import CLIPTextModel

from models.adnb import ADNB
from models.sketch_encoder import RobustSketchEncoder


class STDFNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sketch_encoder = RobustSketchEncoder(
            output_dim=config.SKETCH_EMBED_DIM,
            backbone=config.SKETCH_BACKBONE,
            use_mixstyle=config.USE_MIXSTYLE,
        )
        self.text_encoder = CLIPTextModel.from_pretrained(config.CLIP_MODEL, use_safetensors=True)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.coord_embed = nn.Linear(3, config.HIDDEN_DIM)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
        )
        self.frame_mlp = nn.Sequential(
            nn.Linear(1, config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
        )
        self.vertex_embed = nn.Parameter(torch.empty(1, config.N_VERTICES, config.HIDDEN_DIM))
        nn.init.normal_(self.vertex_embed, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            [
                ADNB(
                    dim=config.HIDDEN_DIM,
                    n_heads=config.N_HEADS,
                    k_eig=config.K_EIG,
                    text_dim=config.TEXT_EMBED_DIM,
                    sketch_dim=config.SKETCH_EMBED_DIM,
                    use_spectral_diffusion=config.USE_SPECTRAL_DIFFUSION,
                    use_dual_path_attention=config.USE_DUAL_PATH_ATTENTION,
                )
                for _ in range(config.N_BLOCKS)
            ]
        )
        self.out_head = nn.Linear(config.HIDDEN_DIM, 3)

    def forward(self, x_t, t, sketch, input_ids, evals, evecs, frame_t=None):
        batch, _, _ = x_t.shape
        sketch_feat = self.sketch_encoder(sketch)
        with torch.no_grad():
            text_feat = self.text_encoder(input_ids)[0]

        h = self.coord_embed(x_t)
        h = h + self.vertex_embed
        h = h + self.time_mlp(t.view(batch, 1)).unsqueeze(1)
        if frame_t is not None:
            h = h + self.frame_mlp(frame_t.view(batch, 1)).unsqueeze(1)
        for block in self.blocks:
            h = block(h, evals, evecs, text_feat, sketch_feat)
        return self.out_head(h)
