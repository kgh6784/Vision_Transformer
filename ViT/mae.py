import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from ViT import Transformer


class MAE(nn.Module):
    def __init__(
            self,
            *,
            encoder,
            decoder_dim,
            masking_ratio=0.75,
            decoder_depth=1,
            decoder_heads=8,
            decoder_dim_head=64
    ):
        super().__init__()
        assert masking_ratio>0 and masking_ratio<1, 'masking ration must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:] # nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2] # (1) Rearrange b (h w) (p1 p2 c) || (2) Linear
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]  # patch당 픽셀이 몇개 있는지?

        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim!=decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads,
                                   dim_head = decoder_dim_head, mlp_dim = decoder_dim*4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self,img):
        device = img.device

        #get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches+1)]

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        batch_range = torch.arange(batch, devicde=device)[:,None]
        tokens = tokens[batch_range, unmasked_indices]

        masked_patches = patches[batch_range, masked_indices]

        encoded_tokens = self.encoder.transformer(tokens)

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)

        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)

        return recon_loss