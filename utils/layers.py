import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import torch.nn.functional as F
from monai.networks.blocks.unetr_block import UnetrUpBlock

class HGNN_layer(nn.Module):
    """
        Writen by Shaocong Mo,
        College of Computer Science and Technology, Zhejiang University,
    """

    def __init__(self, in_ch, embed_dim = 768):
        super(HGNN_layer, self).__init__()




        self.emb1= nn.Sequential(
            nn.Conv1d(embed_dim,in_ch,kernel_size=1,stride=1),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(),
        )

        self.out = nn.Linear(embed_dim, in_ch)
        self.bn = nn.BatchNorm1d(in_ch)
        self.in_ch=in_ch



    def forward(self, image, text, k):
        residual = image
        text_emb = self.emb1(text.transpose(2, 1)).transpose(2, 1)

        alpha = 1
        K_neigs = (image.shape[1] * k * alpha).round().int()

        # a = torch.ones_like(k)*0.5
        # K_neigs = (image.shape[1] * a).round().int()

        # alpha = 1.5
        # beta = torch.clamp(k * alpha, max=1.0)
        # K_neigs = (image.shape[1] * beta).round().int()



        attn = torch.matmul(image, text_emb.transpose(2, 1)) / math.sqrt(self.in_ch)

       
        max_k = K_neigs.max().item()

        topk_values, topk_indices = torch.topk(attn, max_k, dim=1, largest=True)

        
        range_matrix = torch.arange(max_k).unsqueeze(0).expand(K_neigs.shape[0], -1).to(attn.device)

     
        K_neigs_expanded = K_neigs.unsqueeze(1).expand(-1, max_k)
        mask = range_matrix < K_neigs_expanded  # (batch_size, max_k)

       
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, attn.shape[-1])

        
        full_mask = torch.zeros_like(attn, dtype=torch.bool)
        full_mask.scatter_(1, topk_indices, mask_expanded)

        
        attn = nn.Softmax(dim=-2)(attn)
        attn_topk = attn * full_mask
        out = attn_topk.matmul(self.out(text))


        out = F.relu(self.bn(out.permute(0, 2, 1).contiguous())).permute(0, 2, 1).contiguous() + residual
        return out


class GuideDecoder(nn.Module):

    def __init__(self,in_channels, out_channels, spatial_size) -> None:

        super().__init__()

        self.guide_layer = HGNN_layer(in_channels)   # for skip
        self.spatial_size = spatial_size
        self.decoder = UnetrUpBlock(2,in_channels,out_channels,3,2,norm_name='BATCH')

    
    def forward(self, vis, skip_vis, txt,k):

        if txt is not None:

            vis =  self.guide_layer(vis, txt, k)

        vis = rearrange(vis,'B (H W) C -> B C H W',H=self.spatial_size,W=self.spatial_size)
        skip_vis = rearrange(skip_vis,'B (H W) C -> B C H W',H=self.spatial_size*2,W=self.spatial_size*2)

        output = self.decoder(vis,skip_vis)
        output = rearrange(output,'B C H W -> B (H W) C')

        return output


