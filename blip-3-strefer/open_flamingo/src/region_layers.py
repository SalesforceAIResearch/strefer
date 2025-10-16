import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

mask_default_shape = (1, 1, 384, 384)


def token_merge(x, r):
    """
    x: intput token [bs, n, d]
    r indicates the number of tokens to remove
    """
    x1 = x[:, :-1, :]
    x2 = x[:, 1:, :]
    norm_x1 = F.normalize(x1, p=2, dim=-1)  # l2 norm
    norm_x2 = F.normalize(x2, p=2, dim=-1)  # l2 norm
    sim = torch.sum(norm_x1*norm_x2, dim=-1)  # dot product
    # sim is cosine similarity

    values, indices = torch.topk(sim.flatten(), r) 
    kth_largest = values[-1]

    new_tokens = []
    merged_tokens = []
    for i in range(sim.shape[1]):
        merged_tokens.append(x[:, i:i+1, :])
        if sim[0,i]<kth_largest:
            if len(merged_tokens)>0:
                new_tokens.append(torch.mean(torch.cat(merged_tokens, dim=1), dim=1, keepdim=True))
                merged_tokens = []
    
    merged_tokens.append(x[:, sim.shape[1]:sim.shape[1]+1, :])
    if len(merged_tokens)>0:
        new_tokens.append(torch.mean(torch.cat(merged_tokens, dim=1), dim=1, keepdim=True))

    return torch.cat(new_tokens, dim=1)

class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MaskExtractor(nn.Module):
    def __init__(self, depth=2, region_token_num=4):
        super(MaskExtractor, self).__init__()

        mm_hidden_size = 1152
        hidden_size = 3072
        self.image_aspect_ratio = 'square'

        self.mask_pooling = MaskPooling()
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        self.feat_linear =  nn.Sequential(*modules)
        self.region_token_num = region_token_num

    def forward(self, feats, masks, X_features, ann_indices, frame_nums):
        query_feats = []

        batch_size = len(masks)
        region_token_nums = []
        for idx in range(batch_size):
            mask = masks[idx].unsqueeze(0).float()
            if len(mask[0])==0:
                print('mask error')
                mask = torch.zeros(mask_default_shape).to(X_features.device).float()
            
            if self.image_aspect_ratio == 'pad':
                _h, w = mask.shape[-2:]
                max_hw = max(_h, w)
                try:
                    mask = F.pad(mask,
                                ((max_hw-w)//2,(max_hw-w)-(max_hw-w)//2,
                                (max_hw-_h)//2,(max_hw-_h)-(max_hw-_h)//2,
                                0,0,
                                0,0)
                                )
                except:
                    print(mask.shape)
                    mask = torch.zeros(mask_default_shape).to(X_features.device).float()

            ann_index = []
            for index in ann_indices[idx]:
                for id_ in index:
                    ann_index.append(id_)
            
            feat = feats[ann_index]

            N = int(pow(feat.shape[1], 0.5))
            feat = feat.reshape(feat.shape[0], N, N, -1).permute(0,3,1,2)

            raw_dtype = feat.dtype
            feat = feat.to(mask.dtype)

            
            mask_feat_raw = self.mask_pooling(feat, mask) # [n, 1024]

            merged_mask_feats = []

            start_index = 0
            for index in ann_indices[idx]:
                mask_feat = mask_feat_raw[start_index: start_index+len(index), :].unsqueeze(0)
                if mask_feat.shape[1]>self.region_token_num:
                    mask_feat = token_merge(mask_feat, mask_feat.shape[1]-self.region_token_num)
                region_token_nums.append(mask_feat.shape[1])
                merged_mask_feats.append(mask_feat)
                start_index+=len(index)

            mask_feats = torch.cat(merged_mask_feats, dim=1)
            mask_feats = mask_feats.reshape(-1, mask_feat_raw.shape[-1])
            mask_feats = mask_feats.to(raw_dtype)
            query_feats.append(mask_feats)
        mask_feats = torch.cat(query_feats, dim=0)
        mask_feats_linear = self.feat_linear(mask_feats)

        return mask_feats_linear, region_token_nums

    
class MaskPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):

        if not x.shape[-2:] == mask.shape[-2:]:
            # reshape mask to x
            mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        mask = (mask > 0).to(mask.dtype)
        mask = mask.permute(1,0,2,3)
        denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8
        
        try:
            mask_pooled_x = (x * mask/denorm).sum(-1).sum(-1)
        except Exception as e:
            print(f"An error occurred: {e}")
            mask_pooled_x = x.sum(-1).sum(-1)
            
        return mask_pooled_x
