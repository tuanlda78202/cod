import torch
import torch.nn as nn


def init_prompt(dim1, dim2, dim3=None):
    if dim3 is None:
        prompt = nn.Parameter(torch.FloatTensor(dim1, dim2), requires_grad=True)
    else:
        prompt = nn.Parameter(torch.FloatTensor(dim1, dim2, dim3), requires_grad=True)
    return prompt


class PromptCOD(nn.Module):
    """
    Query: CLIP Image features
    Key: CLIP Label embeddings
    """

    def __init__(
        self,
        emb_dim=512,
        key_dim=512,
        pool_size=40,
        p_length=20,
        p_layers=[0, 1],
        top_k=1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.key_dim = key_dim

        self.top_k = top_k
        self.p_layers = p_layers
        self.pool_size = pool_size
        self.p_length = p_length

        for e in self.p_layers:
            p = init_prompt(self.pool_size, self.p_length, emb_dim)
            setattr(self, f"p_{e}", p)

    def forward(self, l, x_block, x_query, key):
        x_query = x_query.squeeze(1)

        if l in self.p_layers:
            B, C = x_query.shape
            p = getattr(self, f"p_{l}")
            K = key
            # cosine similarity to match keys/queries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_query, dim=1).detach()
            cos_sim = torch.einsum("bj, kj->bk", q, n_K)

            # top-k
            top_k = torch.topk(cos_sim, self.top_k, dim=1)
            k_idx = top_k.indices
            P_ = p[k_idx]

            # select prompts
            i = int(self.p_length / 2)
            Pk = P_[:, :, :i, :].reshape((B, -1, self.emb_dim))
            Pv = P_[:, :, i:, :].reshape((B, -1, self.emb_dim))

            p_return = [Pk, Pv]
        else:
            p_return = None

        return p_return, x_block


# Self Attention - Prefix tuning
class PromptAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, register_hook=False, prompt=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        if prompt is not None:
            pk, pv = prompt
            pk = pk.reshape(B, -1, self.num_heads, C // self.num_heads).permute(
                0, 2, 1, 3
            )
            pv = pv.reshape(B, -1, self.num_heads, C // self.num_heads).permute(
                0, 2, 1, 3
            )
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
