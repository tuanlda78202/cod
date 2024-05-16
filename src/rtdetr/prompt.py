import torch
import torch.nn as nn


def init_prompt(dim1, dim2, dim3=None):
    if dim3 is None:
        prompt = nn.Parameter(torch.FloatTensor(dim1, dim2), requires_grad=True)
    else:
        prompt = nn.Parameter(torch.FloatTensor(dim1, dim2, dim3), requires_grad=True)
    return prompt


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PromptCOD(nn.Module):
    """
    Query: CLIP Image features
    Key: CLIP Label embeddings
    """

    def __init__(
        self,
        emb_dim=512,
        key_dim=512,
        top_k=4,
        pool_size=40,
        c_length=20,
        g_length=12,
        g_layers=[0, 1],
        c_layers=[2, 3],
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.key_dim = key_dim

        self.top_k = top_k
        self.pool_size = pool_size

        self.g_layers = g_layers
        self.c_layers = c_layers
        self.g_length = g_length
        self.c_length = c_length

        # MLP
        self.image_project = FFN(emb_dim, emb_dim)
        self.text_project = FFN(key_dim, key_dim)

        for l in self.g_layers:
            p = init_prompt(self.g_length, emb_dim)
            setattr(self, f"g_{l}", p)

        for l in self.c_layers:
            p = init_prompt(self.pool_size, self.c_length, emb_dim)
            setattr(self, f"c_{l}", p)

    def forward(self, l, x_block, x_query, key):
        x_query = x_query.squeeze(1).to("cuda")
        key = key.to("cuda")
        if l in self.g_layers:
            p = getattr(self, f"g_{l}")
            P_ = p.expand(len(x_query), -1, -1)

            j = int(self.g_length / 2)
            Gk = P_[:, :j, :]
            Gv = P_[:, j:, :]
            p_return = [Gk, Gv]

            prompt_loss = 0

        if l in self.c_layers:
            B, C = x_query.shape
            p = getattr(self, f"c_{l}")
            K_fix = key

            x_query = self.image_project(x_query)
            K_fix = self.text_project(K_fix)

            q = nn.functional.normalize(x_query, dim=1)
            n_K = nn.functional.normalize(K_fix, dim=1)
            sim = torch.einsum("bj,kj->bk", q, n_K)

            top_k = torch.topk(sim, self.top_k, dim=1)
            k_idx = top_k.indices
            prompt_loss = (1.0 - sim[:, k_idx]).sum()
            P_ = p[k_idx]

            i = int(self.c_length / 2)
            Ck = P_[:, :, :i, :].reshape((B, -1, self.emb_dim))
            Cv = P_[:, :, i:, :].reshape((B, -1, self.emb_dim))

            p_return = [Ck, Cv]

        else:
            p_return = None
            prompt_loss = 0

        return p_return, prompt_loss, x_block


# Prefix Tuning
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

    def forward(self, x, prompt=None):
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

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
