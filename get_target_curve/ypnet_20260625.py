# !/usr/bin/env python
# coding:utf8

import torch
import torch.nn as nn
import torch.nn.functional as F 
DEVICE = "cuda"

class AutoDisEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, H_j=20, alpha=0.1, t=1e-5):
        super(AutoDisEmbedding, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w_j = nn.Linear(in_dim, H_j)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.W_j = nn.Linear(H_j, H_j)
        self.alpha = alpha
        # self.t = t
        self.softmax = nn.Softmax(dim=-1)
        self.ME = nn.Parameter(torch.randn(H_j, out_dim))

    def forward(self, x):
        h_j = self.leaky_relu(self.w_j(x))
        x_hat_j = self.W_j(h_j) + self.alpha * h_j
        # x_hat_j_h = self.softmax(x_hat_j / self.t)
        x_hat_j_h = self.softmax(x_hat_j)
        e_j = x_hat_j_h @ self.ME
        return e_j


class SENETLayer(nn.Module):
    """SENETLayer used in FiBiNET.
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,filed_size,embedding_size)``.
      Output shape
        - A list of 3D tensor with shape: ``(batch_size,filed_size,embedding_size)``.
      Arguments
        - **filed_size** : Positive integer, number of feature groups.
        - **reduction_ratio** : Positive integer, dimensionality of the
         attention network output space.
        - **seed** : A Python integer to use as random seed.
      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
        Tongwen](https://arxiv.org/pdf/1905.09433.pdf)
    """

    def __init__(self, filed_size, reduction_ratio=3, seed=1024):
        super(SENETLayer, self).__init__()
        self.seed = seed
        self.filed_size = filed_size
        self.reduction_size = max(1, filed_size // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(self.filed_size, self.reduction_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.reduction_size, self.filed_size, bias=False),
            nn.Sigmoid()
        )
        # self.to(device)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        Z = torch.mean(inputs, dim=-1, out=None)
        A = self.excitation(Z)
        V = torch.mul(inputs, torch.unsqueeze(A, dim=2))

        return V

class ResidualConv1D(nn.Module):
    """用于 Head 的轻量残差块，不改变长度"""
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.ELU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):  # (B, C, L)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out + identity)
        return out

class MLP(nn.Module):
    """单字段 MLP: in_dim -> emb_dim，可选残差"""
    def __init__(self, in_dim: int, emb_dim: int, use_residual: bool = True):
        super().__init__()
        hidden = max(emb_dim * 2, 32)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, emb_dim),
        )
        self.act = nn.ReLU(inplace=True)
        self.use_residual = use_residual and (in_dim == emb_dim)
        if self.use_residual:
            self.proj = nn.Identity()
        else:
            # 当 in_dim != emb_dim 时，不做残差（也可以投影一下，这里先简单点）
            self.proj = None

    def forward(self, x):  # (B, in_dim)
        out = self.net(x)
        if self.use_residual and self.proj is not None:
            out = out + self.proj(x)
        out = self.act(out)
        return out  # (B, emb_dim)

class MixerBlock(nn.Module):
    """
    标准 MLP-Mixer Block:
    - token-mixing: 在 L 维度上做 MLP
    - channel-mixing: 在 C 维度上做 MLP
    适用于输入 (B, L, C)
    """
    def __init__(self, num_tokens: int, emb_dim: int, token_mlp_ratio: float = 0.5, channel_mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        # token-mixing: 作用在 L 维
        token_hidden = int(num_tokens * token_mlp_ratio)
        self.token_norm = nn.LayerNorm(emb_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, token_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden, num_tokens),
            nn.Dropout(dropout),
        )

        # channel-mixing: 作用在 C 维
        channel_hidden = int(emb_dim * channel_mlp_ratio)
        self.channel_norm = nn.LayerNorm(emb_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(emb_dim, channel_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):  # x: (B, L, C)
        # token-mixing: 在 L 维操作，需要把 (B, L, C) -> (B, C, L)
        h = x
        x_norm = self.token_norm(x)                 # (B, L, C)
        x_perm = x_norm.permute(0, 2, 1)           # (B, C, L)
        x_perm = self.token_mlp(x_perm)            # (B, C, L)
        x = x_perm.permute(0, 2, 1) + h            # (B, L, C)

        # channel-mixing: 在 C 维操作
        h = x
        x_norm = self.channel_norm(x)              # (B, L, C)
        x = self.channel_mlp(x_norm) + h           # (B, L, C)
        return x

class CNNHead(nn.Module):
    """
    升级版 CNN Head:
    - FC -> 粗略 (B, 3, 16)
    - + 频点 embedding
    - Conv -> ResidualConvBlock * 2 -> Conv 输出
    """
    def __init__(self, in_dim: int, line_num: int, points_num_each_line: int):
        super().__init__()
        self.line_num = line_num
        self.points = points_num_each_line
        mid_channels = 32  # 可以稍微加一点通道

        self.fc = nn.Linear(in_dim, line_num * points_num_each_line)
        self.freq_emb = nn.Parameter(torch.randn(1, 1, self.points))

        self.stem = nn.Sequential(
            nn.Conv1d(line_num, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ELU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            ResidualConv1D(mid_channels, kernel_size=3),
            ResidualConv1D(mid_channels, kernel_size=5),
        )
        self.out_conv = nn.Conv1d(mid_channels, line_num, kernel_size=1)

    def forward(self, cls_out):  # (B, C)
        B = cls_out.size(0)
        x = self.fc(cls_out)                     # (B, 3*16)
        x = x.view(B, self.line_num, self.points)
        x = x + self.freq_emb
        x = self.stem(x)                         # (B, mid_channels, 16)
        x = self.res_blocks(x)                   # (B, mid_channels, 16)
        x = self.out_conv(x)                     # (B, 3, 16)
        return x.view(B, -1)

class SimpleTransformerEncoderLayer(nn.Module):
    """字段级 TransformerEncoderLayer，用在 Mixer 之后"""
    def __init__(self, emb_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim,
                                          num_heads=num_heads,
                                          batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(emb_dim)
        hidden = int(emb_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):  # (B, L, C)
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = self.drop1(x)
        x = x + h

        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + h
        return x

def rotate_half(x):
    """将输入的后半部分旋转到前半部分，用于 RoPE 的快速实现"""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin):
    """对 q 和 k 应用旋转位置编码
    q, k: (B, H, L, D)  D 为 head_dim，必须为偶数
    cos, sin: (L, D) 或 (1, L, 1, D) 等广播兼容形状
    """
    # 将 q,k 与 cos,sin 相乘：标准 RoPE 公式：x_rot = x * cos + rotate_half(x) * sin
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

class RoPEMultiheadAttention(nn.Module):
    """支持 RoPE 的多头注意力，取代 nn.MultiheadAttention"""
    def __init__(self, emb_dim, num_heads, dropout=0.1, base=10000):
        super().__init__()
        assert emb_dim % num_heads == 0
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.base = base

        # QKV 投影（不区分，一次性投影后拆分）
        self.qkv_proj = nn.Linear(emb_dim, 3 * emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout_layer = nn.Dropout(dropout)

        # 预计算频率（用于 RoPE），实际使用时会根据序列长度生成 cos/sin
        self.register_buffer("inv_freq", None)  # 延迟初始化
        self._init_inv_freq("no_device")

    def _init_inv_freq(self, device):
        """根据最大序列长度预计算 inv_freq"""
        if self.inv_freq is not None:
            return
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def _get_rotary_embeddings(self, seq_len, device):
        """获取 cos 和 sin 表，形状 (seq_len, head_dim)"""
        self._init_inv_freq(device)
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, head_dim//2)
        emb = torch.cat((freqs, freqs), dim=-1)            # (seq_len, head_dim)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        device = x.device

        # 1. 投影得到 Q, K, V
        qkv = self.qkv_proj(x)  # (B, L, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)  # 每个 (B, L, C)

        # 2. 重塑为多头格式 (B, L, H, D) -> (B, H, L, D)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 应用 RoPE
        cos, sin = self._get_rotary_embeddings(L, device)  # (L, head_dim)
        # 调整形状以便广播: (1, 1, L, D)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, L, D)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rope(q, k, cos, sin)

        # 4. 缩放点积注意力
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        attn_output = torch.matmul(attn_weights, v)  # (B, H, L, D)

        # 5. 合并多头并输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, C)
        attn_output = self.out_proj(attn_output)
        return attn_output, None   # 返回 (output, attention_weights) 以兼容原接口

class SimpleTransformerEncoderLayerWithRoPE(nn.Module):
    """使用 RoPE 的 TransformerEncoderLayer"""
    def __init__(self, emb_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = RoPEMultiheadAttention(emb_dim, num_heads, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(emb_dim)
        hidden = int(emb_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):  # (B, L, C)
        # 自注意力 + 残差
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x)   # 注意：RoPEMultiheadAttention 返回 (output, None)
        x = self.drop1(x)
        x = x + h

        # FFN + 残差
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + h
        return x

class YPNet(nn.Module):
    def __init__(self):
        super(YPNet, self).__init__()
        hidden_dim = 128
        self.hidden_tokens = 9
        self.info_offset = [8, 2, 0, 11, 23, 58, 132]
        self.emb_names = ["gender", "his", "db","outlook", "mic", "rec", "deafness_type"]
        # self.emb_dims = [3, 6, 2, 12, 35, 74, 3]
        # self.emb_idx = [0, 1, 2, 3, 4, 5, 6]
        # ====== 更新：删除outlook, mic, rec等特征 ======
        self.emb_names = ["gender", "his", "db", "deafness_type"]
        self.emb_dims = [3, 6, 2, 3]
        self.emb_idx = [0, 1, 2, 6]
        # ==============================================
        self.emb_dict = nn.ModuleDict({
            name: nn.Embedding(num_embeddings=dim, embedding_dim=hidden_dim)
            for name, dim in zip(self.emb_names, self.emb_dims)
        })
        mlp_names = ["htl", "bcl", "nal"]
        mlp_in_dims = [10, 10, 3*9*18]
        self.mlp_dict = nn.ModuleDict({
            name: MLP(in_dim, hidden_dim) for name, in_dim in zip(mlp_names, mlp_in_dims)
        })
        self.linear_emd = AutoDisEmbedding(1, hidden_dim)
        self.layerNorm = nn.LayerNorm(hidden_dim)
        # ====== 字段类型 Embedding（区分不同字段） ======
        self.field_type_emb = nn.Embedding(self.hidden_tokens, hidden_dim)
        # ====== CLS token ======
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        #====== SENETLayer ======
        self.senet = SENETLayer(filed_size=self.hidden_tokens, reduction_ratio=2) # 不用bcl，如果用bcl就让filed_size=12
        # ====== 多层 MLP-Mixer ======
        mixer_layers = 6
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(num_tokens=self.hidden_tokens + 1,  # +1 for CLS
                       emb_dim=hidden_dim,
                       token_mlp_ratio=2.0,
                       channel_mlp_ratio=4.0,
                       dropout=0.1)
            for _ in range(mixer_layers)
        ])

        # ====== 字段级 Transformer ======
        transformer_layers = 6
        self.transformer_blocks = nn.ModuleList([
            SimpleTransformerEncoderLayerWithRoPE(emb_dim=hidden_dim,
                                          num_heads=4,
                                          mlp_ratio=4.0,
                                          dropout=0.1)
            for _ in range(transformer_layers)
        ])
        # ----- rope + attn residual ------
        # self.transformer_blocks = SimpleTransformerEncoder(emb_dim=hidden_dim, num_heads=4, num_layers= transformer_layers)
        # ---------------------------------
        self.transformer_norm = nn.LayerNorm(hidden_dim)

        # ====== CNN Head: CLS -> 曲线 ======
        self.line_num = 3
        self.points_num_each_line = 65
        self.head = CNNHead(in_dim=hidden_dim,
                            line_num=self.line_num,
                            points_num_each_line=self.points_num_each_line)
        

    def forward(self, tlt, info_num, info_cat, nal_feat, chip):
        B = len(tlt)
        info_cat = info_cat.type(torch.LongTensor).to(DEVICE).squeeze(1) # torch.Size([256, 1, 7])
        info_num = info_num.to(DEVICE) # torch.Size([256, 1, 2])
        tlt = tlt.to(DEVICE)
        htl = tlt[:, :, 0]  # torch.Size([256, 10, 2])
        bcl = tlt[:, :, 1]
        emb_info_num = self.linear_emd(info_num.permute(0,2,1)) # torch.Size([256, 2, 128])
        all_embs = [emb_info_num]
        for idx, name in zip(self.emb_idx, self.emb_names):
            info_cat_emb = self.emb_dict[name](info_cat[:, idx] - self.info_offset[idx]).unsqueeze(1)
            all_embs.append(info_cat_emb)
        
        htl_emb = self.mlp_dict["htl"](htl).unsqueeze(1)
        bcl_emb = self.mlp_dict["bcl"](bcl).unsqueeze(1)
        nal_emb = self.mlp_dict["nal"](nal_feat.view(B, -1)).unsqueeze(1)
        all_embs.append(htl_emb)
        all_embs.append(bcl_emb)
        all_embs.append(nal_emb)
        x = torch.cat(all_embs, dim=1)
        L = x.size(1)
        
        # x = self.layerNorm(x)

        type_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)  # (B, L)
        type_emb = self.field_type_emb(type_ids)                                # (B, L, C)
        x = x + type_emb 
        # ====== 3. SENETLayer ======
        x = self.senet(x)  # (B, L, C)

        # prepend CLS: (B, 1+L, C)
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, C)
        x = torch.cat([cls, x], dim=1)           # (B, 1+L, C)

        # MLP-Mixer 堆叠
        for blk in self.mixer_blocks:
            x = blk(x)

        # Transformer 堆叠
        # -------- base transformer ------
        for blk in self.transformer_blocks:
            x = blk(x)
        # -------------------------------
        # ------ rope + attn residual ------
        # x = self.transformer_blocks(x)
        # ----------------------------------
        
        x = self.transformer_norm(x)  # (B, 1+L, C)

        # 6. 取 CLS 作为全局表征，再过 CNNHead 输出 3*16 频响点
        cls_out = x[:, 0, :]          # (B, C)
        # ====== 7. CNN 分支：原始输入特征 ======
        out = self.head(cls_out)      # (B, 195)
        return out
