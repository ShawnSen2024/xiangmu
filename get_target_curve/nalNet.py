# !/usr/bin/env python
# coding:utf8
import math
import torch
import torch.nn as nn
from einops import rearrange

DEVICE = "cuda"


class LinearAttention(nn.Module):
    """
    copied and modified from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L159
    """

    def __init__(self, item_len, dim, heads=4, dim_head=32, verbose=True):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        self.ff = nn.Sequential(
            nn.Conv1d(dim, dim * 4, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(dim * 4, dim, kernel_size=1),
            nn.LeakyReLU()
        )
        self.ln1 = nn.LayerNorm([dim, item_len])
        self.ln2 = nn.LayerNorm([dim, item_len])

    def forward(self, x):
        '''
        Args:
            x: torch.tensor (B,C,N), C=num-channels, N=num-points
        Returns:
            out: torch.tensor (B,C,N)
        '''
        x = x.unsqueeze(-1)  # add w dimension
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        out = self.to_out(out)
        out = out.squeeze(-1)  # B,C,N,1 -> B,C,N
        out = self.ln1(x.squeeze(-1) + out)
        out = out + self.ff(out)
        out = self.ln2(out)
        return out


class NalNet(nn.Module):
    def __init__(self):
        super(NalNet, self).__init__()

        self.emd = nn.Embedding(num_embeddings=256,
                                embedding_dim=128,
                                padding_idx=0)
        # self.emb.weight = nn.Parameter(embeddings)
        self.info_len = 5
        self.tlt_len = 2
        self.tlt_shape_net = nn.Conv1d(10, 128, kernel_size=1)
        self.trans1 = LinearAttention(self.info_len + self.tlt_len, 128, dim_head=16)
        self.trans2 = LinearAttention(1 + 2 * (self.info_len + self.tlt_len), 128, dim_head=16)
        self.trans3 = LinearAttention(1 + 2 * (1 + 2 * (self.info_len + self.tlt_len)), 128, dim_head=16)
        self.trans7 = LinearAttention(128, 64, dim_head=8)
        self.sen_len = 10
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(64, 7), stride=(64, 7), padding=(1, 0))
        self.conv2 = nn.Conv2d(64, 18, kernel_size=1)
        # self.conv2 = nn.Conv2d(64, 9, kernel_size=7, stride=7, padding=1)

        # self.cn = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=1)
        # self.ln3 = nn.LayerNorm([self.sen_len, 43])
        # self.ln4 = nn.LayerNorm([self.sen_len, 32])
        # "X_WDRC", "X_HC", "X_EQ", "X_FBC", "X_FE", "X_NR", "X_SG", "OTHERS", "AU"

        # self.wdrc_lv = nn.Linear(self.sen_len * (self.info_len + self.tlt_len), self.sen_len * 2)
        # hc 9和8 的离散值，对8补1变成9和9，一共18个参数，取前17个
        self.out = nn.Sequential(
            nn.Linear(18 * 18, 18 * 9),
            # nn.Dropout(),
            # nn.LeakyReLU(),
            # nn.Softmax()
            nn.Linear(18 * 9, 18 * 9)
        )

    def forward(self, tlt, info):
        # print(self.emd)
        # print(info.shape)
        batch_size = len(tlt)
        ''' Bi-LSTM Computation '''
        # 1x10
        info = info.type(torch.LongTensor).to(DEVICE)
        # 2x16
        tlt = tlt.to(DEVICE)
        # 64x9
        emd = self.emd(info).squeeze(1).permute(0, 2, 1)
        # 64x2
        tlt_emd = self.tlt_shape_net(tlt)
        # 64x11
        # print(tlt_emd.size(), emd.size())
        x = torch.concat((tlt_emd, emd), dim=-1)
        # print("aa", x.shape)
        new_x = self.trans1(x)
        x = torch.concat((tlt_emd[..., 0].unsqueeze(-1), new_x, x), dim=-1)
        new_x = self.trans2(x)
        x = torch.concat((tlt_emd[..., 0].unsqueeze(-1), new_x, x), dim=-1)
        new_x = self.trans3(x)
        x = torch.concat((tlt_emd, new_x, x), dim=-1).permute(0, 2, 1)
        x = self.trans7(x).unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        out_prob = self.out(x.view(batch_size, -1))

        return out_prob


NAL_MODEL_LOCATION = r"nal_model.pth"
def load_nal_model():
    nal = NalNet()  # 创建一个与保存模型参数相同的模型实例
    nal.load_state_dict(torch.load(NAL_MODEL_LOCATION))
    nal.eval()
    return nal

nal_model = load_nal_model()
nal_model.to(DEVICE)

def nal_feat(tlt, nal_info, batch_size):
    # tlt = tlt.repeat(3, 1, 1)
    data = torch.concat((nal_info.repeat(3, 1), torch.tensor([128, 129, 130], dtype=torch.float32).unsqueeze(-1).repeat(batch_size, 1).to(DEVICE)), dim=-1)
    return nal_model(tlt.repeat(3, 1, 1), data).reshape((batch_size, 3, 9, 18)).detach()

