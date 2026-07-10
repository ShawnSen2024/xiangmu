import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
# from transformers.activations import ACT2FN

DEVICE = "cuda"

class Dice(nn.Module):
    def __init__(self, input_dim, eps=1e-9):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=eps, momentum=0.01)
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X):
        p = torch.sigmoid(self.bn(X))
        output = p * X + self.alpha * (1 - p) *  X
        return output

def get_activation(activation, hidden_units=None):
    if isinstance(activation, str):
        if activation.lower() in ["prelu", "dice"]:
            assert type(hidden_units) == int
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "softmax":
            return nn.Softmax(dim=-1)
        elif activation.lower() == "prelu":
            return nn.PReLU(hidden_units, init=0.1)
        elif activation.lower() == "dice":
            return Dice(hidden_units)
        else:
            return getattr(nn, activation)()
    elif isinstance(activation, list):
        if hidden_units is not None:
            assert len(activation) == len(hidden_units)
            return [get_activation(act, units) for act, units in zip(activation, hidden_units)]
        else:
            return [get_activation(act) for act in activation]
    return activation


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


class BiLSTM(nn.Module):
    def __init__(self, in_dim, embed_dim, lstm_hidden_dim,lstm_num_layers,dropout,emd=True):
        super(BiLSTM, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.num_layers = lstm_num_layers
        self.emd = emd
        if emd:
            self.embed = AutoDisEmbedding(in_dim,embed_dim)
        self.bilstm = nn.LSTM(embed_dim, self.hidden_dim // 2, num_layers=1, dropout=dropout, bidirectional=True,
                              bias=False)
    def forward(self, x):
        # print(x.shape)
        x = x.unsqueeze(-1)
        if self.emd:
            x = self.embed(x)
        bilstm_out, _ = self.bilstm(x)
        bilstm_out = torch.transpose(bilstm_out, 2, 1)
        bilstm_out = F.tanh(bilstm_out)
        logit = F.max_pool1d(bilstm_out.permute(0,2,1), bilstm_out.size(1))
        return bilstm_out, logit


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

    def __init__(self, filed_size, reduction_ratio=3, seed=1024, device='cpu'):
        super(SENETLayer, self).__init__()
        self.seed = seed
        self.filed_size = filed_size
        self.reduction_size = max(1, filed_size // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(self.filed_size, self.reduction_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.reduction_size, self.filed_size, bias=False),
            nn.ReLU()
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

class LinearAttention(nn.Module):
    """
    copied and modified from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L159
    """

    def __init__(self, item_len, dim, heads=8, dim_head=32, verbose=True):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        self.ff = nn.Sequential(
            nn.Conv1d(dim, dim * 8, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(dim * 8, dim, kernel_size=1),
            nn.LeakyReLU()
        )
        # self.ff = MLP(dim, dim*4,drop=0.1)
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
        # out = out.squeeze(-1)  # B,C,N,1 -> B,C,N
        out = self.ln1((x+out).squeeze(-1))
        out = out + self.ff(out)
        out = self.ln2(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels // reduction_ratio, kernel_size=1)
        self.conv3 = nn.Conv2d(num_channels // reduction_ratio, num_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.ln = nn.LayerNorm([num_channels // reduction_ratio, 1, 1])
        # 使用1x1卷积来减少维度，并增加非线性
        self.relu = nn.ReLU()

    def forward(self, x_):
        b, c, h, w = x_.shape
        x = self.conv1(x_).view(b, 1, w * h).permute(0, 2, 1)
        x = self.softmax(x)
        i = x_.view([b, c, h * w])
        x = torch.bmm(i, x).view([b, c, 1, 1])
        x = self.conv2(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x + x_


class FactorizationMachineBlock(nn.Module):
    """ Factorization Machine Block (FMB) """

    def __init__(self, input_features=16, output_features=16, embedding_dim=16, rank_k=8,
                 mlp_hidden_units=[16, 16], mlp_hidden_activations="relu", mlp_dropout=0):
        super(FactorizationMachineBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.output_features = output_features
        self.rank_k = rank_k
        self.input_features = input_features
        if self.rank_k is not None:
            # optimized FM
            self.proj_Y = nn.Parameter(torch.randn(self.input_features, self.rank_k))
            fm_out_dim = input_features * rank_k
        else:
            # vanilla FM
            fm_out_dim = input_features * input_features
        self.layer_norm = nn.LayerNorm(fm_out_dim)
        self.mlp = MLP_Block(input_dim=fm_out_dim,
                             output_dim=output_features * embedding_dim,
                             hidden_units=mlp_hidden_units,
                             hidden_activations=mlp_hidden_activations,
                             output_activation="relu",
                             dropout_rates=mlp_dropout)

    def forward(self, x):
        flatten_fm = self.optimized_fm(x)
        mlp_in = self.layer_norm(flatten_fm)
        mlp_out = self.mlp(mlp_in)
        return mlp_out.view(-1, self.output_features, self.embedding_dim)

    def optimized_fm(self, x):
        _, n, d = x.shape
        if self.rank_k is not None:
            projected = x.transpose(1, 2) @ self.proj_Y  # b x d x k
            fm_matrix = torch.bmm(x, projected)  # b x n x k
        else:
            fm_matrix = torch.bmm(x, x.transpose(1, 2))  # b x n x n
        return fm_matrix.flatten(start_dim=1)


class LinearCompressionBlock(nn.Module):
    """ Linear Compression Block (LCB) """

    def __init__(self, input_features=16, output_features=8):
        super(LinearCompressionBlock, self).__init__()
        self.linear = nn.Linear(input_features, output_features, bias=False)

    def forward(self, x):
        out = self.linear(x.transpose(1, 2))
        return out.transpose(1, 2)


class MLP_Block(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_units=[],
                 hidden_activations="ReLU",
                 output_dim=None,
                 output_activation=None,
                 dropout_rates=0.0,
                 batch_norm=False,
                 bn_only_once=False,  # Set True for inference speed up
                 use_bias=True):
        super(MLP_Block, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = get_activation(hidden_activations, hidden_units)
        hidden_units = [input_dim] + hidden_units
        if batch_norm and bn_only_once:
            dense_layers.append(nn.BatchNorm1d(input_dim))
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm and not bn_only_once:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if output_activation is not None:
            dense_layers.append(get_activation(output_activation))
        self.mlp = nn.Sequential(*dense_layers)  # * used to unpack list

    def forward(self, inputs):
        return self.mlp(inputs)


class WuKongLayer(nn.Module):
    def __init__(self, input_features=16, lcb_features=8, fmb_features=8, embedding_dim=16,
                 fmp_rank_k=4, fmb_mlp_units=[16, 16], fmb_mlp_activations="relu",
                 fmb_dropout=0.1, layer_norm=True):
        super(WuKongLayer, self).__init__()
        self.fmb = FactorizationMachineBlock(input_features,
                                             fmb_features,
                                             embedding_dim,
                                             fmp_rank_k,
                                             fmb_mlp_units,
                                             fmb_mlp_activations,
                                             fmb_dropout)
        self.lcb = LinearCompressionBlock(input_features, lcb_features)
        self.layer_norm = nn.LayerNorm(embedding_dim) if layer_norm else None
        if input_features != lcb_features + fmb_features:
            self.residual_proj = nn.Linear(input_features, lcb_features + fmb_features)

    def forward(self, x):
        fmb_out = self.fmb(x)
        lcb_out = self.lcb(x)
        # print(fmb_out.shape,lcb_out.shape)
        concat_out = torch.cat([fmb_out, lcb_out], dim=1)  # b x (fmb + lcb) x d
        # print(concat_out.shape)
        out = self.residual(concat_out, x)
        # print(4,out.shape)
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        # print("out:",out.shape)
        return out

    def residual(self, out, x):
        if out.shape[1] != x.shape[1]:
            res = self.residual_proj(x.transpose(1, 2)).transpose(1, 2)
        else:
            res = x
        return out + res


class MLP(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels=None,
            out_channels=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act1 = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.act2 = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc_out = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop(x)
        x = self.fc_out(x)
        return x


class TimeMoeTemporalBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.ReLU(inplace=True)

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class TimeMoeSparseExpertsLayer(nn.Module):
    def __init__(self, num_experts, num_experts_per_tok,intermediate_size, hidden_size, hidden_act):
        super().__init__()
        # self.config = config
        self.top_k = num_experts_per_tok
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.norm_topk_prob = False

        moe_intermediate_size = intermediate_size // self.top_k

        # gating
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [TimeMoeTemporalBlock(
                hidden_size=hidden_size,
                intermediate_size=moe_intermediate_size,
                hidden_act=hidden_act,
            ) for _ in range(num_experts)]
        )

        self.shared_expert = TimeMoeTemporalBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
        )
        self.shared_expert_gate = torch.nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits -> (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights_ = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights_, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        loss = self._compute_aux_loss(routing_weights_, selected_experts)
        return final_hidden_states, loss

    def _compute_aux_loss(self, probs, gates):
        expert_mask = F.one_hot(gates.argmax(dim=-1), num_classes=self.num_experts)
        expert_freq = expert_mask.float().mean(dim=0)
        freq_ratio = expert_freq/(1/self.num_experts)
        balance_loss = self.num_experts * (freq_ratio*F.normalize(probs, p=1, dim=-1)).sum(dim=-1)
        return balance_loss*0.01

