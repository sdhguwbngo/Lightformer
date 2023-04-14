# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import math
from typing import Optional, Callable, List, Tuple, Sequence
import numpy as np
import pdb

import torch.nn.functional as F

from torch.cuda.amp import autocast
from einops import rearrange, repeat

from contextlib import contextmanager
import pdb
import time


import deepspeed
import torch
import torch.nn as nn
from scipy.stats import truncnorm

from openfold.utils.checkpointing import get_checkpoint_fn
from openfold.utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
    _chunk_slice,
)
from openfold.tool.global_config import GlobalConfig

from distutils.version import LooseVersion
TORCH_GE_1_8_0 = LooseVersion(torch.__version__) >= LooseVersion('1.8.0')

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)

def exists(val):
    return val is not None

def empty(tensor):
    return tensor.numel() == 0

def default(val, d):
    return val if exists(val) else d

class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")


class LayerNorm(nn.Module):
    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):
        d = x.dtype
        if d is torch.bfloat16 and not deepspeed.comm.is_initialized():
            with torch.cuda.amp.autocast(enabled=False):
                out = nn.functional.layer_norm(
                    x,
                    self.c_in,
                    self.weight.to(dtype=d),
                    self.bias.to(dtype=d),
                    self.eps,
                )
        else:
            out = nn.functional.layer_norm(
                x,
                self.c_in,
                self.weight,
                self.bias,
                self.eps,
            )

        return out


@torch.jit.ignore
def softmax(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax, but without automatic casting to fp32 when the input is of
    type bfloat16
    """
    d = t.dtype
    if d is torch.bfloat16 and not deepspeed.comm.is_initialized():
        with torch.cuda.amp.autocast(enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:
        s = torch.nn.functional.softmax(t, dim=dim)

    return s


# @torch.jit.script
def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: List[torch.Tensor],
) -> torch.Tensor:
    # [*, H, Q, C_hidden]
    query = permute_final_dims(query, (1, 0, 2))

    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 2, 0))

    # [*, H, V, C_hidden]
    value = permute_final_dims(value, (1, 0, 2))

    # [*, H, Q, K]
    a = torch.matmul(query, key)
    
    try:
        for b in biases:
            a += b
            
    except:
        if len(biases)>=2:
            u = biases[-1][-2]
            v = permute_final_dims(biases[-1][-1],(0,2,1))
            b = torch.matmul(u,v)
            try:
                a += b
            except:
                a += b[0]

    a = softmax(a, -1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    # [*, Q, H, C_hidden]
    a = a.transpose(-2, -3)

    return a


@torch.jit.ignore
def _attention_chunked_trainable(
    query,
    key,
    value,
    biases,
    chunk_size,
    chunk_dim,
    checkpoint,
):
    if checkpoint and len(biases) > 2:
        raise ValueError("Checkpointed version permits only permits two bias terms")

    def _checkpointable_attention(q, k, v, b1, b2):
        bs = [b for b in [b1, b2] if b is not None]
        return _attention(q, k, v, bs)

    o_chunks = []
    checkpoint_fn = get_checkpoint_fn()
    count = query.shape[chunk_dim]
    for start in range(0, count, chunk_size):
        end = start + chunk_size
        idx = [slice(None)] * len(query.shape)
        idx[chunk_dim] = slice(start, end)
        idx_tup = tuple(idx)
        q_chunk = query[idx_tup]
        k_chunk = key[idx_tup]
        v_chunk = value[idx_tup]

        def _slice_bias(b):
            idx[chunk_dim] = (
                slice(start, end) if b.shape[chunk_dim] != 1 else slice(None)
            )
            return b[tuple(idx)]

        if checkpoint:
            bias_1_chunk, bias_2_chunk = [
                _slice_bias(b) if b is not None else None
                for b in (biases + [None, None])[:2]
            ]

            o_chunk = checkpoint_fn(
                _checkpointable_attention,
                q_chunk,
                k_chunk,
                v_chunk,
                bias_1_chunk,
                bias_2_chunk,
            )
        else:
            bias_chunks = [_slice_bias(b) for b in biases]

            o_chunk = _attention(q_chunk, k_chunk, v_chunk, bias_chunks)

        o_chunks.append(o_chunk)

    o = torch.cat(o_chunks, dim=chunk_dim)
    return o



class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        use_attn: bool=False,
        rank_factor:int=-1,
        use_pair_bias: bool=True
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
            use_attn:
                Whether use _attention 
            rank_factor:
                pair factor rank
            use_pair_bias:
                Whether use pair bias
        """
        super(Attention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating
        self.use_attn = use_attn
        self.rank_factor= rank_factor
        self.use_pair_bias = use_pair_bias

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = Linear(
            self.c_q, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_k = Linear(
            self.c_k, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_v = Linear(
            self.c_v, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_o = Linear(self.c_hidden * self.no_heads, self.c_q, init="final")

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(
                self.c_q, self.c_hidden * self.no_heads, init="gating"
            )

        self.sigmoid = nn.Sigmoid()

        # self.flowformer = Flow_Attention()
        if GlobalConfig.FLOW_ATTN:
            self.flowformer = Flow_Attention()
        elif GlobalConfig.PER_ATTN:
            if GlobalConfig.PAIR_FACTOR:
                if self.use_pair_bias:
                    self.performer = FastAttention(dim_heads=self.c_hidden + self.rank_factor + 1)
                else:
                    self.performer = FastAttention(dim_heads=self.c_hidden+1)
            else:
                self.performer = FastAttention(dim_heads=self.c_hidden)

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)
        
        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))
        q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))

            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
        use_lma: bool = False,
        q_chunk_size: Optional[int] = None,
        kv_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
            use_lma:
                Whether to use low-memory attention
            q_chunk_size:
                Query chunk size (for LMA)
            kv_chunk_size:
                Key/Value chunk size (for LMA)
        Returns
            [*, Q, C_q] attention update
        """
        if biases is None:
            biases = []
        if use_lma and (q_chunk_size is None or kv_chunk_size is None):
            raise ValueError(
                "If use_lma is specified, q_chunk_size and kv_chunk_size must "
                "be provided"
            )

        # [*, Q/K, H, C_hidden]
        q, k, v = self._prep_qkv(q_x, kv_x)
       
        if use_lma:
            biases = [
                b.expand(b.shape[:-2] + (q_x.shape[-2],) + (kv_x.shape[-2],))
                for b in biases
            ]

            o = _lma(q, k, v, biases, q_chunk_size, kv_chunk_size)
        else:
            # compute MSE , mem, speed
            if self.use_attn: # use _attn, in triangular start attention
                o = _attention(q, k, v, biases)
            else: # use linear attn
                if GlobalConfig.FLOW_ATTN:
                    o = self.flowformer(q, k, v, biases,use_pair_bias=self.use_pair_bias)

                # elif GlobalConfig.PER_ATTN:
                #     o = self.performer(q,k,v,biases,use_pair_bias=self.use_pair_bias)
                
                else:
                    o = _attention(q, k, v, biases)    
            

        o = self._wrap_up(o, q_x)

        return o

#===========================Performer===================================
# cross attention
class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features = None, ortho_scaling = 0, causal = False, generalized_attention = False, kernel_fn = nn.ReLU(), no_projection = False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))
        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

        self.causal = causal
        if causal:
            try:
                import fast_transformers.causal_product.causal_product_cuda
                self.causal_linear_fn = partial(causal_linear_attention)
            except ImportError:
                print('unable to import cuda code for auto-regressive Performer. will default to the memory inefficient non-cuda version')
                self.causal_linear_fn = causal_linear_attention_noncuda
        

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device = device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v,biases,is_global=False, use_pair_bias=True):
        device = q.device
        q_shape = q.shape
        k_shape = k.shape
    
        if is_global: 
            q = q.view(-1, *q_shape[-3:])
            k = k.view(-1, *k_shape[-3:])
            v = v.view(-1, *k_shape[-3:])
        else:
            # q [*,H,Q,C_hidden]
            q = q.view(-1, *q_shape[-3:]).transpose(-2,-3)
            k = k.view(-1, *k_shape[-3:]).transpose(-2,-3)
            v = v.view(-1, *k_shape[-3:]).transpose(-2,-3)
        # add bias
        # mask: [1,1,K], pair_bias: [2,*,H,Q,r]
        if GlobalConfig.PAIR_FACTOR:
            if len(biases) == 1 or (len(biases) == 2 and not use_pair_bias):
                # mask biases q[H,Q,C_hidden+1]
                q_bias = torch.ones([*q.shape[:-1],1]).to(q.device)
                k_bias = biases[0].transpose(-1,-2)
                k_bias_shape = k_bias.shape
                k_bias = k_bias.reshape(-1,*k_bias_shape[-3:]).repeat(1,k_shape[-2],1,1)
                q = torch.concat([q,q_bias],dim=-1)
                k = torch.concat([k,k_bias],dim=-1)
            elif len(biases) ==2:
                pair_shape= biases[1].shape
                q_bias = torch.ones([*q.shape[:-1],1]).to(q.device)
                k_bias = biases[0].transpose(-1,-2)
                k_bias_shape = k_bias.shape
                #[*,H,K,1]
                k_bias = k_bias.view(-1,*k_bias_shape[-3:]).repeat(1,k_shape[-2],1,1)
                q = torch.concat([q, q_bias, biases[1][0].repeat(1,q_shape[1],1,1,1).view(-1,*pair_shape[-3:])],dim=-1)
                k = torch.concat([k, k_bias, biases[1][1].repeat(1,q_shape[1],1,1,1).view(-1,*pair_shape[-3:])],dim=-1)
            else: # no bias
                q_bias = torch.zeros([*q.shape[:-1],1]).to(q.device)
                k_bias = torch.zeros([*k.shape[:-1],1]).to(k.device)
                q = torch.concat([q,q_bias],dim=-1)
                k = torch.concat([k,k_bias],dim=-1)


       
       
        create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
        q = create_kernel(q, is_query = True)
        k = create_kernel(k, is_query = False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v).transpose(-2,-3)
    
        out = out.view(*q_shape)
        return out

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):
    
    b, h, *_ = data.shape
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)
  
    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h = h)
    mem = torch.cuda.memory_allocated()
    

    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)
    mem = torch.cuda.memory_allocated()
   

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()) + eps)
    mem = torch.cuda.memory_allocated()
  
    return data_dash.type_as(data)

def generalized_kernel(data, *, projection_matrix, kernel_fn = nn.ReLU(), kernel_epsilon = 0.001, normalize_data = True, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)

def orthogonal_matrix_chunk(cols, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    if TORCH_GE_1_8_0:
        # torch.linalg.qr: decompose unstructured_block : A=QR
        q, r = torch.linalg.qr(unstructured_block.cpu(), mode = 'reduced')
    else:
        q, r = torch.qr(unstructured_block.cpu(), some = True)
    # q is orthogonal, r is upper triangular wigh real diagonal
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, device = None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q[:remaining_rows])
    # [nb_rows,nb_columns]
    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix

# non-causal linear attention
def linear_attention(q, k, v):
    mem0 = torch.cuda.memory_allocated()
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    mem = torch.cuda.memory_allocated()
    context = torch.einsum('...nd,...ne->...de', k, v)
    mem = torch.cuda.memory_allocated()
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    mem = torch.cuda.memory_allocated()
    # print(f'performer max mem :{mem}')
    return out




# =========================Flowformer====================================
class Flow_Attention(nn.Module):
    # flow attention in normal version
    def __init__(self,drop_out=0.05, eps=1e-6):  # d_
        super(Flow_Attention, self).__init__()
        # self.n_heads = n_heads
        # self.out_projection = nn.Linear(d_model, d_output)
        self.dropout = nn.Dropout(drop_out)
        self.eps = eps

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def dot_product(self, q, k, v):
        kv = torch.einsum("nhld,nhlm->nhdm", k, v)
        qkv = torch.einsum("nhld,nhdm->nhlm", q, kv)
        return qkv

    def forward(self, queries, keys, values,biases,is_global=False,use_pair_bias=True):
       
        shape_q = queries.shape #[*,Q,h,d]
        shape_k = keys.shape
        if is_global:
            queries = queries.view(-1,*shape_q[-3:])
            keys = keys.view(-1,*shape_k[-3:])
            values = values.view(-1,*shape_k[-3:])
            if GlobalConfig.PAIR_FACTOR:
                # q_bias [*,Q,H,C_hidden+1]
                q_bias = torch.ones([*queries.shape[:-1],1]).to(queries.device)
                k_bias = biases[0].transpose(-1,-2)
                # k_bias [*,K,seq,C_hidden+1]
                queries = torch.concat([queries,q_bias],dim=-1)
                try:
                    keys = torch.concat([keys,k_bias],dim=-1)
                except:
                    k_bias = k_bias[...,None].view(*keys.shape[:-1],-1)
                    keys = torch.concat([keys,k_bias],dim=-1)

                    
        else: 
        # [*,h,Q,d]
            queries = queries.view(-1,*shape_q[-3:]).transpose(-2,-3)
            keys = keys.view(-1,*shape_k[-3:]).transpose(-2, -3)
            values = values.view(-1,*shape_k[-3:]).transpose(-2, -3)
            # add bias
            if GlobalConfig.PAIR_FACTOR:
                if len(biases) == 1 or (len(biases) == 2 and not use_pair_bias):
                    # mask biases q[H,Q,C_hidden+1]
                    q_bias = torch.ones([*queries.shape[:-1],1]).to(queries.device)
                    k_bias = biases[0].transpose(-1,-2)
                    k_bias_shape = k_bias.shape
                    k_bias = k_bias.reshape(-1,*k_bias_shape[-3:]).repeat(1,shape_k[-2],1,1)
                    queries = torch.concat([queries,q_bias],dim=-1)
                    keys = torch.concat([keys,k_bias],dim=-1)

                elif len(biases) ==2:
                    pair_shape= biases[1].shape
                    q_bias = torch.ones([*queries.shape[:-1],1]).to(queries.device)
                    k_bias = biases[0].transpose(-1,-2)
                    k_bias_shape = k_bias.shape
                    #[*,H,K,1]
                    k_bias = k_bias.view(-1,*k_bias_shape[-3:]).repeat(1,shape_k[-2],1,1)
                    try:
                        queries = torch.concat([queries, q_bias, biases[1][0].repeat(1,shape_q[1],1,1,1).view(-1,*pair_shape[-3:])],dim=-1)
                        keys = torch.concat([keys, k_bias, biases[1][1].repeat(1,shape_q[1],1,1,1).view(-1,*pair_shape[-3:])],dim=-1)
                    except: # inference
                        queries = torch.concat([queries, q_bias, biases[1][0][0].repeat(shape_q[0],1,1,1).view(-1,*pair_shape[-3:])],dim=-1)
                        keys = torch.concat([keys, k_bias, biases[1][1][0].repeat(shape_q[0],1,1,1).view(-1,*pair_shape[-3:])],dim=-1)
                        
                        
                else: # no bias
                    q_bias = torch.zeros([*queries.shape[:-1],1]).to(q.device)
                    k_bias = torch.zeros([*keys.shape[:-1],1]).to(k.device)
                    queries = torch.concat([queries,q_bias],dim=-1)
                    keys = torch.concat([keys,k_bias],dim=-1)
        # 2. Non-negative projection
        queries = self.kernel_method(queries) # [b,h,seq,c]
        keys = self.kernel_method(keys) # [b,res,seq,c]
        ## 3. Flow-Attention
        # (1) Calculate incoming and outgoing flow
        sink_incoming = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + self.eps, keys.sum(dim=-2) + self.eps))
        source_outgoing = 1.0 / (torch.einsum("nhld,nhd->nhl", keys + self.eps, queries.sum(dim=-2) + self.eps))
        # (2) conservation refine for source and sink
        conserved_sink = torch.einsum("nhld,nhd->nhl", queries + self.eps,
                                      (keys * source_outgoing[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.einsum("nhld,nhd->nhl", keys + self.eps,
                                        (queries * sink_incoming[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
        # (3) Competition & Allocation
        sink_allocation = torch.sigmoid(conserved_sink * (float(queries.shape[2]) / float(keys.shape[2])))
        source_competition = torch.softmax(conserved_source, dim=-1) * float(keys.shape[2])
        # (4) dot product
        x = (self.dot_product(queries * sink_incoming[:, :, :, None],  # for value normalization
                              keys,
                              values * source_competition[:, :, :, None])  # competition
             * sink_allocation[:, :, :, None]).transpose(-2, -3)  # allocation
        ## (5) Final projection
        x = x.reshape(*shape_q)
        x = self.dropout(x)
        return x
    






class GlobalAttention(nn.Module):
    """
    Alg.19 MSA Column Global Attention
    note:
        the implementation is a little different from Alg.19.
    """

    def __init__(self, c_in, c_hidden, no_heads, inf, eps):
        super(GlobalAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.eps = eps

        self.linear_q = Linear(c_in, c_hidden * no_heads, bias=False, init="glorot")

        self.linear_k = Linear(
            c_in,
            c_hidden,
            bias=False,
            init="glorot",
        )
        self.linear_v = Linear(
            c_in,
            c_hidden,
            bias=False,
            init="glorot",
        )
  
        if GlobalConfig.FLOW_ATTN :
            self.flowformer = Flow_Attention()
        self.linear_g = Linear(c_in, c_hidden * no_heads, init="gating")
        self.linear_o = Linear(c_hidden * no_heads, c_in, init="final")

        self.sigmoid = nn.Sigmoid()

    def forward(self, m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        the following implementation is different from Alg.19
        """
        # [*, N_res, C_in]
        q = torch.sum(m * mask.unsqueeze(-1), dim=-2) / (
            torch.sum(mask, dim=-1)[..., None] + self.eps
        )

        # [*, N_res, H * C_hidden]
        q = self.linear_q(q)
        q *= self.c_hidden ** (-0.5)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, N_seq, C_hidden]
        k = self.linear_k(m)
        v = self.linear_v(m)

        # [*, N_res, H, C_hidden]
        bias = (self.inf * (mask - 1))[..., :, None, :]
        if GlobalConfig.FLOW_ATTN :
            o = self.flowformer(q,k,v,[bias],is_global=True)
        else:
            a = torch.matmul(
            q,
            k.transpose(-1, -2),  # [*, N_res, C_hidden, N_seq]
            )
        
            a += bias
            a = softmax(a)
        
            # [*, N_res, H, C_hidden]
            o = torch.matmul(
            a,
            v,
            )


        # [*, N_res, N_seq, C_hidden]
        g = self.sigmoid(self.linear_g(m))

        # [*, N_res, N_seq, H, C_hidden]
        g = g.view(g.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, N_seq, H, C_hidden]
        o = o.unsqueeze(-3) * g

        # [*, N_res, N_seq, H * C_hidden]
        o = o.reshape(o.shape[:-2] + (-1,))

        # [*, N_res, N_seq, C_in]
        m = self.linear_o(o)

        return m





def _lma(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    biases: List[torch.Tensor],
    q_chunk_size: int,
    kv_chunk_size: int,
):
    no_q, no_kv = q.shape[-3], k.shape[-3]

    # [*, Q, H, C_hidden]
    o = q.new_zeros(q.shape)
    for q_s in range(0, no_q, q_chunk_size):
        q_chunk = q[..., q_s : q_s + q_chunk_size, :, :]
        large_bias_chunks = [b[..., q_s : q_s + q_chunk_size, :] for b in biases]

        maxes = []
        weights = []
        values = []
        for kv_s in range(0, no_kv, kv_chunk_size):
            k_chunk = k[..., kv_s : kv_s + kv_chunk_size, :, :]
            v_chunk = v[..., kv_s : kv_s + kv_chunk_size, :, :]
            small_bias_chunks = [
                b[..., kv_s : kv_s + kv_chunk_size] for b in large_bias_chunks
            ]

            a = torch.einsum(
                "...qhd,...khd->...hqk",
                q_chunk,
                k_chunk,
            )

            for b in small_bias_chunks:
                a += b

            a = a.transpose(-2, -3)

            max_a = torch.max(a, dim=-1, keepdim=True)[0]
            exp_a = torch.exp(a - max_a)
            exp_v = torch.einsum("...vhf,...qhv->...qhf", v_chunk, exp_a)

            maxes.append(max_a.detach().squeeze(-1))
            weights.append(torch.sum(exp_a, dim=-1))
            values.append(exp_v)

        chunk_max = torch.stack(maxes, dim=-3)
        chunk_weights = torch.stack(weights, dim=-3)
        chunk_values = torch.stack(values, dim=-4)

        global_max = torch.max(chunk_max, dim=-3, keepdim=True)[0]
        max_diffs = torch.exp(chunk_max - global_max)
        chunk_values *= max_diffs.unsqueeze(-1)
        chunk_weights *= max_diffs

        all_values = torch.sum(chunk_values, dim=-4)
        all_weights = torch.sum(chunk_weights.unsqueeze(-1), dim=-4)

        q_chunk_out = all_values / all_weights

        o[..., q_s : q_s + q_chunk_size, :, :] = q_chunk_out

    return o

class VariableLinear(nn.Module):
    def __init__(self, n_res: int, r: int):
        super(VariableLinear,self).__init__()
        self.crop_size = n_res
        self.low_rank = r
        self.factor_u_l1 = Linear(n_res,int(r/4))
        self.factor_u_l2 = Linear(int(n_res/2),int(r/4))
        self.factor_u_merge = Linear(r,r)
    
    def forward(self, z:torch.Tensor):
        #[*,c_z,N_res,N_res]
        n_res = z.shape[-1]
        if n_res < self.crop_size: # padding for l1
            l_pad = int((self.crop_size - n_res)/2)
            padding = (l_pad,l_pad+1)
            u = F.pad(z, padding, "constant", 0)
        else:
            u = z
        # l1
        stride = n_res - self.crop_size if self.crop_size < n_res else 0
        u1 = self.factor_u_l1(u[...,:self.crop_size])
        u2 = self.factor_u_l1(u[...,stride : stride + self.crop_size])
        # l2
        stride = n_res - int(self.crop_size /2 ) if n_res > int(self.crop_size /2 ) else 0
        u3 = self.factor_u_l2(u[...,:int(self.crop_size/2)])
        u4 = self.factor_u_l2(u[...,stride : stride + int(self.crop_size/2)])
        # r*r
        tu = self.factor_u_merge(torch.cat([u1,u2,u3,u4],dim=-1))
    

        return  tu 