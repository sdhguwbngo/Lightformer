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

from functools import partialmethod, partial
import math
from typing import Optional, List
import pdb
import torch
import torch.nn as nn

from openfold.model.primitives import Linear, LayerNorm, Attention
from openfold.utils.tensor_utils import (
    chunk_layer,
    permute_final_dims,
    flatten_final_dims,
)

from openfold.utils.logger import Logger

logger = Logger.logger

from openfold.tool.global_config import GlobalConfig


class TriangleAttention(nn.Module):
    def __init__(self, c_in, c_hidden, no_heads, starting, inf=1e9,rank_factor=-1):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Overall hidden channel dimension (not per-head)
            no_heads:
                Number of attention heads
        """
        super(TriangleAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf
        self.rank_factor=rank_factor

        self.layer_norm = LayerNorm(self.c_in)

        self.linear = Linear(c_in, self.no_heads, bias=False, init="normal")
        if self.starting:
            self.mha = Attention(
                self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads, use_attn=True, rank_factor=self.rank_factor)
        else:
            self.mha = Attention(
                self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads, use_attn=False, use_pair_bias= False,rank_factor=self.rank_factor)

    @torch.jit.ignore
    def _chunk(
        self,
        x: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
    ) -> torch.Tensor:
        mha_inputs = {
            "q_x": x,
            "kv_x": x,
            "biases": biases,
        }
        return chunk_layer(
            partial(self.mha),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, I, J, C_in] input tensor (e.g. the pair representation)
        Returns:
            [*, I, J, C_in] output tensor
        """

        # start to record running msa_transition time
        if GlobalConfig.DEV_PROF_TIME_MODE:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            starter.record()

        if mask is None:
            # [*, I, J]
            mask = x.new_ones(
                x.shape[:-1],
            )

        # Shape annotations assume self.starting. Else, I and J are flipped
        if not self.starting:
            # [*, J, I, C_in]
            x = x.transpose(-2, -3)
            # [*, J, 1, 1, I]
            mask = mask.transpose(-1, -2)

        """
        if tri_att_start: [*, I, J, C_in]
        else tri_att_end: [*, J, I, C_in]
        """
        x = self.layer_norm(x)

        """
        if tri_att_start: [*, I, 1, 1, J]
        else tri_att_end: [*, J, 1, 1, I]
        """
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        """
        if tri_att_start: [*, H, I, J]
        if tri_att_end: [*, H, J, I]
        """
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1))

        """
        if tri_att_start: [*, 1, H, I, J]
        if tri_att_end: [*, 1, H, J, I]
        """
        triangle_bias = triangle_bias.unsqueeze(-4)
        biases = [mask_bias, triangle_bias]

        if chunk_size is not None:
            x = self._chunk(x, biases, chunk_size)
        else:
            # x = self.mha(q_x=x, kv_x=x, biases=biases)
            if GlobalConfig.ADD_BIAS_V or GlobalConfig.ADD_BIAS_S:
                x = self.mha(q_x=x, kv_x=x, biases=biases)
            else:
                x = self.mha(q_x=x, kv_x=x, biases=[biases[0]])

        if not self.starting:
            # [*, I, J, C_in]
            x = x.transpose(-2, -3)

        # stop to record running msa_transition time
        if GlobalConfig.DEV_PROF_TIME_MODE:
            ender.record()
            torch.cuda.synchronize()  # WAIT FOR GPU SYNC
            logger.info(f"init model time: {starter.elapsed_time(ender)/1000}s")

        # [*, I, J, C_in]
        return x


class TriangleAttentionStartingNode(TriangleAttention):
    """
    Implements Algorithm 13.
    """

    __init__ = partialmethod(TriangleAttention.__init__, starting=True)


class TriangleAttentionEndingNode(TriangleAttention):
    """
    Implements Algorithm 14.
    """

    __init__ = partialmethod(TriangleAttention.__init__, starting=False)
