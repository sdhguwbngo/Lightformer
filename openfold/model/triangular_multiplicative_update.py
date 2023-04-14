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

from functools import partialmethod
from typing import Optional

import torch
import torch.nn as nn

from openfold.model.primitives import Linear, LayerNorm, VariableLinear
from openfold.utils.tensor_utils import permute_final_dims
# from openfold.model.embedders import VariableLinear
from openfold.utils.logger import Logger

logger = Logger.logger

from openfold.tool.global_config import GlobalConfig
import pdb

class TriangleMultiplicativeUpdate(nn.Module):
    """
    Implements Algorithms 11 and 12.
    """

    def __init__(self, c_z, c_hidden, _outgoing=True,rank_factor=0,crop_size=0):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(TriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing
        self.rank_factor = rank_factor
        self.crop_size = crop_size
        self.linear_a_p = Linear(self.c_z, self.c_hidden)
        self.linear_a_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_b_p = Linear(self.c_z, self.c_hidden)
        self.linear_b_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_g = Linear(self.c_z, self.c_z, init="gating")
        self.linear_z = Linear(self.c_hidden, self.c_z, init="final")

        self.layer_norm_in = LayerNorm(self.c_z)
        self.layer_norm_out = LayerNorm(self.c_hidden)
        self.sigmoid = nn.Sigmoid()

        if GlobalConfig.PAIR_FACTOR and self.rank_factor > 0:
            if self._outgoing:
                self.linear_uv = VariableLinear(self.crop_size,self.rank_factor)
            else:
                self.linear_uv = Linear(self.rank_factor, 2* self.crop_size)
        else:
            self.linear_uv = None
        

    def _combine_projections(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("This method needs to be overridden")

    def forward(
        self, z: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """

        # start to record max allocated memory in tri_mul
        if GlobalConfig.DEV_PROF_MEM_MODE:
            torch.cuda.reset_peak_memory_stats(device=0)

        # start to record running tri_mul time
        if GlobalConfig.DEV_PROF_TIME_MODE:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            starter.record()

        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        # mask: [*, N_res, N_res, 1]
        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        a = self.linear_a_p(z) * self.sigmoid(self.linear_a_g(z))
        # a: [*, N_res, N_res, C_z]
        a = a * mask
        b = self.linear_b_p(z) * self.sigmoid(self.linear_b_g(z))
        # b: [*, N_res, N_res, C_z]
        b = b * mask
        # tri_mul_out and tri_mul_in are different here
        # a b [*,L,r,c] 
        
        x = self._combine_projections(a, b)
        
        if self.linear_uv is not None:
            if self._outgoing:
                x = self.linear_uv(x)
                # z = self.linear_uv(z.transpose(-1,-2)).transpose(-1,-2)
            else:
                x = self.linear_uv(x.transpose(-2,-1)).transpose(-1,-2)
                # z = self.linear_uv(z.transpose(-3,-1)).transpose(-1,-3)
        
        # [*, N_i, N_j, C]
        x = permute_final_dims(x, (1, 2, 0))
        # start to record allocated memory
        if GlobalConfig.DEV_PROF_MEM_MODE:
            logger.info(
                f"allocated memory before del a, b: {torch.cuda.memory_allocated(device=0)}"
            )

        # Possibly prevents memory fragmentation
        del a, b

        # start to record allocated memory
        if GlobalConfig.DEV_PROF_MEM_MODE:
            logger.info(
                f"allocated memory after del a, b: {torch.cuda.memory_allocated(device=0)}"
            )

        x = self.layer_norm_out(x)
        # x: [*, N_res, N_res, C_z]
        x = self.linear_z(x)
        # g: [*, N_res, N_res, C_z]
        g = self.sigmoid(self.linear_g(z))
        # z: [*, N_res, N_res, C_z]
        # val seq_len != crop_size, so starting node learning from r -> seq_len
        if not self._outgoing:
            seq_len = g.shape[-3]
            z = (x.transpose(-3,-1)[...,:seq_len] * g.transpose(-1,-3)).transpose(-1,-3)
   
        
        else:
            z = x * g
      
      
   

        # start to record allocated memory
        if GlobalConfig.DEV_PROF_MEM_MODE:
            logger.info(
                f"allocated memory before del x, g: {torch.cuda.memory_allocated(device=0)}"
            )

        # Possibly prevents memory fragmentation
        del x, g

        # stop to record allocated memory
        if GlobalConfig.DEV_PROF_MEM_MODE:
            logger.info(
                f"allocated memory after del x, g: {torch.cuda.memory_allocated(device=0)}"
            )

        # print max allocated memory in tri_mul
        if GlobalConfig.DEV_PROF_MEM_MODE:
            logger.info(
                f"max allocated memory in tri_mul: {torch.cuda.max_memory_allocated(device=0)}"
            )

        # stop to record running tri_mul time
        if GlobalConfig.DEV_PROF_TIME_MODE:
            ender.record()
            torch.cuda.synchronize()  # WAIT FOR GPU SYNC
            logger.info(f"init tri_mul time: {starter.elapsed_time(ender)/1000}s")

        return z


class FusedTriangleMultiplicativeUpdate(nn.Module):
    """
    Implements Algorithms 11 and 12. Updated in multimer_v3.
    """

    def __init__(self, c_z, c_hidden):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(FusedTriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden

        self.linear_a_p = Linear(self.c_z, 2 * self.c_hidden)
        self.linear_a_g = Linear(self.c_z, 2 * self.c_hidden, init="gating")
        self.linear_g = Linear(self.c_z, self.c_z, init="gating")
        self.linear_z = Linear(self.c_hidden, self.c_z, init="final")

        self.layer_norm_in = LayerNorm(self.c_z)
        self.layer_norm_out = LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def _combine_projections(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("This method needs to be overridden")

    def forward(
        self, z: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """

        # start to record max allocated memory in tri_mul
        if GlobalConfig.DEV_PROF_MEM_MODE:
            torch.cuda.reset_peak_memory_stats(device=0)

        # start to record running tri_mul time
        if GlobalConfig.DEV_PROF_TIME_MODE:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            starter.record()

        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        # mask: [*, N_res, N_res, 1]
        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        a = self.linear_a_p(z) * self.sigmoid(self.linear_a_g(z))
        # a: [*, N_res, N_res, C_z]
        a = a * mask

        b = a[..., self.c_hidden :]
        a = a[..., : self.c_hidden]

        # tri_mul_out and tri_mul_in are different here
        x = self._combine_projections(a, b)

        # start to record allocated memory
        if GlobalConfig.DEV_PROF_MEM_MODE:
            logger.info(
                f"allocated memory before del a, b: {torch.cuda.memory_allocated(device=0)}"
            )

        # Possibly prevents memory fragmentation
        del a, b

        # start to record allocated memory
        if GlobalConfig.DEV_PROF_MEM_MODE:
            logger.info(
                f"allocated memory after del a, b: {torch.cuda.memory_allocated(device=0)}"
            )

        x = self.layer_norm_out(x)
        # x: [*, N_res, N_res, C_z]
        x = self.linear_z(x)
        # g: [*, N_res, N_res, C_z]
        g = self.sigmoid(self.linear_g(z))
        # z: [*, N_res, N_res, C_z]
        z = x * g

        # start to record allocated memory
        if GlobalConfig.DEV_PROF_MEM_MODE:
            logger.info(
                f"allocated memory before del x, g: {torch.cuda.memory_allocated(device=0)}"
            )

        # Possibly prevents memory fragmentation
        del x, g

        # stop to record allocated memory
        if GlobalConfig.DEV_PROF_MEM_MODE:
            logger.info(
                f"allocated memory after del x, g: {torch.cuda.memory_allocated(device=0)}"
            )

        # print max allocated memory in tri_mul
        if GlobalConfig.DEV_PROF_MEM_MODE:
            logger.info(
                f"max allocated memory in tri_mul: {torch.cuda.max_memory_allocated(device=0)}"
            )

        # stop to record running tri_mul time
        if GlobalConfig.DEV_PROF_TIME_MODE:
            ender.record()
            torch.cuda.synchronize()  # WAIT FOR GPU SYNC
            logger.info(f"init tri_mul time: {starter.elapsed_time(ender)/1000}s")

        return z


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 11. This is L^2r and should be linear one with performer or flowformer
    """

    def _combine_projections(
        self,
        a: torch.Tensor,  # [*, N_i, N_k, C]
        b: torch.Tensor,  # [*, N_j, N_k, C]
    ):
        # [*, C, N_i, N_j]
        p = torch.matmul(
            permute_final_dims(a, (2, 0, 1)),
            permute_final_dims(b, (2, 1, 0)),
        )
        
        return p


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 12.
    """

    def _combine_projections(
        self,
        a: torch.Tensor,  # [*, N_k, N_i, C]
        b: torch.Tensor,  # [*, N_k, N_j, C]
    ):
        # [*, C, N_i, N_j]
        p = torch.matmul(
            permute_final_dims(a, (2, 1, 0)),
            permute_final_dims(b, (2, 0, 1)),
        )

    
        return p

#         # [*, N_i, N_j, C]
#         return permute_final_dims(p, (1, 2, 0))


class FusedTriangleMultiplicationOutgoing(FusedTriangleMultiplicativeUpdate):
    """
    Implements Algorithm 11. Updated in multimer_v3.
    """

    def _combine_projections(
        self,
        a: torch.Tensor,  # [*, N_i, N_k, C]
        b: torch.Tensor,  # [*, N_j, N_k, C]
    ):
        # [*, C, N_i, N_j]
        p = torch.matmul(
            permute_final_dims(a, (2, 0, 1)),
            permute_final_dims(b, (2, 1, 0)),
        )
        # [*, N_i, N_j, C]
        return permute_final_dims(p, (1, 2, 0))


class FusedTriangleMultiplicationIncoming(FusedTriangleMultiplicativeUpdate):
    """
    Implements Algorithm 12. Updated in multimer_v3.
    """

    def _combine_projections(
        self,
        a: torch.Tensor,  # [*, N_k, N_i, C]
        b: torch.Tensor,  # [*, N_k, N_j, C]
    ):
        # [*, C, N_i, N_j]
        # see commit b88f8da on the Alphafold repo
        # Alphafold swaps the pseudocode's a and b between the incoming/outcoming
        # iterations of triangle multiplication
        p = torch.matmul(
            permute_final_dims(b, (2, 1, 0)),
            permute_final_dims(a, (2, 0, 1)),
        )
        # [*, N_i, N_j, C]
        return permute_final_dims(p, (1, 2, 0))

