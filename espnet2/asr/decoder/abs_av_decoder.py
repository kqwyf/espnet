from abc import ABC
from abc import abstractmethod
from typing import Tuple
from typing import Dict

import torch

from espnet.nets.scorer_interface import ScorerInterface
from espnet2.asr.decoder.abs_decoder import AbsDecoder


class AbsAVDecoder(AbsDecoder, ScorerInterface, ABC):
    @abstractmethod
    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        additional: Dict,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
