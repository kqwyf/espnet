from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import List
from typing import Tuple
from typing import Dict

import torch
from espnet2.asr.encoder.abs_encoder import AbsEncoder


class AbsAVEncoder(AbsEncoder, ABC):
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def output_size_v(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        additional: Dict,
        prev_states: torch.Tensor = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[torch.Tensor], List[Dict]]:
        raise NotImplementedError
