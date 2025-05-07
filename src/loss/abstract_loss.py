import abc

import torch
import torch.nn as nn
from jaxtyping import Float


class AbstractLoss(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(
        self,
        pred: Float[torch.Tensor, "B C H W"],
        gt: Float[torch.Tensor, "B C H W"],
        step: int,
        **kwargs,
    ) -> Float[torch.Tensor, ""]:
        pass

    def reset(self):
        pass
