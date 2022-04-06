import numpy as np
from nn_lib.mdl.loss import Loss
from nn_lib import Tensor
import nn_lib.tensor_fns as F


class CELoss(Loss):
    """
    cross entropy loss
    """
    # In order to avoid over- or underflow we clip prediction logits into [-MAX_LOG, MAX_LOG]
    MAX_LOG = 50

    def _clip(self, value: Tensor) ->Tensor:
        return F.clip(value, Tensor(-self.MAX_LOG, True), Tensor(self.MAX_LOG, True))

    def forward(self, prediction_logits: Tensor, target: Tensor) -> Tensor:
        """
        Compute a loss Tensor based on logit predictions and ground truth labels
        """
        max_log = Tensor(self.MAX_LOG, requires_grad=True)
        softmax_clip_pred = F.softmax(F.clip(prediction_logits, -max_log, max_log))
        sum = -F.reduce(F.log(softmax_clip_pred) * target, axis=1, keepdims=True)
        return F.reduce(sum)