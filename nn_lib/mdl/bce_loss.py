from nn_lib.mdl.loss import Loss
from nn_lib import Tensor
import nn_lib.tensor_fns as F


class BCELoss(Loss):
    """
    Binary cross entropy loss
    Similar to this https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    """

    # In order to avoid over- or underflow we clip prediction logits into [-MAX_LOG, MAX_LOG]
    MAX_LOG = 50

    def forward(self, prediction_logits: Tensor, target: Tensor) -> Tensor:
        """
        Compute a loss Tensor based on logit predictions and ground truth labels
        :param prediction_logits: prediction logits returned by a model (i.e. sigmoid argument) of shape (B,)
        :param target: binary ground truth labels of shape (B,)
        :return: a loss Tensor; if reduction is True, returns a scalar, otherwise a Tensor of shape (B,) -- loss value
            per batch element
        """

        activ_pred = F.sigmoid(prediction_logits)
        one = Tensor(1, requires_grad=True)
        max_log = Tensor(self.MAX_LOG, requires_grad=True)
        log1 = F.clip(F.log(activ_pred), -max_log, max_log)
        log2 = F.clip(F.log(one - activ_pred), -max_log, max_log)

        res = -(target * log1 + (one - target) * log2)

        if self.reduce:
            return F.reduce(res)
        return res