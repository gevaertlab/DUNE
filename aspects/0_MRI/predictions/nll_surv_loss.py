# adapted from https://github.com/mahmoodlab/PORPOISE/blob/master/utils/loss_func.py

import torch
import torch.nn as nn
import torch
from sksurv.metrics import concordance_index_censored


class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
        Set importance of censored loss
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """

    def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, h, y, c):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y: (n_batches, 1)
            The true time bin index label.
        c: (n_batches, 1)
            The censoring status indicator.
        """

        return nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)


def nll_loss(h, y, c, alpha=0.0, eps=1e-6, reduction='mean'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    y: (n_batches, 1)
        The true time bin index label.
    c: (n_batches, 1)
        The censoring status indicator.
    alpha: float
        Set importance of censored loss
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """
    # print("h shape", h.shape)

    # make sure these are ints
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(h)

    S = torch.cumprod(1 - hazards, dim=1)

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    
    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)

    # c = 0 is uncensored (deceased)
    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    # c = 1 is censored (living)
    censored_loss = - c * torch.log(s_this)

    # neg_l = censored_loss + uncensored_loss
    # if alpha is not None:
    #     loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = (1 - alpha)*censored_loss + uncensored_loss


    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss


def get_ci(predictions, vital_status, survival_months):
    predictions = torch.tensor(predictions)
    hazards = torch.sigmoid(predictions)
    survival = torch.cumprod(1 - hazards, dim=-1)
    risk_all = -torch.sum(survival, dim=-1).detach().cpu().numpy().flatten()

    conc_index = concordance_index_censored(
        vital_status.astype(bool), survival_months, risk_all, tied_tol=1e-08)[0]

    return conc_index
