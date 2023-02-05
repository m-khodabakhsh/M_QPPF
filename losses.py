import torch
import torch.nn.functional as F
from torch import nn

class RankListWiseLoss():
    def __init__(self,device):
        self.device = device
    def loss(self, y_pred: torch.Tensor, y_true: [torch.Tensor]) -> torch.Tensor:
        y_pred_soft = F.softmax(y_pred)
        y_true_soft = F.softmax(torch.Tensor(y_true))
        kl_loss = nn.KLDivLoss(size_average=False)(y_pred_soft.log(), y_true_soft.to(self.device))
        return (kl_loss)

class QPPLoss():
    def __init__(self,device):
        self.device = device
    def loss(self, y_pred: torch.Tensor, y_true: [torch.Tensor]) -> torch.Tensor:
        qpp_loss = nn.MSELoss()(y_pred,torch.Tensor(y_true).to(self.device))
        return (qpp_loss)