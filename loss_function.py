import torch
import torch.nn as nn

# class TripletLossWithHammingDistance(nn.Module):
#     def __init__(self, margin=4.0, _lambda=10.0):
#         super(TripletLossWithHammingDistance, self).__init__()
#         self.margin = margin
#         self._lambda = _lambda

#     def sgn(self, x):
#         return x.sgn()

#     def half_inner_product(self, x1, x2):
#         return (torch.dot(x1.reshape(-1, ), x2.reshape(-1, )) / 2.0)

#     def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
#         loss1 = 0
#         len_anchor = anchor.shape[1]
#         for i in range(anchor.shape[0]):
#             temp = (
#                 self.half_inner_product(anchor[i], positive[i]) / len_anchor
#                 - self.half_inner_product(anchor[i], negative[i]) / len_anchor
#                 - self.margin)
#             loss1 += - (temp - torch.log(1 + torch.exp(temp)))

#         loss2 = self._lambda * ((self.sgn(anchor) - anchor).pow(2).sum()) / len_anchor  # quantization error due to relaxing
#         # print('loss1: ', loss1)
#         # print('loss2: ', loss2)
#         losses = (loss1 + loss2) / anchor.shape[0]
#         return losses
    
class TripletLossWithHammingDistance(nn.Module):
    def __init__(self, dims, margin=4.0, lamb=0.01):
        super(TripletLossWithHammingDistance, self).__init__()
        self.dims = dims
        self.margin = margin
        self.lamb = lamb

    def dot_product(self, x1, x2):
        return (x1*x2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, all_out: torch.Tensor) -> torch.Tensor:       
        distance_positive = -self.dot_product(anchor, positive)
        distance_negative = self.dot_product(anchor, negative)
        loss1 = distance_positive + distance_negative + self.margin
        loss2 = self.lamb * ((torch.sign(all_out) - all_out).pow(2).sum())/self.dims

        return torch.relu(loss1 + loss2).mean()

    
class TripletLossEuclidean_Criteria(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, margin=2.0):
        super(TripletLossEuclidean_Criteria, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        print('Create euclidean loss with hyperparameters: a=%f b=%f m=%f'%(self.alpha, self.beta, self.margin))
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def dot_product(self, x1, x2):
        return (x1*x2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        additional_constraint = torch.abs(self.dot_product(anchor, negative))
        losses = torch.relu(self.alpha*distance_positive - distance_negative + self.beta*additional_constraint + self.margin)

        return losses.mean()
    
    
class TripletLossEuclidean_Criteria_V2(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, margin=2.0):
        super(TripletLossEuclidean_Criteria_V2, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        print('Create euclidean loss 2 with hyperparameters: a=%f b=%f m=%f'%(self.alpha, self.beta, self.margin))
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def dot_product(self, x1, x2):
        return (x1*x2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        additional_constraint = torch.square(self.dot_product(anchor, negative))
        losses = torch.relu(self.alpha*distance_positive - distance_negative + self.beta*additional_constraint + self.margin)

        return losses.mean()


class TripletLossHamming_Criteria(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, margin=2.0):
        super(TripletLossHamming_Criteria, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        print('Create hamming loss with hyperparameters: a=%f b=%f m=%f'%(self.alpha, self.beta, self.margin))
       
    def dot_product(self, x1, x2):
        return (x1*x2).sum(1)

    def calc_hamming_dist(self, x1, x2):
        return self.dot_product(x1, x2)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = -self.calc_hamming_dist(anchor, positive)
        distance_negative = self.calc_hamming_dist(anchor, negative)
        additional_constraint = torch.abs(self.dot_product(anchor, negative))
        losses = torch.relu(self.alpha*distance_positive + distance_negative + self.beta*additional_constraint + self.margin)

        return losses.mean()
    
class TripletLossHamming_Criteria_V2(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, margin=2.0):
        super(TripletLossHamming_Criteria_V2, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        print('Create hamming loss 2 with hyperparameters: a=%f b=%f m=%f'%(self.alpha, self.beta, self.margin))
       
    def dot_product(self, x1, x2):
        return (x1*x2).sum(1)

    def calc_hamming_dist(self, x1, x2):
        return self.dot_product(x1, x2)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = -self.calc_hamming_dist(anchor, positive)
        distance_negative = self.calc_hamming_dist(anchor, negative)
        additional_constraint = torch.square(self.dot_product(anchor, negative))
        losses = torch.relu(self.alpha*distance_positive + distance_negative + self.beta*additional_constraint + self.margin)

        return losses.mean()
    

class TripletLossHammingConstraint(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, margin=5.0):
        super(TripletLossHammingConstraint, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta

    def dot_product(self, x1, x2):
        return (x1*x2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:       
        distance_positive = -self.dot_product(anchor, positive)
        additional_constraint = torch.abs(self.dot_product(anchor, negative))
        loss = torch.relu(distance_positive + additional_constraint + self.margin)
        #  loss2 = self.lamb * ((torch.sign(all_out) - all_out).pow(2).sum())/self.dims

        return loss.mean()

    
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()