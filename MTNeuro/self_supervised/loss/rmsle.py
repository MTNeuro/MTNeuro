import torch


def MSE(pred, actual):
    return torch.sum((pred - actual) ** 2)

def RMSLE(pred, actual):
    return torch.sqrt(MSE(torch.log(pred + 1), torch.log(actual + 1)))