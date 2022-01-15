import torch

def RMSE(yHat, yTrue):
    return torch.sqrt(torch.mean((yTrue - yHat)**2))