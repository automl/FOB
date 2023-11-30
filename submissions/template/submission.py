import torch

def configure_optimizers(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return optimizer
