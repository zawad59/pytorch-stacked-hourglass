import pytest
import torch

from stacked_hourglass import hg2
from stacked_hourglass.train import do_training_step, do_validation_step
from torch.optim import Adam

ALL_DEVICES = ['cpu']
# Add available GPU devices.
ALL_DEVICES.extend(f'cuda:{i}' for i in range(torch.cuda.device_count()))


@pytest.mark.parametrize('device', ALL_DEVICES)
def test_do_training_step(device):
    device = torch.device(device)
    model = hg2(pretrained=False)
    model = model.to(device)
    model.train()
    optimiser = Adam(model.parameters())
    inp = torch.randn((1, 3, 256, 256), device=device)
    target = torch.randn((1, 16, 64, 64), device=device)
    output, loss = do_training_step(model, optimiser, inp, target)
    assert output.shape == (1, 16, 64, 64)
    assert loss > 0


@pytest.mark.parametrize('device', ALL_DEVICES)
def test_do_validation_step(device):
    device = torch.device(device)
    model = hg2(pretrained=False)
    model = model.to(device)
    model.eval()
    inp = torch.randn((1, 3, 256, 256), device=device)
    target = torch.randn((1, 16, 64, 64), device=device)
    output, loss = do_validation_step(model, inp, target)
    assert output.shape == (1, 16, 64, 64)
    assert loss > 0
