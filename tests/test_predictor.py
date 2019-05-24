from stacked_hourglass import HumanPosePredictor, hg2
import pytest
import torch
from torch.testing import assert_allclose


ALL_DEVICES = ['cpu']
# Add available GPU devices.
ALL_DEVICES.extend(f'cuda:{i}' for i in range(torch.cuda.device_count()))


@pytest.mark.parametrize('device', ALL_DEVICES)
def test_do_forward(device, example_input):
    device = torch.device(device)
    model = hg2(pretrained=True)
    predictor = HumanPosePredictor(model, device=device)
    output = predictor.do_forward(example_input.to(device))
    assert len(output) == 2  # Expect one set of heatmaps per stack.
    heatmaps = output[-1]
    assert heatmaps.shape == (1, 16, 64, 64)


@pytest.mark.parametrize('device', ALL_DEVICES)
def test_prepare_image(device, man_running_image):
    device = torch.device(device)
    model = hg2(pretrained=True)
    predictor = HumanPosePredictor(model, device=device)
    image = predictor.prepare_image(man_running_image)
    assert image.shape == (3, 256, 256)
    assert image.device.type == 'cpu'


@pytest.mark.parametrize('device', ALL_DEVICES)
def test_estimate_heatmaps(device, man_running_image):
    device = torch.device(device)
    model = hg2(pretrained=True)
    predictor = HumanPosePredictor(model, device=device)
    heatmaps = predictor.estimate_heatmaps(man_running_image)
    assert heatmaps.shape == (16, 64, 64)


@pytest.mark.parametrize('device', ALL_DEVICES)
def test_estimate_joints(device, man_running_image, man_running_pose):
    device = torch.device(device)
    model = hg2(pretrained=True)
    predictor = HumanPosePredictor(model, device=device)
    joints = predictor.estimate_joints(man_running_image)
    assert joints.shape == (16, 2)
    assert_allclose(joints, man_running_pose, rtol=0, atol=20)


@pytest.mark.parametrize('device', ALL_DEVICES)
def test_estimate_joints_with_flip(device, man_running_image, man_running_pose):
    device = torch.device(device)
    model = hg2(pretrained=True)
    predictor = HumanPosePredictor(model, device=device)
    joints = predictor.estimate_joints(man_running_image, flip=True)
    assert joints.shape == (16, 2)
    assert_allclose(joints, man_running_pose, rtol=0, atol=20)
