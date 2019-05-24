from pathlib import Path

import pytest
import torch

from stacked_hourglass.utils.imutils import load_image, resize
from stacked_hourglass.utils.transforms import color_normalize

DATA_DIR = Path(__file__).parent.joinpath('data')


@pytest.fixture()
def man_running_image():
    return load_image(str(DATA_DIR.joinpath('man_running.jpg')))


@pytest.fixture()
def man_running_pose():
    return torch.as_tensor([
        [215, 449],  # right_ankle
        [214, 345],  # right_knee
        [211, 244],  # right_hip
        [266, 244],  # left_hip
        [258, 371],  # left_knee
        [239, 438],  # left_ankle
        [237, 244],  # pelvis
        [244, 113],  # spine
        [244,  94],  # neck
        [244,  24],  # head_top
        [179, 198],  # right_wrist
        [182, 142],  # right_elbow
        [199, 103],  # right_shoulder
        [296, 105],  # left_shoulder
        [330, 171],  # left_elbow
        [299, 165],  # left_wrist
    ], dtype=torch.float32)


@pytest.fixture()
def example_input(man_running_image):
    mean = torch.as_tensor([0.4404, 0.4440, 0.4327])
    std = torch.as_tensor([0.2458, 0.2410, 0.2468])
    image = resize(man_running_image, 256, 256)
    image = color_normalize(image, mean, std)
    return image.unsqueeze(0)
