import sys
import os
import pytest
from torchvision.models import MobileNet_V2_Weights

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mobilenetv2 import MobileNetV2


@pytest.mark.parametrize("width_mult", [1.0])
def test_mobilenetv2_structure(width_mult):

    weights = MobileNet_V2_Weights.DEFAULT
    official_state = weights.get_state_dict()

    model = MobileNetV2(0.1, 1000)
    my_state = model.state_dict()

    # check number of parameters
    assert len(my_state) == len(official_state), \
        f"Different number of parameters: {len(my_state)} vs {len(official_state)}"

    # check parameter types and shapes
    for (k1, v1), (k2, v2) in zip(my_state.items(), official_state.items()):

        assert k1.split('.')[-1] == k2.split('.')[-1], \
            f"Parameter type mismatch: {k1} vs {k2}"

        assert v1.shape == v2.shape, \
            f"Shape mismatch: {k1} {v1.shape} vs {k2} {v2.shape}"