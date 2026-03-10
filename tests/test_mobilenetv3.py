import sys
import os
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mobilenetv3 import MobileNetV3Large, MobileNetV3Small


def test_mobilenetv3_structure():
    weights = MobileNet_V3_Large_Weights.DEFAULT
    official_state = weights.get_state_dict()

    model = MobileNetV3Large(0.1, 1000, 1280)
    my_state = model.state_dict()

    
    assert len(my_state) == len(official_state), "Different number of parameters"

    for (k1, v1), (k2, v2) in zip(my_state.items(), official_state.items()):

        #we need just to know if it is bias/weight etc 
        assert k1.split('.')[-1] == k2.split('.')[-1], \
            f"Parameter type mismatch: {k1} vs {k2}"

        # check tensor shape
        assert v1.shape == v2.shape, \
            f"Shape mismatch: {k1} {v1.shape} vs {k2} {v2.shape}"


def test_mobilenetv3_small_structure():
    weights = MobileNet_V3_Small_Weights.DEFAULT
    official_state = weights.get_state_dict()

    model = MobileNetV3Small(0.1, 1000, 1024)
    my_state = model.state_dict()

    
    assert len(my_state) == len(official_state), "Different number of parameters"

    for (k1, v1), (k2, v2) in zip(my_state.items(), official_state.items()):

        #we need just to know if it is bias/weight etc 
        assert k1.split('.')[-1] == k2.split('.')[-1], \
            f"Parameter type mismatch: {k1} vs {k2}"

        # check tensor shape
        assert v1.shape == v2.shape, \
            f"Shape mismatch: {k1} {v1.shape} vs {k2} {v2.shape}"