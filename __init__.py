from .mobilenetV3 import MobileNetv3_large
from .resnet50 import ResNet50
from .Xception import Xception
from .dmbottleneck import Dmbottleneck


get_model_from_name = {
    "mobilenet"     : MobileNetv3_large,
    "resnet50"      : ResNet50,
    "xception"      : Xception,
    "dmbottleneck": Dmbottleneck
}

freeze_layers = {
    "mobilenet"     : 192,
    "resnet50"      : 173,
    "xcepetion"     : 125,
}
