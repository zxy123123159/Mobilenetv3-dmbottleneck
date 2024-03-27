#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
from nets.Xcepetion import Xception
from nets.mobilenetV3 import MobileNetv3_large
from nets.resnet50 import ResNet50
from nets.dmbottleneck import Dmbottleneck

if __name__ == "__main__":
    # model = Xception([224, 224, 3], classes=2)
    # model.summary()
    # print("11111111111111111111111111111111111111111111111111111")

    model = Dmbottleneck([224, 224, 3], classes=2)
    model.summary()
    print("11111111111111111111111111111111111111111111111111111")

    # model = ResNet50([224, 224, 3], classes=2)
    # model.summary()
    # print("11111111111111111111111111111111111111111111111111111")

    # model = MobileNetv3_large([224, 224, 3], classes=2)
    # model.summary()
    # print("11111111111111111111111111111111111111111111111111111")

    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)
