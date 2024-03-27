 ### 训练步骤
1. datasets文件夹下存放的图片分为两部分，train里面是训练图片，test里面是测试图片。  
2. 在训练之前需要首先准备好数据集，在train或者test文件里里面创建不同的文件夹，每个文件夹的名称为对应的类别名称，文件夹下面的图片为这个类的图片。文件格式可参考如下：
```
|-datasets
    |-train
        |-web1
            |-001.jpg
            |-002.jpg
        |-web2
            |-001.jpg
            |-002.jpg
        |-...
    |-test
        |-web1
            |-001.jpg
            |-002.jpg
        |-web2
            |-001.jpg
            |-002.jpg
        |-...
```
3. 之后修改model_data文件夹下的cls_classes.txt，使其也对应自己需要分的类。  
4. 在准备好数据集后，需要在根目录运行txt_annotation.py生成训练所需的cls_train.txt，运行前需要修改其中的classes_path，classes_path需要指向model_data下的txt文件，txt文件中是自己所要去区分的种类，将其修改成自己需要分的类。  
5. 在train.py里面调整自己要选择的网络和权重后，就可以开始训练了！  

### 预测步骤
#### a、使用预训练权重
1. model_data中已经分别存在resnet50、xception以及mobilenet50的预训练权重文件，直接更改路径即可使用。  

#### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在classification.py文件里面，在如下部分修改model_path、classes_path、backbone和alpha使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类，backbone对应使用的主干特征提取网络，alpha是当使用mobilenet的alpha值**。  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"    : 'model_data/mobilenet***.h5',
    "classes_path"  : 'model_data/cls_classes.txt',
    #--------------------------------------------------------------------#
    #   输入的图片大小
    #--------------------------------------------------------------------#
    "input_shape"   : [224, 224],
    #--------------------------------------------------------------------#
    #   所用模型种类：
    #   mobilenet、resnet50、xception是常用的分类网络
    #--------------------------------------------------------------------#
    "backbone"      : 'mobilenet',
    #--------------------------------------------------------------------#
    #   当使用mobilenet的alpha值
    #   仅在backbone='mobilenet'的时候有效
    #--------------------------------------------------------------------#
    "alpha"         : 0.25
}
```
3. 运行predict.py，输入  
```python
img/fake1.jpg
```  


### 评估步骤
1. datasets文件夹下存放的图片分为两部分，train里面是训练图片，test里面是测试图片，在评估的时候，我们使用的是test文件夹里面的图片。  
2. 在评估之前需要首先准备好数据集，在train或者test文件里里面创建不同的文件夹，每个文件夹的名称为对应的类别名称，文件夹下面的图片为这个类的图片。文件格式可参考如下：
```
|-datasets
    |-train
        |-cat
            |-001.jpg
            |-002.jpg
        |-dog
            |-001.jpg
            |-002.jpg
        |-...
    |-test
        |-cat
            |-001.jpg
            |-002.jpg
        |-dog
            |-001.jpg
            |-002.jpg
        |-...
```
3. 在准备好数据集后，需要在根目录运行txt_annotation.py生成训练所需的cls_test.txt，运行前需要修改其中的classes_path，classes_path需要指向model_data下的txt文件，txt文件中是自己所要去区分的种类，将其修改成自己需要分的类。  
4. 之后在classification.py文件里面修改如下部分model_path、classes_path、backbone和alpha使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类，backbone对应使用的主干特征提取网络，alpha是当使用mobilenet的alpha值**。  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"    : 'model_data/mobilenet025_catvsdog.h5',
    "classes_path"  : 'model_data/cls_classes.txt',
    #--------------------------------------------------------------------#
    #   输入的图片大小
    #--------------------------------------------------------------------#
    "input_shape"   : [224, 224],
    #--------------------------------------------------------------------#
    #   所用模型种类：
    #   mobilenet、resnet50、vgg16是常用的分类网络
    #--------------------------------------------------------------------#
    "backbone"      : 'mobilenet',
    #--------------------------------------------------------------------#
    #   当使用mobilenet的alpha值
    #   仅在backbone='mobilenet'的时候有效
    #--------------------------------------------------------------------#
    "alpha"         : 0.25
}
```
5. 运行eval.py来进行模型准确率评估。

### Reference
https://github.com/keras-team/keras-applications   
