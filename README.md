# fastseg_pytorch
PyTorch implementation of MobileNetV3 for real-time semantic segmentation.
该仓库用于训练以MobileNetV3为主干，LR-ASPP为分割头的实时分割网络，训练模型的代码复现自[fastseg](https://github.com/ekzhang/fastseg)该仓库，不需要依赖任何的预训练权重，也更方便部署到onnx，并且也很轻松地转换到ncnn，便于部署到端侧。训练代码参考自[deeplabv3-plus-pytorch](https://github.com/bubbliiiing/deeplabv3-plus-pytorch)。


### 所需环境
torch==1.2.0    
torchvision==0.4.0   

### 训练步骤
#### 一、训练voc数据集
1、将我提供的voc数据集放入VOCdevkit中（无需运行voc_annotation.py）。  
2、运行train.py进行训练，默认参数已经对应voc数据集所需要的参数了。  

#### 二、训练自己的数据集
1、本文使用VOC格式进行训练。  
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的SegmentationClass中。    
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。    
4、在训练前利用voc_annotation.py文件生成对应的txt。    
5、注意修改train.py的num_classes为分类个数+1。    
6、运行train.py即可开始训练。  

#### 三、预测步骤
#### 一、使用预训练权重
##### a、VOC预训练权重
1. 下载完库后解压，如果想要利用voc训练好的权重进行预测，在百度网盘或者release下载权值，放入model_data，运行即可预测。  
```python
img/street.jpg
```    
2. 在predict.py里面进行设置可以进行fps测试和video视频检测。    
##### b、医药预训练权重
1. 下载完库后解压，如果想要利用医药数据集训练好的权重进行预测，在百度网盘或者release下载权值，放入model_data，修改unet.py中的model_path和num_classes；
```python
_defaults = {
    #-------------------------------------------------------------------#
    #   model_path指向logs文件夹下的权值文件
    #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
    #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
    #-------------------------------------------------------------------#
    "model_path"    : 'model_data/unet_vgg_medical.pth',
    #--------------------------------#
    #   所需要区分的类的个数+1
    #--------------------------------#
    "num_classes"   : 2,
    #--------------------------------#
    #   所使用的的主干网络：vgg、resnet50   
    #--------------------------------#
    "backbone"      : "vgg",
    #--------------------------------#
    #   输入图片的大小
    #--------------------------------#
    "input_shape"   : [512, 512],
    #--------------------------------#
    #   blend参数用于控制是否
    #   让识别结果和原图混合
    #--------------------------------#
    "blend"         : True,
    #--------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #--------------------------------#
    "cuda"          : True,
}
```
2. 运行即可预测。  
```python
img/cell.png
```
#### 二、使用自己训练的权重
1. 按照训练步骤训练。    
2. 在unet.py文件里面，在如下部分修改model_path、backbone和num_classes使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件**。    
```python
_defaults = {
    #-------------------------------------------------------------------#
    #   model_path指向logs文件夹下的权值文件
    #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
    #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
    #-------------------------------------------------------------------#
    "model_path"    : 'model_data/unet_vgg_voc.pth',
    #--------------------------------#
    #   所需要区分的类的个数+1
    #--------------------------------#
    "num_classes"   : 21,
    #--------------------------------#
    #   所使用的的主干网络：vgg、resnet50   
    #--------------------------------#
    "backbone"      : "vgg",
    #--------------------------------#
    #   输入图片的大小
    #--------------------------------#
    "input_shape"   : [512, 512],
    #--------------------------------#
    #   blend参数用于控制是否
    #   让识别结果和原图混合
    #--------------------------------#
    "blend"         : True,
    #--------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #--------------------------------#
    "cuda"          : True,
}
```
3. 运行predict.py，输入    
```python
img/street.jpg
```   
4. 在predict.py里面进行设置可以进行fps测试和video视频检测。    

### 评估步骤
1、设置get_miou.py里面的num_classes为预测的类的数量加1。  
2、设置get_miou.py里面的name_classes为需要去区分的类别。  
3、运行get_miou.py即可获得miou大小。  

## Reference
-[fastseg](https://github.com/ekzhang/fastseg) 
-[deeplabv3-plus-pytorch](https://github.com/bubbliiiing/deeplabv3-plus-pytorch)

