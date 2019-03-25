# MFRN
MFRN model realization 
Requirements:

tensorflow_gpu 1.4 
Keras 2.1.5
CUDA 8.1
cudnn



my_classes_fortest.py & my_classes_fortrain.py are two data generators for the model to obtain a batch of sample from the disk according to the index.
my_classes_fortest.py & my_classes_fortrain.py 是两个数据生成器，用于为模型从硬盘读取一个batch的训练样本。

the one_hot_it() function can be modified for multi classification tasks. eg.sx = np.zeros([320,320,2]) 320X320 is the size of the sample image and 2 is according to the class number.
the one_hot_it() 可以被修改为多分类任务。

def __data_generation(self, list_IDs_temp) function can be modified for the address to load images.
def __data_generation(self, list_IDs_temp) 内部可以被修改来调整数据录入的地址。

aprilmodel0514B_0517.py and lib_0514.py are the realization of MFRN and the dense block respectively.
aprilmodel0514B_0517.py 和 lib_0514.py 分别是模型MFRN 和 dense block 的实现

def __init__(self, img_rows = 320, img_cols = 320, growth_rate = 12) can be modified to set the size of the image sample and the growth rate of dense block.
def __init__(self, img_rows = 320, img_cols = 320, growth_rate = 12) 可以设置样本大小和denseblock的增长率。

variables ch and reduction of def get_model() function are the compression factors for the skip connections and compression transition in MFRN.
变量ch和reduction 是跳跃链接和压缩上采样的压缩参与。

use def train(self) to train the model with options to save weights and check train history.
The two data generators are called in train() to provide train and test data for the model.

使用 def train(self) 来训练模型。其中两个data generator 被调用来给模型提供数据。

ljdefs0312.py contains several functions for batch processing including def createIDs(): for creating IDlist which is the index of the data generators
ljdefs0312.py 包含一些用于批处理的函数, 其中的def createIDs(): 用于生成data generator 所需要的索引。

The workflow0503_PM.py is used to load the trained model and predict a complete remote sensing imagery (1500*1500 pixels in our experiment).
the padding and overlap are used to set the size of slide cutting.
def createlblimg(patches,smallbox) is responsible for loading the trained model and predict the patches cropped from the imagery.
def recover_imagery(imglist,boxlist_inner,padding_w,padding_h,pathout,finalbox) is responsible for recovering the complete imagery from the segmented patches.
The workflow0503_PM.py 用于调用训练好的模型预测完整的遥感影像。参数 padding 和 overlap 用于调整滑动窗口的步长。
def createlblimg(patches,smallbox) 用于调用模型和预测切割好的图块。
def recover_imagery(imglist,boxlist_inner,padding_w,padding_h,pathout,finalbox)  用于将预测好图块还原为原来的大小。


If this code is useful for your paper, please cite:
Li, L.; Liang, J.; Weng, M.; Zhu, H. A multiple-feature reuse network to extract buildings from remote sensing imagery. Remote Sensing 2018, 10, 1350.
