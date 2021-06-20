# Kaggle-Cassava-Leaf-Disease-Classification
Kaggle木薯叶疾病分类 Bronze

比赛连接 ：[Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification)

[模型权重下载(百度云)](https://pan.baidu.com/s/1EUsgH-P0kR_bxf_5cHKYlQ ) 提取码：funv 

## 代码说明
## src ：比赛代码
## tools ：数据分析，数据扩充和转换Kaggle提交格式等一些工具

## 比赛背景

根据提供的数据，对木薯叶图像进行 <b>Cassava Bacterial Blight (CBB)</b>，<b>Cassava Brown Streak Disease (CBSD)</b>，
<b>Cassava Green Mottle (CGM)</b>，<b>Cassava Mosaic Disease (CMD)</b>和<b>Healthy</b>共5个类别的图像分类。

<img src="https://github.com/ielym/Kaggle-Cassava-Leaf-Disease-Classification/blob/main/tools/datas/1.jpg" height="200" />

## 比赛流程和任务
比赛分为ab榜，参加比赛的时间比较晚，打了两周就面临寒假离校，并且在家期间由于没有计算资源也就耽搁下来了，最终只获得了铜牌 (309 / 3900)。

但这场比赛是我第一次参加Kaggle竞赛，对Kaggle的比赛流程有了大概的了解，所以还是有必要记录一下的。

## 数据分析
* 按照以往的比赛经验，首先还是对训练集图像的分辨率大小，高宽比进行了统计分析，以确定输入图像分辨率，网络深度以及是否多尺度训练和是否Padding。
* 由于缺少领域知识，对木薯叶疾病了解较少，结合网络资料也无法人工对图像标注是否正确进行评估。但是根据算法表现，发现存在大量的噪声数据。
具体是通过更换损失函数来发现的。
* 图像类别分布不平衡，对于数据量较少的类别，使用伪标签法对2019年的比赛数据进行了扩充（2019年数据包括训练集（有标注）和额外数据（无标注））
但是担心训练集标注数据中也存在大量噪声，因此都使用了伪标签法进行重新标注。
* 由于图像是通过搜集不同农民手机拍摄的照片，因此不同图像的质量，拍摄角度，图像背景，以及疾病区域大小的差异是比较显著的。

## 模型设计
* 使用了<b>ResNext101_32x8d</b>模型，5折交叉验证训练并提交多模型融合结果。
* 使用<b>ResNext101_32x8d</b>和<b>Efficient-b7</b>进行额外数据的半监督标注。
* 使用SnapMix数据增强策略，实验发现效果比MixUP更好。此外还做了翻转，旋转，转置，亮度，cutout等数据增强方式。

## 模型训练与优化
* 分别尝试了<b>CE Loss</b>，<b>LabelSmooth CE Loss</b>。为了解决类别分布不平衡的问题，尝试使用了<b>Focal Loss</b>，但是发现效果还不如只使用交叉熵损失。
也正基于这个现象，我们才发现训练数据中存在大量噪声的问题。因此最后使用了<b>Weighted CE Loss</b>和<b>LabelSmooth</b>。

