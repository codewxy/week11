## 一、 word embedding部分

### 1.代码文件如下：

word2vec_week11.py

### 2.输出的图片结果:

![tsne_w11](tsne_w11.png)

图中能看到很多距离较近的字。

### 3.代码中处理matplotlib 中文显示异常

使用了如下的方法：

```python
from pylab import mpl
# 修改ubuntu字体用来兼容中文显示
zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
```

下面调用pyplot的时候，加上参数 fontproperties=zhfont；就能解决中文显示方框的问题。

### 4.理解

embedding 就是将自然语言间的联系用数学向量表示在空间坐标中。方便了计算机的识别和操作。word2vec是一种embedding算法，使用稠密的表达方式，对one_hot降维，得到低维实数向量，将特征向量映射到多维空间，不仅仅是坐标轴上，避免产生维度爆炸。



## 二、RNN部分

### 1.运行情况：

模型地址：https://www.tinymind.com/code-wxy/week11

训练模型运行结果：

验证模型结果：

log文件：

### 2.理解

循环神经网络，我的理解就是相较于全连接及卷积神经网络的区别是增加了时间的维度。在一段时间内的数据作为数据集里，每一步的权重计算时都要将之前一步的权重参与计算。在反向传播中，也同样将后一个权重的因子参与到前一个权重的计算中来更新前一个权重，将所有权重串起来，从而达到权重W在时间维度上的体现。

### 3.心得体会

本次作业有点拖沓，想比较前面的作业，感觉这个作业结构稍微复杂一点，开始有点蒙，慢慢理了一遍。至于写诗机器人，代码中我使用了word embeding作业中生成的embeding_file.npy ，最后的写诗的效果不是很好，按照作业最基本的完成，没进行优化。

