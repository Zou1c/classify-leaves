# classify-leaves
李沐《动手学深度学习》的树叶分类实验尝试的一个小结  
竞赛地址：https://www.kaggle.com/c/classify-leaves

## 下载的数据集
### 图片
#### images(文件夹)
- 0.jpg
- 1.jpg
- ...
- 27152.jpg  
每张图片都为RGB彩图(3通道)，224x224像素大小。可以看几张长啥样：  
![image](https://github.com/Zou1c/classify-leaves/assets/58977192/2eed46eb-6160-495e-8c86-c384682078ac)
![image](https://github.com/Zou1c/classify-leaves/assets/58977192/6fa64a42-345d-4e4f-9aba-580606c80a53)
![image](https://github.com/Zou1c/classify-leaves/assets/58977192/da71e54b-8988-4095-8010-2da6bc78fd73)
![image](https://github.com/Zou1c/classify-leaves/assets/58977192/128c7ba8-ed6a-427e-aae0-d17b2191a056)  
可以看到背景还是比较单一的，因此训练起来还是容易收敛的。
*** 
### csv文件
#### train.csv（算上头部总共18354行，两列，需要自己划分训练集和验证集）
![image](https://github.com/Zou1c/classify-leaves/assets/58977192/3960fd95-f9b7-49d8-9dc5-f3b92cec27b5)
#### test.csv（算上头部8001行，一列，指出最终需要预测的图片有哪些）
![image](https://github.com/Zou1c/classify-leaves/assets/58977192/4a201cb6-c3d2-4830-8f46-1bc1f7b8e346)
#### sample_submission.csv（算上头部8001行，两列，提交样例文件。其实就是对test.csv加上了标签，但里面的树叶种类是随机写的）
![image](https://github.com/Zou1c/classify-leaves/assets/58977192/126d8ae8-ef10-48ad-875a-a5bc9fd87141)

## 尝试过程
### 第一次尝试(ResNet，private score 0.79，train:valid = 1:1)
主要是熟悉平台上的使用和操作吧。  
1.把文件上传到kaggle的个人数据集里，搞清楚kaggle的notebook的文件路径。  
2.继承Dataset类，创建训练验证数据类和测试数据类。  
3.标签类型还是字符串类型呢，需要自己转化成数值型，才能进行训练。    

总之这一次就以跑成功为主，看完ResNet就直接用那一节的代码和参数，使用的d2l.train_ch6()函数来训练数据。  
没有k则交叉验证（因为想快点跑完看看结果），直接取train.csv前9000个作为训练数据，剩余的作为验证数据（差不多是1：1）  
训练图像大概长这样：  
![image](https://github.com/Zou1c/classify-leaves/assets/58977192/14b94053-1801-41d6-92d2-9c72cd74b1b3)  
notebook地址：https://www.kaggle.com/code/zixthw/classify-leaves-18000train-176category  

总之第一次尝试还是有许多问题的，但我当时想的就是这个准确率太低了，而且调参后也有比较大的震荡（train_loss，test_acc）。所以我决定重新划分训练数据。  

### 第二次尝试(ResNet，private score 0.85, train:valid = 13:5)
第一次用的GPU T4 x2，但实际上我只用了一块GPU。所以这次换成了计算精度和速度都更高的P100。  
将train.csv的前13000条作为训练数据，其余的作为验证数据。  
仍然使用d2l.train_ch6方法进行训练。  
训练图像大概长这样：  
![image](https://github.com/Zou1c/classify-leaves/assets/58977192/c0b020d7-d0f7-4eef-acee-82ab7c7d922f)  
notebook地址：https://www.kaggle.com/code/zixthw/classify-leaves-13000-5000  

这一次精度有了明显提升，只是将训练数据多“喂”给模型就可以，5000多个图片作为验证集也够用了。因为大致看了一下每个种类数所包含的图片数量，最少的也有个一半左右该类别的数量了，验证集的大部分也都在50张图片以上。    

但是我感觉精度一直上不去是过拟合的原因，所以我的下一次思路是在模型末尾添加dropout层，也许能提升精度——大概吧，我当时想。  

### 第三次尝试(ResNet + 个人的dropout，private score 0.81, train:valid = 13:5)
跟第二次比相当于就改了模型结构  
但结果比较惨，我在如下代码的Linear部分改来改去都没什么太好的效果，还不如不加这个dropout：  
![image](https://github.com/Zou1c/classify-leaves/assets/58977192/d7e0e65c-9f88-475b-99eb-1170ea021142)

notebook地址：https://www.kaggle.com/code/zixthw/classifyleaves-deal-overfitting-13-5  
看来确实是这样的，不是专家不要动已经提出的模型/(ㄒoㄒ)/~~


### 第四次尝试(ResNet，private score 0.89, train:valid = 2:16)
前面的尝试中我也不停地调超参数，感觉此时调超参数好像对精度大幅提升还是比较无力。  
于是我又回归到数据集的划分上，我再多给模型一些训练数据会怎样，验证集的数量还够用吗？  
于是做了两次划分：    
一次将前2000个作为验证集；令一次将后2000个作为验证集。  
效果还不错，两个模型中有一个能达到私榜接近90%的正确率了：  
![image](https://github.com/Zou1c/classify-leaves/assets/58977192/1e996237-ac06-4e49-b06d-c1eee93a5fe7)  
notebook地址：https://www.kaggle.com/code/zixthw/classify-leaves-1-8  
这应该算是我个人的最后一次尝试了。我此时仍然用的d2l里的api进行训练，并且一想到基本的ResNet就可以达到96%的正确率，我感觉会不会是我的ResNet层数太少了，还是训练过程还是自己手写来的好？  
我有点迷茫了，毕竟在数据集划分上，（我感觉啊，后面没有尝试了）应该不会再有比较大的提升了。
我可能要考虑换整个模型了， 并且是时候考虑k则交叉验证了（虽然花的时间就是k倍了）。正好kaggle一周30h免费GPU也被用完了，就先缓一缓。

### 第五次尝试(ResNest，k_fold=5，private score 0.98)




