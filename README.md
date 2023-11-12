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
#### train.csv（两列，需要自己划分训练集和验证集）
![image](https://github.com/Zou1c/classify-leaves/assets/58977192/3960fd95-f9b7-49d8-9dc5-f3b92cec27b5)
#### test.csv（一列，指出最终需要预测的图片有哪些）
![image](https://github.com/Zou1c/classify-leaves/assets/58977192/4a201cb6-c3d2-4830-8f46-1bc1f7b8e346)
#### sample_submission.csv（两列，提交样例文件。其实就是对test.csv加上了标签，但里面的树叶种类是随机写的）
![image](https://github.com/Zou1c/classify-leaves/assets/58977192/126d8ae8-ef10-48ad-875a-a5bc9fd87141)

## 尝试过程
### 第一次尝试
