## 运行本程序的注意事项
1. 建议的Python版本为3.5，对应的工具包版本见文件TensorFlow-env.txt
2. 注意在运行猫狗分类模型程序的先后顺序
3. 文档用latex写成，编译需要相应的环境，建议用xelatex编译


## 文件结构
cat_and_dog
  --__pycache__ 
  --data  数据存放，由于数据较大，未将将其打包，需要数据到此处下载：https://pan.baidu.com/s/1EUGv_9TtsVkxPlqEESvyaw 提取码：62l9 
    --test  网络爬虫获得数据标存储
    --test_min  简单测试
    --train 训练数据存储

  --doc  文档撰写
    --result_figure  文档中所用图片存储

  --logs  神经网络训练日志

  --activate_function_plot.py  绘制激活函数源程序

  --get_test_figure.py   网络爬虫获得数据源程序

  --input_data.py   卷积神经网络数据输入准备

  --model.py   卷积神经网络模型

  --my_test.py  测试源程序

  --plot_all.py  绘制数据集中的部分图片

  --sample_tensorflow.py  二维区域上的简单二分类

  --small_try.py  二维区域上数据拟合

  --TesnsorFlow-env.txt  运行程序所包含的Python工具包

  --training.py  训练模型

  --程序运行.mp4  运行程序的录制

