'''小试牛刀
应用机器学习的方法来简单拟合数据
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 生成模拟训练数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

plt.figure()  # 绘制生产的训练点集
plt.scatter(x_data, y_data, s=12.0)
plt.title("train data($y=x^2-0.5$)")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.savefig("doc/result_figure/fiting_trian.eps", dpi=300, pad_inches=0)
plt.show()


# 定义网络输入信息
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


def add_layer(inputs, in_size, out_size, activation_function=None):
    """定义神经网络中的网络层
    将每层的信息定义好，后续直接调用

    Args:
      inputs：输入层的数据或者是前一层网络的输出
      in_size：与权重相关的参数，与前一层的输出确定
      out_size：与权重相关的参数，与当前层的输出相关
      activation_function：激活函数，默认没有


    Returns:
      返回一个tensorflow张量
    """
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 隐藏层的权重
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # 隐藏层的偏移量
    Wx_plus_b = tf.matmul(inputs, Weights) + biases  # 向前传播计算
    if activation_function is None:  # 设置激活函数，默认没有
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 添加隐藏层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 添加输出层
prediction = add_layer(l1, 10, 1, activation_function=None)

# 计算真值与预层值之间的误差，设置损失函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                      reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 启动tensorflow会话
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data, label="trian data", s=12.0)  # 画出真实数据的点状图
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
plt.ion()
# 如果不加这个，每画完一条线程序会暂停，加了后会一直画，如果一直画不是会很
# 密密麻麻看不清吗，后面 有语句会移除当前线条防止出现该情况
plt.show()


for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 10 == 0:
        # 改进绘画的效果
        try:  # 尝试以下语句
            ax.lines.remove(lines[0])  # 抹除画出来的第一条线
        except Exception:
            # 如果没有的话，忽略第一次的错误，因为这时还没画第一条线
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})  # 预测值
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, '-.', lw=3, c='orange', label="prediction line")
        ax.set_title("Prediction function and test data iterated {:d} times".format(i))
        # 画出预测线，r-表红色，lw表粗度为5
        plt.pause(1)  # 每画完一条线暂停一秒

y_data_real = np.square(x_data)-0.5
ax.plot(x_data, y_data_real, "r-", label="real function")
ax.legend()
ax.set_ylim(-0.7, 0.8)
plt.savefig("doc/result_figure/fiting_result.eps", dpi=300, pad_inches=0)