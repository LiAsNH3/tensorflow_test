"""小试牛刀
利用tensorflow对二维区域上的数据进行二分类
"""
import tensorflow as tf
from numpy.random import RandomState
import matplotlib.pyplot as plt
import numpy as np


# 定义参数和输入节点
w = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
b = tf.Variable(tf.random_normal([1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义前向传播过程及其损失函数
y = tf.nn.sigmoid(tf.matmul(x, w) + b)
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                                + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


# 生成模拟训练数据
rdm = RandomState(1)
X = rdm.rand(256, 2)  # 模拟数据集
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]  # 类标


# 生成模拟测试数据
rdm = RandomState(2)
X_test = rdm.rand(100, 2)
Y_test = [[int(x1+x2 < 1)] for (x1, x2) in X_test]  # 类标
test_feed = {x: X_test, y_: Y_test}


# 定义正确率计算
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 创建一个tenor flow会话，训练结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 训练模型
    STEPS = 500
    for i in range(STEPS):
        # 从测试集中选取适当数量的样本进行训练
        for input_x, input_y in zip(X, Y):
            input_x = np.reshape(input_x, (1, 2))
            input_y = np.reshape(input_y, (1, 1))
            sess.run(train_step, feed_dict={x: input_x, y_: input_y})
        if i % 100 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

    # 输出训练后的参数取值。
    print("\n")
    print(sess.run(w))
    print("\n"+"---"*50)

    # 绘制测试数据
    plt.figure()
    t = np.linspace(0, 1, 100)
    lin = -t + 1
    plt.plot(t, lin, ls="-", c="r", label="classification line")

    # 下面开始进行测试
    pred_Y = sess.run(y, feed_dict={x: X_test})
    pred_A = []
    pred_B = []
    index = np.zeros((100, 1), np.int)
    for i in range(100):
        if pred_Y[i] > 0.5:
            pred_A.append(i)
        else:
            pred_B.append(i)
    plt.scatter(X_test[pred_A, 0], X_test[pred_A, 1], c="b", label="predict A")
    plt.scatter(X_test[pred_B, 0], X_test[pred_B, 1], c="g", label="predict B")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("test data classification results")
    plt.savefig("doc/result_figure/sample_class.eps", dpi=300)
    plt.show()
