"""
绘制激活函数的图像
"""
import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(-10, 10, 100)


def relu(x):
    result = 0
    if x > 0:
        result = x
    else:
        result = 0
    return result


def sigmod(x):
    result = 1.0/(1+np.exp(-x))
    return result

y1 = np.array([relu(t) for t in x])
y2 = np.array([sigmod(t) for t in x])
y3 = np.tanh(x)

plt.figure(1)
plt.title("ReLU fucntion")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, y1, "-")
plt.savefig("doc/result_figure/relu_function.eps", transparent=True, dpi=300, pad_inches=0)
plt.show()


plt.figure(2)
plt.title("sigmoid fucntion")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, y2, "-")
plt.savefig("doc/result_figure/sigmoind_function.eps", transparent=True, dpi=300, pad_inches=0)
plt.show()

plt.figure(3)
plt.title("tanh fucntion")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, y3, "-")
plt.savefig("doc/result_figure/tanh_function.eps", transparent=True, dpi=300, pad_inches=0)
plt.show()
