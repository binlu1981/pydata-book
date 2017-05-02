# -*- coding: utf-8 -*-
import pandas as pd
#reading a csv into pandas
#读取英国降雨量
df = pd.read_csv('D:/megasync/research/ngs/GitHub/pydata_book/ch02/uk_rain_2014.csv',header=0)
print(df.head(5))


#http://codingpy.com/article/a-quick-intro-to-matplotlib/
#十分钟入门Matplotlib
import matplotlib.pyplot as plt
import numpy as np
# 简单的绘图 画出一个简单的正弦曲线
x = np.linspace(0, 2 * np.pi, 50)
#这段代码将会生成一个包含 50 个元素的数组，这 50 个元素均匀的分布在 [0, 2pi] 的区间上
print(x)
plt.plot(x, np.sin(x)) # 如果没有第一个参数 x，图形的 x 坐标默认为数组的索引
#plt.show() # 显示图形
#大多数时候读者可能更想在一张图上绘制多个数据集。用 Matplotlib 也可以轻松实现这一点
plt.plot(x, np.sin(x),x, np.sin(2 * x))
#plt.show()
plt.plot(x, np.sin(x), 'r-o',x, np.cos(x), 'g--')
#plt.show()
"""
上述代码展示了两种不同的曲线样式：'r-o' 和 'g--'。字母 'r' 和 'g' 代表线条的颜色，后面的符号代表线和点标记的类型。例如 '-o' 代表包含实心点标记的实线，'--' 代表虚线。其他的参数需要读者自己去尝试，这也是学习 Matplotlib 最好的方式。
颜色： 蓝色 - 'b' 绿色 - 'g' 红色 - 'r' 青色 - 'c' 品红 - 'm' 黄色 - 'y' 黑色 - 'k'（'b'代表蓝色，所以这里用黑色的最后一个字母） 白色 - 'w' 线： 直线 - '-' 虚线 - '--' 点线 - ':' 点划线 - '-.' 常用点标记 点 - '.' 像素 - ',' 圆 - 'o' 方形 - 's' 三角形 - '^'
"""

#使用子图
x = np.linspace(0, 2 * np.pi, 50)
plt.subplot(2, 1, 1) # （行，列，活跃区）
plt.plot(x, np.sin(x), 'r')
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x), 'g')
#plt.show()
"""
使用子图只需要一个额外的步骤，就可以像前面的例子一样绘制数据集。即在调用 plot() 函数之前需要先调用 subplot() 函数。该函数的第一个参数代表子图的总行数，第二个参数代表子图的总列数，第三个参数代表活跃区域。
活跃区域代表当前子图所在绘图区域，绘图区域是按从左至右，从上至下的顺序编号。例如在 4×4 的方格上，活跃区域 6 在方格上的坐标为 (2, 2)
"""


# 简单的散点图
x = np.linspace(0, 2 * np.pi, 50)
y = np.sin(x)
plt.scatter(x,y)
#plt.show()
"""
调用 scatter() 函数并传入两个分别代表 x 坐标和 y 坐标的数组。注意，我们通过 plot 命令并将线的样式设置为 'bo' 也可以实现同样的效果
"""

# 彩色映射散点图
x = np.random.rand(1000)
y = np.random.rand(1000)
size = np.random.rand(1000) * 50
colour = np.random.rand(1000)
plt.scatter(x, y, size, colour)
plt.colorbar()
plt.show()
"""
同前面一样我们用到了 scatter() 函数，但是这次我们传入了另外的两个参数，分别为所绘点的大小和颜色。通过这种方式使得图上点的大小和颜色根据数据的大小产生变化
"""

#直方图
x = np.random.randn(1000)
plt.hist(x, 50)
plt.show()
"""
直方图是 Matplotlib 中最简单的图形之一。你只需要给 hist() 函数传入一个包含数据的数组。第二个参数代表数据容器的个数。数据容器代表不同的值的间隔，并用来包含我们的数据。数据容器越多，图形上的数据条就越多
"""

#标题，标签和图例
# 添加标题，坐标轴标记和图例
x = np.linspace(0, 2 * np.pi, 50)
plt.plot(x, np.sin(x), 'r-x', label='Sin(x)')
plt.plot(x, np.cos(x), 'g-^', label='Cos(x)')
plt.legend() # 展示图例
plt.xlabel('Rads') # 给 x 轴添加标签
plt.ylabel('Amplitude') # 给 y 轴添加标签
plt.title('Sin and Cos Waves') # 添加图形标题
plt.show()



#使用 Python 进行科学计算：NumPy入门
"""
NumPy 的核心是数组（arrays）。具体来说是多维数组（ndarrays），但是我们不用管这些。通过这些数组，我们能以闪电般的速度使用像向量和数学矩阵之类的功能。赶紧捡起你的线性代数吧
"""
# 1D Array
a = np.array([0, 1, 2, 3, 4])
b = np.array((0, 1, 2, 3, 4))
c = np.arange(5)
d = np.linspace(0, 2*np.pi, 5)
"""
上边的代码展示了创建数组的四种不同方式。最基本的方式是传递一个序列给 NumPy 的 array() 函数；你可以传给它任意的序列，不仅仅是我们常见的列表之类的。

注意，当输出的数组中的数值长度不一样的时候，它会自动对齐。这在查看矩阵的时候很有用。数组的索引和 Python 中的列表或其他序列很像。你也可以对它们使用切片，这里我不再演示一维数组的切片，如果你想知道更多关于切片的信息，查看这篇文章。

上边数组的例子给你展示了如何在 NumPy 中表示向量，接下来我将带你们领略一下怎么表示矩阵和多维数组。
"""

# MD Array,
a = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28 ,29, 30],
              [31, 32, 33, 34, 35]])

print(a[2,4]) # >>>25
"""
通过给 array() 函数传递一个列表的列表（或者是一个序列的序列），可以创建二维数组。如果我们想要一个三维数组，那我们就传递一个列表的列表的列表，四维数组就是列表的列表的列表的列表，以此类推。
注意二维数组是如何成行成列排布的（在我们的朋友--空格的帮助下）。如果要索引一个二维数组，只需要引用相应的行数和列数即可

背后的数学知识

为了更好的理解这些，我们需要来看一下什么是向量和矩阵。

向量是一个有方向和大小的量，通常用来表示速度、加速度和动量等。向量能以多种方式书写，但是我们最有用的方式是把它们写在有 n 个元素的元组里边，比如（1， 4， 6， 9）。这就是它们在 NumPy 中的表示方式。

矩阵和向量很像，除了它是由行和列组成的；更像一个网格（grid）。矩阵中的数值可以用它们所在的行和列来表示。在 NumPy 中，可以像我们前面所做的那样，通过传递序列的序列来创建数组。
"""

#多维数组切片
# MD slicing
print(a[0, 1:4]) # >>>[12 13 14]
print(a[1:4, 0]) # >>>[16 21 26]
print(a[::2,::2]) # >>>[[11 13 15]
                  #     [21 23 25]
                  #     [31 33 35]]
print(a[:, 1]) # >>>[12 17 22 27 32]

#数组属性
# Array properties
a = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28 ,29, 30],
              [31, 32, 33, 34, 35]])

print(type(a)) # >>><class 'numpy.ndarray'>
print(a.dtype) # >>>int64
print(a.size) # >>>25
print(a.shape) # >>>(5, 5)
print(a.itemsize) # >>>8
print(a.ndim) # >>>2
print(a.nbytes) # >>>200
"""
如你所看，在上边的代码中 NumPy 的数组其实被称为 ndarray。我不知道为什么它被称为 ndarray，如果有人知道请在下边留言！我猜测它是表示 n 维数组（n dimensional array）。

数组的形状（shape）是指它有多少行和列，上边的数组有五行五列，所以他的形状是（5，5）。

'itemsize' 属性是每一个条目所占的字节。这个数组的数据类型是 int64，一个 int64 的大小是 64 比特，8 比特为 1 字节，64 除以 8 就得到了它的字节数，8 字节。

'ndim' 属性是指数组有多少维。这个数组有二维。但是，比如说向量，只有一维。

'nbytes' 属性表示这个数组中所有元素占用的字节数。你应该注意，这个数值并没有把额外的空间计算进去，因此实际上这个数组占用的空间会比这个值大点。
"""

#Basic Operators
a = np.arange(25)
a = a.reshape((5,5))
b = np.array([10, 62, 1, 14, 2, 56, 79, 2, 1, 45,
              4, 92, 5, 55, 63, 43, 35, 6, 53, 24,
              56, 3, 56, 44, 78])
b = b.reshape((5,5))

print(a+b)
print(a-b)
print(a*b)
print(a/b)
print(a**2)
print(a<b)
print(a>b)
print(a.dot(b))

"""
除了 dot() 之外，这些操作符都是对数组进行逐元素运算。比如 (a, b, c) + (d, e, f) 的结果就是 (a+d, b+e, c+f)。它将分别对每一个元素进行配对，然后对它们进行运算。它返回的结果是一个数组。注意，当使用逻辑运算符比如 “<” 和 “>” 的时候，返回的将是一个布尔型数组，这点有一个很好的用处，后边我们会提到。

dot() 函数计算两个数组的点积。它返回的是一个标量（只有大小没有方向的一个值）而不是数组。
"""

# dot, sum, min, max, cumsum
a = np.arange(10)

print(a.sum()) # >>>45
print(a.min()) # >>>0
print(a.max()) # >>>9
print(a.cumsum()) # >>>[ 0  1  3  6 10 15 21 28 36 45]
#cumsum() 函数就不是那么明显了。它像 sum() 那样把所有元素加起来，但是它的实现方式是，第一个元素加到第二个元素上，把结果保存到一个列表里，然后把结果加到第三个元素上，再保存到列表里，依次累加。

# Fancy indexing
a = np.arange(0, 100, 10)
indices = [1, 5, -1]
b = a[indices]
print(a) # >>>[ 0 10 20 30 40 50 60 70 80 90]
print(b) # >>>[10 50 90]


# Boolean masking
import matplotlib.pyplot as plt

a = np.linspace(0, 2 * np.pi, 50)
b = np.sin(a)
plt.plot(a,b)
mask = b >= 0
plt.plot(a[mask], b[mask], 'bo')
mask = (b >= 0) & (a <= np.pi / 2)
plt.plot(a[mask], b[mask], 'go')
plt.show()
"""
上边的代码展示了实现布尔屏蔽。你需要做的就是传递给数组一个与它有关的条件式，然后它就会返回给定条件下为真的值。
我们用条件式选择了图中不同的点。蓝色的点（也包含图中的绿点，只是绿点覆盖了蓝点），显示的是值大于零的点。绿点显示的是值大于 0 小于 Pi / 2 的点。
"""

# Incomplete Indexing
a = np.arange(0, 100, 10)
b = a[:5]
c = a[a >= 50]
print(b) # >>>[ 0 10 20 30 40]
print(c) # >>>[50 60 70 80 90]

"""
Where 函数

where() 函数是另外一个根据条件返回数组中的值的有效方法。只需要把条件传递给它，它就会返回一个使得条件为真的元素的列表。
"""
# Where
a = np.arange(0, 100, 10)
b = np.where(a < 50) 
c = np.where(a >= 50)[0]
print(b) # >>>(array([0, 1, 2, 3, 4]),)
print(c) # >>>[5 6 7 8 9]

