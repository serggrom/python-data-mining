import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
from numpy.linalg import inv, qr
from random import normalvariate
import random

data = [6, 7.5, 8, 0, 1]
arr = np.array(data)

#print(arr)

data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)

#print(arr2)

#print(arr2.ndim)

#print(arr2.shape)


#print(arr.dtype, ' and ', arr2.dtype)


#print(np.zeros((3, 6)))

#print(np.empty((2, 3, 2)))


#print(np.arange(15))

arr3 = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])

#print(arr3)

#print(arr3.astype(np.int32))


numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)

#print(numeric_strings.astype(float))

int_array = np.arange(10)

calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)

#print(int_array.astype(calibers.dtype))

arr4 = np.array([[1., 2., 3.], [4., 5., 6.]])
#print(arr4)


#print(arr4-arr4)

#print(1 / arr4)

#print(arr4 ** 0.5)

arr6 = np.arange(10)

arr6[5:8] = 12

#print(arr6)

arr6_slice = arr6[5:8]

arr6_slice[1] = 12345

#print(arr6)

arr6_slice[:] = 64

#print(arr6)


arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

#print(arr2d[2])

#print(arr2d[0, 2])

arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

#print(arr3d)


old_values = arr3d[0].copy()

#print(arr3d[0])

arr3d[0] = old_values

#print(arr3d)

#print(arr3d[1, 0])

#print(arr6[1:6])

#print(arr2d[:2])

#print(arr2d[2, :1])

#print(arr2d[:, :1])

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])

data = randn(7, 4)

#print(names)

#print(data)

#print(names == 'Bob')

#print(data[names == 'Bob'])

#print(data[names == 'Bob', 2:])


#print(data[~(names == 'Bob')])


mask = (names == 'Bob') | (names == 'Will')

#print(mask)

#print(data[mask])

#data[names != 'Joe'] = 7

#print(data)


arr7 = np.empty((8, 4))

for i in range(8):
    arr7[i] = i
#print(arr7)

#print(arr7[[4, 3, 0, 6]])

#print(arr7[[-3, -5, -7]])

arr7 = np.arange(32).reshape((8, 4))

#print(arr7[[1, 5, 7, 2], [0, 3, 1, 2]])

#print(arr7[[1, 5, 7, 2]][:, [0, 3, 1, 2]])

#print(arr7[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])])

arr8 = np.arange(15).reshape((3, 5))

#print(arr8)

#print(arr8.T)

arr8 = np.random.randn(6, 3)

#print(np.dot(arr8.T, arr8))

arr8 = np.arange(16).reshape((2, 2, 4))

#print(arr8)

#print(arr8.transpose((1, 0, 2)))

#print(arr8.swapaxes(1, 2))

arr9 = np.arange(10)

#print(np.sqrt(arr9))

#print(np.exp(arr9))

x = randn(8)
y = randn(8)

#print(x)
#print(y)

#print(np.maximum(x, y))

arr9 = randn(7) * 5

#print(arr9)

#print(np.modf(arr9))

points = np.arange(-5, 5, 0.01)

xs, ys = np.meshgrid(points, points)

#print(ys)


z = np.sqrt(xs ** 2 + ys **2)

#print(z)

#plt.imshow(z, cmap=plt.cm.gray)
#plt.colorbar()
#plt.title("Image plot of $\sqrt(x^2 + y^2)$ for a grid values")
#plt.show()


xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])

cond = np.array([True, False, True, True, False])

#result = [(x if c else y)
#         for x, y, c in zip(xarr, yarr, cond)]

#print(result)

result = np.where(cond, xarr, yarr)

#print(result)

arr10 = randn(4, 4)

#print(arr10)

arr10 = np.where(arr10 > 0, 2, -2)
#print(arr10)

#arr10 = np.where(arr10 > 0, 2, arr10)

'''
result = []
for i in range(n):
    if cond1[i] and cond2[i]:
        result.append(0)
    elif cond1[i]:
        result.append(1)
    elif cond2[i]:
        result.append(2)
    else:
        result.append(3)
'''


arr11 = np.random.randn(5, 4)

#print(arr10.mean())
#print(np.mean(arr10))


#print(arr11.sum())
#print(arr11)

#print(arr11.mean(axis=1))
#print(arr11.sum(0))


arrays = np.array([[0, 1, 2], [3, 4, 5], [7, 8, 9]])

#print(arrays.cumsum())

#print(arrays.cumprod(1))

arr12 = randn(100)

pos = (arr12 > 0).sum()

#print(pos)

bools = np.array([False, False, True, False])

#print(bools.any())
#print(bools.all())


arr13 = randn(8)

#print(arr13)
arr13.sort()

arr13 = randn(5, 3)

#print(arr13)

arr13.sort(1)

#print(arr13)

large_arr = randn(1000)
large_arr.sort()

quantil = large_arr[int(0.05 * len(large_arr))]

#print(quantil)


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Joe', 'Joe'])

#print(np.unique(names))

ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])

#print(np.unique(ints))

#print(sorted(set(names)))


values = np.array([6, 0, 0, 3, 2, 5, 6])

srt = np.in1d(values, [2, 3, 6])

#print(srt)


arr14 = np.arange(10)

np.save('some_array', arr14)

#print(np.load('some_array.npy'))

np.savez('array_archive.npz', a=arr14, b=arr14)

arch = np.load('array_archive.npz')

#print(arch['b'])

x = np.array([[1., 2., 3.], [4., 5., 6.]])

y = np.array([[6., 23.], [-1., 7.], [8, 9]])

z = x.dot(y)

#print(z)

one = np.ones(3)

z = np.dot(x, one)

#print(z)

X = randn(5, 5)

mat = X.T.dot(X)

#print(inv(X))

#print(mat.dot(inv(mat)))

q, r = qr(mat)

#print(r)


samples = np.random.normal(size=(4, 4))
x_samples = randn(1000)

#print(samples)

N = 1000000
#samples.reshape(0, 16)

#%timeit samples = normalvariate(0, 1) for _ in xrange(N)

#plt.plot(x_samples)
#plt.show()

#np.random.seed(12345)

position = 0
walk = [position]

steps = 1000

for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)

#plt.plot(walk)
#plt.show()


































