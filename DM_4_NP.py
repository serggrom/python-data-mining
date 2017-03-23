import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn

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



























