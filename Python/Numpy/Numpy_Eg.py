#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-09-15 20:01:15
# @Author  : Leo Zhu (leapold_zhu@yeah.net)
# @Link    : ${link}
# @Version : $Id$

import numpy as np

# Class 1
# Basic atributes
array = np.array([[[1, 2, 3]],
                  [[2, 3, 4]],
                  [[1, 2, 3]]])

print(array)
print('number of dim: ', array.ndim)
print('shape: ', array.shape)
print('size: ', array.size)

# Class 2
# Generate numpy object
# float64/32/16
print("====================")

a = np.array([2, 23, 5], dtype = np.int)
print(a)
print(a.dtype)
a = np.zeros((3, 4))
print(a)
a = np.ones((3, 4))
print(a)
b = np.empty((3, 4))
print(b)
a = np.arange(10, 20, 2)
print(a)
a = np.arange(12).reshape((3, 4))
print(a)
a = np.linspace(1, 10, 5)
print(a)

# Class 3
# Basic operations P1
print("====================")

a = np.array([10,20,30,40])
b = np.arange(4)

print(a, b)
print(a - b)
print(a + b)
print(b**2)
print(np.sin(a))
print(b < 3)
print(b[b < 3])

a = np.array([[1, 1],
              [0, 1]])
b = np.arange(4).reshape(2, 2)
print(a, '\n', b) 
# num multi num
c = a*b
print(c)
# matrix multi
c = a.dot(b)
print(c)

a = np.random.random((2, 4))
print(a)
print(np.sum(a))
print(np.max(a))
print(np.max(a))

# Class 4
# Basic operations P2
print("====================")
A = np.arange(2, 14).reshape(3, 4)
print(A)
print(np.argmin(A))
print(np.argmax(A))
print(A.mean())
print(np.average(A))
print(np.median(A))
print(np.cumsum(A))
print(np.diff(A))
# first array represent #rows, second array represent #cols
print(np.nonzero(A))
print(np.sort(A))
# also note as A.T
print(np.transpose(A))
print(np.clip(A, 5, 9))

# Class 5
# Index of numpy
print("====================")
A = np.arange(3, 15).reshape((3, 4))
print(A)
print(A[2])
print(A[1, 1])
print(A[:, 1])
print(A[:, 1:2])
print(A[:, 1:3])

for col in A.T:
    print(col)

print(A.flatten())

for item in A.flat:
    print(item)

# Class 6
# Merge array
A = np.array([1,1,1])
B = np.array([2,2,2])

print(np.vstack((A, B)))
print(np.hstack((A, B)))
print(A.reshape(-1,1))
print(A[np.newaxis, :])
print(np.concatenate((A, B, B, A), axis = 0))

# Class 6
# Split array
print("====================")
A = np.arange(12).reshape((3, 4))

print(A)
# equal split
print(np.split(A, 2, axis = 1))
print(np.array_split(A, 3, axis = 1))
print(np.vsplit(A, 3))
print(np.hsplit(A, 2))

# Class 7
# Numpy copy and deep copy
print("====================")
a = np.arange(4)
b = a
c = a
d = b

print(b is a)

# deep copy
b = a.copy()
print(b is a)
