# Numpy Tutorial
## 1 Setup
    pip install Numpy    
    # If you use windows, you may need launch cmd with administrator permission 

## 2 Attributes
### 2.1 Basic attributes
- ndim: Dim of array
- shape: Check the shape of matrix
- size: How many nubers in it

### 2.2 Generate array
- np.array: Transform the list to numpy
- dtype: You can appoint type of array
- np.zeros: Matrix all 0
- np.ones: Matrix all 1
- np.arange: Generate a 1 dim list, can appoint step size
- np.linspace: Generate numbers in a range, uniformally
- np.reshape: Rearange the matrix to the exact shape
- np.random.random: Generate a random matrix

### 2.3 Basic operations
- +/-
- a**2 ==> a^2
- np.sin/cos/tan
- a*b ==> dot multi(element multiply)
- np.dot(A, B) ==> AB
- np.sum: Sum of all elements in array; axis = 1, sum of row;
axis = 0, sum of column
- np.max: axis = 1, sum of row; axis = 0, sum of column
- np.min: axis = 1, sum of row; axis = 0, sum of column
- np.argmin: Index of min ele
- np.argmax: Index of max ele
- np.mean: Mean of the matrix
- np.median: Median number of the matrix
- np.cumsum: Cumulate sum, add one by one, output all results
- np.diff: Difference between adjascent elements
- np.nonzero: Return rows & cols of nonzero elements
- np.sort: Rearrange the elements
- np.transpose/A.T ==> A'
- np.clip: Clip value of A

### 2.4 Index
- A[:, 1] is different from A[:, 1:2]
- for row in A: Row is considered as rows in A, if you want to 
iterate col of A, you can use 'for col in A.T:'
- A.flatten: Flat the matrix
- A.flat: Return an iterator

### 2.5 Merge array
- np.vstack: Vertical stack, when two matrix identical in cols
- np.hstack: Horizontal stack, when two matrix identical in rows
- A[np.newaxis, :]: Add a dimension on row; A[:, np.newaxis],add a dimension on col
- np.concatenate: You can use it concatenate multi matrix

### 2.6 Split array
- np.split: Only can split array into equal parts
- np.array_split: Can split array into different parts
- np.vsplit: Can split array into different parts, vertical
- np.hsplit: Can split array into different parts, horizontal

### 2.7 Copy and Deep Copy
- np.copy: Copy value of array

# Reference

[Vedio course](https://www.bilibili.com/video/av16378934)
