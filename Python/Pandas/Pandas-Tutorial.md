# Pandas Tutorial
## 1 Setup
    pip install Pandas 
    # If you use windows, you may need launch cmd with administrator permission 
It's more like a numpy in dictionary

## 2 Attributes
### 2.1 Generate Pandas Data
- pd.Series
- pd.date_range: periods appoint number of items generated
- pd.DataFrame
- dtypes
- parts of dataframe
    - index: Label for rows
    - columns: Label for cols
    - values: Label for values
- describe: show some statistic information of data
- .T
- sort_index: Sort data at some appoint order
- sort_values: Sort data by some appointed values

### 2.2 Choose data
Just like list, np.DataFrame can use slice to choose data. We also can make a boolean selection as A[A.attribute > 0].
- .loc: Select by label
- .iloc: Select by position
- .ix: Mixed selection, use label and position

### 2.3 Set value
Just like list in Python, choose data first, then change it's value
You can also choose to add a new column by definite it's value.

### 2.4 Solve Nan value
- dropna(axis, how): Drop data according to axis and how, 0: row; 1: column; any: any data is Nan, drop; all: all data is Nan, drop
- fillna: fill Nan with appointed value
- isnull: Check if there's Nan ==> np.any(df.isnull()) == True

### 2.5 Import and Export Data
Import: read; Export: to

[Official Doc](http://pandas.pydata.org/pandas-docs/stable/io.html)

### 2.6 Concatenating
- pd.concat: join attribute: outer, fill with Nan; inner, cut different parts; join_axes, keep data in both DataFrame with same index
- append: vertical append

### 2.7 Merge
- pd.merge: Merge two DataFrame base on appointed attribute, 'key'. You may use two 'key'.
- pd.join: similar to merge

# Reference

[Vedio course](https://www.bilibili.com/video/av16378934)

[Code](https://github.com/MorvanZhou/tutorials/tree/master/numpy%26pandas)