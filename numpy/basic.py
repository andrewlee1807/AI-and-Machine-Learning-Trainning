import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float')
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

# Solution :
temp = temp = np.where(np.isnan(iris_2d))
nan_array_position = np.c_[temp[0], temp[1]]

sample = iris_2d[nan_array_position[0,0], nan_array_position[0,1]]

# compute the row wise counts of all possible values in an array
np.random.seed(100)
arr = np.random.randint(1,11,size=(6, 10))

# Q. Compute the euclidean distance between two arrays a and b.
a = np.array([1,2,3,4,5])
b = np.array([4,5,6,7,8])
print(np.linalg.norm(a-b))



