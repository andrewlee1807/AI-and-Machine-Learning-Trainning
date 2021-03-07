import numpy as np

a = np.array(range(5))

# dtype : data type objects
student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])  # The same Data Structures C++
# 'f' − floating-point; 'S', 'a' − (byte-)string
a = np.array([("Andrew", 12, 1)], dtype=student)

a = np.ones([3, 2], dtype=np.dtype([('x', 'i4'), ('y', 'i4')]), order='C')

a = np.arange(2, 10, 2)
a = np.linspace(2, 5, 50, retstep=False)
a = np.arange(10)
b = a[2:7:2]







