import numpy as np
a = np.random.randint(1,20, 10)
print(a)
for i in range(0, len(a)-1):
    min = a[i]
    index = i
    for j in range(i+1, len(a)):
        if a[j] < min:
            min = a[j]
            index = j
    a[i], a[index] = a[index], a[i]
print(a)


