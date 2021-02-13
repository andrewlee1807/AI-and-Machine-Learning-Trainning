# Author : Andrew Lee

class SearchBinary:
    def __init__(self, item, ls):
        self.item = item
        self.ls = ls

    def _dfs(self, head, tail):
        mid = round((head + tail) / 2)
        if self.item == self.ls[mid]:
            return True
        if head >= tail:
            return False
        if self.item < self.ls[mid]:
            return self._dfs(head, mid - 1)
        else:
            return self._dfs(mid + 1, tail)

    def search(self):
        return self._dfs(0, len(self.ls) - 1)


def ex1():
    ls = [i for i in range(100)]
    t = int(input("Input number to find "))
    searchBinary = SearchBinary(t, ls)
    print(searchBinary.search())


def ex2():
    num = int(input("Input : "))
    rs = [i for i in range(1, num + 1) if num % i == 0]
    print(rs)


def ex3():
    st = input("Input : ")
    st = [i for i in st]
    st_reverse = st.copy()
    st_reverse.reverse()
    if st == st_reverse:
        print("Yes")
    else:
        print("No")


# FIBONACCI
def ex4():
    num = int(input("Number of PIBONACCI : "))

    def fibo(i):
        if i > 2:
            return fibo(i - 1) + fibo(i - 2)
        if i == 1 or i == 2:
            print(i, end=',')
            return 1

    print(fibo(num))


def ex5():
    st = input("Input : ")
    st_split = st.split(',')
    st_split.sort()
    st_sort = ','.join(st_split) + '.'
    print(st_sort)


def ex6():
    up = int(input('UP : '))
    down = int(input('DOWN : '))
    left = int(input('LEFT : '))
    right = int(input('RIGHT : '))
    import math
    print("Distance ROBOT from 0 to currently State : " + str(int(math.sqrt((up - down) ** 2 + (right - left) ** 2))))


def main():
    # ex1()
    # ex2()
    # ex3()
    # ex4()
    # ex5()
    ex6()


if __name__ == '__main__':
    main()
