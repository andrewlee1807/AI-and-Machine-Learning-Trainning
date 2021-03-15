x = [0 for i in range(8)]
cot = [False for i in range(8)]
cheo_tong = [False for i in range(15)]
cheo_hieu = [False for i in range(15)]


def show_solution():
    for i in range(8):
        print(i, ' - ', x[i])
    print('------')


def dfs_queen(i):
    for j in range(8):
        if cot[j] == False and cheo_tong[i + j] == False and cheo_hieu[i - j + 7] == False:
            x[i] = j
            cot[j] = cheo_hieu[i - j + 7] = cheo_tong[i + j] = True
            if i == 7:
                show_solution()
            else:
                dfs_queen(i + 1)
            cot[j] = cheo_hieu[i - j + 7] = cheo_tong[i + j] = False


dfs_queen(0)

