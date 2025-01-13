import numpy as np
from itertools import product

def nearest(a, b):

    na, nb = len(a), len(b)
    ## Combinations of a and b
    comb = product(range(na), range(nb))
    ## [[distance, index number(a), index number(b)], ... ]
    l = [[np.linalg.norm(a[ia] - b[ib]), ia, ib] for ia, ib in comb]
    ## Sort with distance
    l.sort(key=lambda x: x[0])

    xa = []
    xb = []
    d = []
    for _ in range(min(na, nb)):
        m, ia, ib = l[0]
        xa.append(ia) # 元データ配列からの削除用に追加
        xb.append(ib) # 同じ
        d.append([m, a[ia], b[ib]])  # 最短結果通知用に格納
        ## Remove items with same index number
        l = list(filter(lambda x: x[1] != ia and x[2] != ib, l))

    a = np.delete(a, xa, 0) # 元データ配列からデータ削除
    b = np.delete(b, xb, 0) # 同じ

    return a, b, d

a = np.array([[38,139], [60,150], [188, 71], [41, 138]])
b = np.array([[23,188], [70, 172], [52, 196]])
ra, rb, rd = nearest(a, b)