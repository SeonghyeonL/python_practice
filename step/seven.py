
"""
# 2738 (numpy - error)

import sys
import numpy as np
input = sys.stdin.readline
N, M = map(int, input().split())
A, B = np.zeros(shape=(N, M)), np.zeros(shape=(N, M))
for n in range(N):
    temp = list(map(int, input().split()))
    A[n] = temp
for n in range(N):
    temp = list(map(int, input().split()))
    B[n] = temp
res = A + B
for n in range(N):
    for m in range(M):
        print(int(res[n][m]), end=" ")
    print()


# 2738

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
A, B = [], []
for n in range(N):
    temp = list(map(int, input().split()))
    A.append(temp)
for n in range(N):
    temp = list(map(int, input().split()))
    B.append(temp)
for n in range(N):
    for m in range(M):
        print(A[n][m]+B[n][m], end=" ")
    print()


# 2566

import sys
input = sys.stdin.readline
a = []
for i in range(9):
    temp = list(map(int, input().split()))
    a.append(temp)
max = -1
max_row = -1
max_col = -1
for i in range(9):
    for j in range(9):
        if a[i][j]>max:
            max = a[i][j]
            max_row, max_col = i, j
print(max)
print(max_row+1, max_col+1)


# 10798

import sys
input = sys.stdin.readline
a = []
for i in range(5):
    temp = input().strip()
    for j in range(15-len(temp)):
        temp += "_"
    a.append(temp)
res = ""
for i in range(15):
    for j in range(5):
        if a[j][i] != "_": res += a[j][i]
print(res)
"""

# 2563

import sys
input = sys.stdin.readline
N = int(input())
a = []
for i in range(100):
    temp = []
    for j in range(100):
        temp.append(0)
    a.append(temp)
for n in range(N):
    x, y = map(int, input().split())
    for i in range(x, x+10):
        for j in range(y, y+10):
            a[i][j] = 1
sum = 0
for i in range(100):
    for j in range(100):
        sum += a[i][j]
print(sum)


