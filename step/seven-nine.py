
"""
# -----------------------------------
# seven


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


# -----------------------------------
# eight


# 2292

import sys
input = sys.stdin.readline
N = int(input())
# 1(1) / 6(7) / 12(19) / 18(37)
room = 0
i = 0
while N > 0:
    if i == 0: N -= 1
    else: N -= 6 * i
    room += 1
    i += 1
print(room)


# 1193

import sys
input = sys.stdin.readline
X = int(input())
# 1(1) / 2(3) / 3(6) / 4(10)
i = 1
while X > 0:
    X -= i
    i += 1
i -= 1
X += i
if i%2==0: print("%d/%d" %(X, i-X+1))
else: print("%d/%d" %(i-X+1, X))


# 2869

import sys
input = sys.stdin.readline
A, B, V = map(int, input().split())
day = 1
if V > A:
    temp = (V-A)//(A-B)
    if (V-A)%(A-B)==0: day = temp + 1
    else: day = temp + 2
print(day)


# 10250

import sys
input = sys.stdin.readline
T = int(input())
for t in range(T):
    H, W, N = map(int, input().split())
    h = (N-1) % H + 1
    w = (N-1) // H + 1
    print(h*100+w)


# 2775

a = []
for i in range(15):
    list = []
    list.append(1)
    if i == 0:
        for j in range(2, 15):
            list.append(j)
    else:
        for j in range(2, 15):
            list.append(0)
    a.append(list)

def apart(k, n):
    if a[k][n] == 0:
        a[k][n] = apart(k, n-1) + apart(k-1, n)
    return a[k][n]

import sys
input = sys.stdin.readline
T = int(input())
for t in range(T):
    k = int(input())    # k???
    n = int(input())    # n???
    # 3???) 1 / 5 / 15 / 35 / 70
    # 2???) 1 / 4 / 10 / 20 / 35
    # 1???) 1 / 3 /  6 / 10 / 15
    # 0???) 1 / 2 /  3 /  4 /  5
    print(apart(k, n-1))


# 2839

import sys
input = sys.stdin.readline
N = int(input())
five = N//5
cant = False
while True:
    if (N-five*5)%3 == 0: break
    else: five -= 1
    if five<0:
        cant = True
        break
if cant: print(-1)
else: print(int(five+(N-five*5)/3))


# 10757

import sys
input = sys.stdin.readline
A, B = map(int, input().split())
print(A+B)


# -----------------------------------
# nine


# 5086

import sys
input = sys.stdin.readline
a, b = map(int, input().split())
while not(a==0 and b==0):
    if b%a==0: print('factor')
    elif a%b==0: print('multiple')
    else: print('neither')
    a, b = map(int, input().split())


# 2501

import sys
input = sys.stdin.readline
N, K = map(int, input().split())
list = []
sqrt = N**0.5
for i in range(1, int(sqrt)+1):
    if N%i==0:
        list.append(i)
        if i!=sqrt:
            list.append(N/i)
list.sort()
if len(list) < K: print(0)
else: print(int(list[K-1]))


# 9506

import sys
input = sys.stdin.readline
n = int(input())
while n != -1:
    list = []
    sqrt = n ** 0.5
    for i in range(1, int(sqrt) + 1):
        if n % i == 0:
            list.append(i)
            if i != sqrt and i != 1:
                list.append(int(n / i))
    list.sort()
    sum = 0
    for i in range(len(list)): sum += list[i]
    if sum == n:
        print(n, "= ", end="")
        for i in range(len(list)-1): print(list[i], "+ ", end="")
        print(list[len(list)-1])
    else: print(n, "is NOT perfect.")
    n = int(input())


# 1978

import sys
input = sys.stdin.readline
N = int(input())
ns = list(map(int, input().split()))
sum = 0
for n in range(N):
    temp = ns[n]
    if temp == 1: continue
    list = []
    sqrt = temp ** 0.5
    for i in range(2, int(sqrt) + 1):
        if temp % i == 0:
            list.append(i)
    if len(list)==0: sum += 1
print(sum)


# 2581

import sys
input = sys.stdin.readline
M = int(input())
N = int(input())
res = []
sum = 0
for m in range(M, N+1):
    if m == 1: continue
    list = []
    sqrt = int(m ** 0.5)
    for i in range(2, sqrt+1):
        if m%i == 0: list.append(i)
    if len(list)==0:
        res.append(m)
        sum += m
if sum==0: print(-1)
else:
    print(sum)
    print(res[0])


# 11653

import sys
input = sys.stdin.readline
N = int(input())
i = 2
while N>1:
    if N%i==0:
        print(i)
        N /= i
    else: i += 1
"""

# 9020

import sys
input = sys.stdin.readline
T = int(input())
total = [False, False] + [True] * 9999
for i in range(2, 501):
    if total[i]:
        for j in range(2*i, 10001, i): total[j] = False
for t in range(T):
    n = int(input())
    for m in range(n//2, 0, -1):
        if total[m]==1 and total[n-m]==1:
            print(m, n-m)
            break

