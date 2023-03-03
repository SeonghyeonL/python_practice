
"""
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
"""

# ten


# 24262

import sys
input = sys.stdin.readline

# https://www.acmicpc.net/step/53

