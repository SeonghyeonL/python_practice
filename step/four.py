
"""
# 10807

import sys
input = sys.stdin.readline
n = int(input())
ns = list(map(int, input().split()))
v = int(input())
cnt = 0
for i in range(n):
    if ns[i]==v: cnt+=1
print(cnt)


# 10871

import sys
input = sys.stdin.readline
n, x = map(int, input().split())
a = list(map(int, input().split()))
for i in range(n):
    if a[i]<x: print(a[i], end=" ")


# 10818

import sys
input = sys.stdin.readline
n = int(input())
ns = list(map(int, input().split()))
min = ns[0]
max = ns[0]
for i in range(n):
    if ns[i]<min: min = ns[i]
    elif ns[i]>max: max = ns[i]
print(min, max)


# 2562

import sys
input = sys.stdin.readline
max = 0
idx = -1
for i in range(1, 10):
    temp = int(input())
    if temp>max:
        max = temp
        idx = i
print(max)
print(idx)


# 10810

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
b = []
for n in range(N):
    b.append(0)
for m in range(M):
    i, j, k = map(int, input().split())
    for idx in range(i-1, j):
        b[idx] = k
for n in range(N):
    print(b[n], end=" ")
"""

# 10813

import sys
input = sys.stdin.readline

