
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


# 10813

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
b = []
temp = 0
for n in range(N):
    b.append(n+1)
for m in range(M):
    i, j = map(int, input().split())
    temp = b[i-1]
    b[i-1] = b[j-1]
    b[j-1] = temp
for n in range(N):
    print(b[n], end=" ")


# 5597

import sys
input = sys.stdin.readline
b = []
temp = 0
for i in range(30):
    b.append(0)
for i in range(28):
    temp = int(input())
    b[temp-1] = 1
for i in range(30):
    if b[i]==0: print(i+1)


# 3052

import sys
input = sys.stdin.readline
a = []
temp = 0
cnt = 0
for i in range(42):
    a.append(0)
for i in range(10):
    temp = int(input())
    a[temp%42] = 1
for i in range(42):
    if a[i]==1: cnt += 1
print(cnt)


# 10811

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
a = []
temp = 0
for n in range(N):
    a.append(n+1)
for m in range(M):
    i, j = map(int, input().split())
    while i < j:
        temp = a[i-1]
        a[i-1] = a[j-1]
        a[j-1] = temp
        i += 1
        j -= 1
for n in range(N):
    print(a[n], end=" ")
"""

# 1546

import sys
input = sys.stdin.readline
n = int(input())
score = list(map(int, input().split()))
max = -1
sum = 0
for i in range(n):
    if score[i]>max: max = score[i]
for i in range(n):
    sum += score[i] / max * 100
print(sum/n)


