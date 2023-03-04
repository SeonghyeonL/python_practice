
"""
# -----------------------------------
# ten


# 24262

# O(1)
import sys
input = sys.stdin.readline
n = int(input())
print(1)
print(0)


# 24263

# O(n)
import sys
input = sys.stdin.readline
n = int(input())
print(n)
print(1)


# 24264

# O(n^2)
import sys
input = sys.stdin.readline
n = int(input())
print(n*n)
print(2)


# 24265

# O(n*(n-1)/2)
# (n-1)+(n-2)+...+(1) = n*(n-1)/2
import sys
input = sys.stdin.readline
n = int(input())
print((n*(n-1))//2)
print(2)


# 24266

# O(n^3)
import sys
input = sys.stdin.readline
n = int(input())
print(n*n*n)
print(3)

# 24267

# O(n*(n-1)*(n-2)/6)
# (n-2)*(n-1)/2+(n-3)*(n-2)/2+...+(1)*(2)/2
# = n*(n-1)*(n-2)/6
import sys
input = sys.stdin.readline
n = int(input())
print((n*(n-1)*(n-2))//6)
print(3)


# 24313

import sys
input = sys.stdin.readline
a1, a0 = map(int, input().split())
c = int(input())
n0 = int(input())
# f = a1*n+a0 / g = n / f <= c*g for n >= n0
if a1<=c:
    if a1*n0+a0 <= c*n0: print(1)
    else: print(0)
else: print(0)


# -----------------------------------
# eleven


# 2750

import sys
input = sys.stdin.readline
N = int(input())
lst = []
for n in range(N):
    temp = int(input())
    lst.append(temp)
lst.sort()
for n in range(N):
    print(lst[n])


# 2587

import sys
input = sys.stdin.readline
lst = []
sum = 0
for _ in range(5):
    temp = int(input())
    sum += temp
    lst.append(temp)
lst.sort()
print(int(sum/5))
print(lst[2])


# 25305

import sys
input = sys.stdin.readline
N, k = map(int, input().split())
grade = list(map(int, input().split()))
grade.sort(reverse=True)
print(grade[k-1])


# 2751

import sys
input = sys.stdin.readline
N = int(input())
lst = []
for n in range(N):
    temp = int(input())
    lst.append(temp)
lst.sort()
for n in range(N):
    print(lst[n])


# 10989

import sys
input = sys.stdin.readline
a = []
for _ in range(10001):
    a.append(0)
N = int(input())
for n in range(N):
    temp = int(input())
    a[temp] += 1
for i in range(10001):
    for _ in range(a[i]):
        print(i)


# 2108

import sys
input = sys.stdin.readline
a = []  # 0~8000 (0=-4000, 4000=0, 8000=4000)
for i in range(8001):
    a.append(0)
sum = 0
list = []
max, min = -4001, 4001
N = int(input())
for n in range(N):
    temp = int(input())
    a[temp+4000] += 1
    sum += temp
    list.append(temp)
    if temp>max: max = temp
    if temp<min: min = temp
list.sort()
print(round(sum/N))     # 산술평균
print(list[int(N/2)])   # 중앙값
maxcnt = 0
for i in range(8001):
    if a[i]>maxcnt: maxcnt=a[i]
maxlst = []
for i in range(8001):
    if a[i]==maxcnt: maxlst.append(i-4000)
if len(maxlst)>1: print(maxlst[1])  # 최빈값
else: print(maxlst[0])
print(abs(max-min))     # 범위


# 1427

import sys
input = sys.stdin.readline
N = input().strip()
lst = []
for i in range(len(N)):
    lst.append(int(N[i]))
lst.sort(reverse=True)
for i in range(len(N)):
    print(lst[i], end="")
"""

# 11650

import sys
input = sys.stdin.readline
N = int(input())
for n in range(N):
    x, y = map(int, input().split())



