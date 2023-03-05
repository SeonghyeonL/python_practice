
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


# 11650

import sys
input = sys.stdin.readline
N = int(input())
a = []
for n in range(N):
    x, y = map(int, input().split())
    a.append((x, y))
a.sort(key=lambda tu: (tu[0], tu[1]))
for n in range(N):
    print(a[n][0], a[n][1])


# 11651

import sys
input = sys.stdin.readline
N = int(input())
a = []
for n in range(N):
    x, y = map(int, input().split())
    a.append((x, y))
a.sort(key=lambda tu: (tu[1], tu[0]))
for n in range(N):
    print(a[n][0], a[n][1])


# 1181

import sys
input = sys.stdin.readline
N = int(input())
a = []
for n in range(N):
    temp = input().strip()
    if temp not in a:
        a.append(temp)
a.sort()
a.sort(key=len)
for n in range(len(a)):
    print(a[n])


# 10814

import sys
input = sys.stdin.readline
N = int(input())
a = []
for n in range(N):
    x, y = map(str, input().strip().split())
    x = int(x)
    a.append((x, y))
a.sort(key=lambda tu: tu[0])
for n in range(N):
    print(a[n][0], a[n][1])


# 18870

import sys
input = sys.stdin.readline
N = int(input())
X = list(map(int, input().split()))
X2 = sorted(X)
dic = {}
cnt = 0
for i in range(N):
    if X2[i] not in dic.keys():
        dic[X2[i]] = cnt
        cnt += 1
for i in range(N):
    print(dic[X[i]], end=" ")


# -----------------------------------
# twelve


# 10872

res = [1]
for i in range(12):
    res.append(0)

def fac(N):
    if res[N] != 0: return res[N]
    else: return fac(N-1)*N

import sys
input = sys.stdin.readline
N = int(input())
print(fac(N))


# 10870

res = [0, 1]
for i in range(20):
    res.append(-1)

def fib(N):
    if res[N] == -1: res[N] = fib(N-2)+fib(N-1)
    return res[N]

import sys
input = sys.stdin.readline
n = int(input())
print(fib(n))


# 25501

def recursion(s, l, r, cnt):
    cnt += 1
    if l >= r: return 1, cnt
    elif s[l] != s[r]: return 0, cnt
    else: return recursion(s, l+1, r-1, cnt)

def isPalindrome(s):
    return recursion(s, 0, len(s)-1, 0)

import sys
input = sys.stdin.readline
T = int(input())
for t in range(T):
    S = input().strip()
    temp = isPalindrome(S)
    print(temp[0], temp[1])


# 24060

def merge_sort(A, p, r, cnt):
    if p<r:
        q = int((p+r)/2)
        cnt1 = merge_sort(A, p, q, cnt)
        cnt2 = merge_sort(A, q+1, r, cnt1)
        cnt = merge(A, p, q, r, cnt2)
    return cnt

def merge(A, p, q, r, cnt):
    i, j = p, q+1
    temp = []
    while i<=q and j<=r:
        if A[i]<=A[j]:
            temp.append(A[i])
            i += 1
        else:
            temp.append(A[j])
            j += 1
    while i<=q:
        temp.append(A[i])
        i += 1
    while j<=r:
        temp.append(A[j])
        j += 1
    i, t = p, 0
    while i<=r:
        A[i] = temp[t]
        cnt += 1
        if cnt == K: print(temp[t])
        i += 1
        t += 1
    return cnt

import sys
input = sys.stdin.readline
N, K = map(int, input().split())
A = list(map(int, input().split()))
cnt = 0
cnt = merge_sort(A, 0, N-1, cnt)
if cnt < K: print(-1)


# 2447

def makestar(star, A, B, n):
    step = int(n/3)
    for a in range(A+step, A+2*step):
        for b in range(B+step, B+2*step):
            star[a][b] = 0
    if step>1:
        makestar(star, A, B, step)
        makestar(star, A, B+step, step)
        makestar(star, A, B+2*step, step)
        makestar(star, A+step, B, step)
        makestar(star, A+step, B+2*step, step)
        makestar(star, A+2*step, B, step)
        makestar(star, A+2*step, B+step, step)
        makestar(star, A+2*step, B+2*step, step)

import sys
input = sys.stdin.readline
N = int(input())
star = []
for i in range(N):
    temp = []
    for j in range(N):
        temp.append(1)
    star.append(temp)
makestar(star, 0, 0, N)
for i in range(N):
    for j in range(N):
        if star[i][j]==1: print("*", end="")
        else: print(" ", end="")
    print()
"""

# 11729

def hanoi(now, to, temp, n):
    if n == 1:
        print(now, to)
    else:
        hanoi(now, temp, to, n-1)
        hanoi(now, to, temp, 1)
        hanoi(temp, to, now, n-1)

import sys
input = sys.stdin.readline
N = int(input())
sum = 1
for _ in range(1, N):
    sum = sum * 2 + 1
print(sum)
hanoi(1, 3, 2, N)

