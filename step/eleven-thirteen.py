
"""
# -----------------------------------
# eleven (시간 복잡도)


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
# twelve (브루트 포스)


# 2798

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
cards = list(map(int, input().split()))
max = 0
for i in range(len(cards)):
    for j in range(i+1, len(cards)):
        for k in range(j+1, len(cards)):
            if cards[i]+cards[j]+cards[k]>max and cards[i]+cards[j]+cards[k]<=M:
                max = cards[i]+cards[j]+cards[k]
print(max)


# 2231

import sys
input = sys.stdin.readline
N = int(input())
found = False
for i in range(N):
    sum = i
    temp2 = i
    if temp2 >= 10:
        while temp2>9:
            sum += temp2%10
            temp2 = temp2//10
    sum += temp2
    if sum == N:
        print(i)
        found = True
        break
if found==False: print(0)


# 19532

import sys
input = sys.stdin.readline
a, b, c, d, e, f = map(int, input().split())
x = (b * f - c * e) // (b * d - a * e)
y = (a * f - c * d) // (a * e - b * d)
print(x, y)


# 7568

import sys
input = sys.stdin.readline
N = int(input())
people = []
for _ in range(N):
    x, y = map(int, input().split())
    people.append([x, y])
for n in range(N):
    cnt = 0
    for i in range(N):
        if i==n: continue
        if people[n][0]<people[i][0] and people[n][1]<people[i][1]: cnt += 1
    print(cnt+1, end=" ")


# 1018

def whitecheck(board, a, b):
    cnt = 0
    for A in range(a, a+8, 2):
        for B in range(b, b+8):
            if B%2==0 and board[A][B]=='B': cnt += 1
            elif B%2==1 and board[A][B]=='W': cnt += 1
    for A in range(a+1, a+8, 2):
        for B in range(b, b+8):
            if B%2==0 and board[A][B]=='W': cnt += 1
            elif B%2==1 and board[A][B]=='B': cnt += 1
    return cnt

def blackcheck(board, a, b):
    cnt = 0
    for A in range(a, a+8, 2):
        for B in range(b, b+8):
            if B%2==0 and board[A][B]=='W': cnt += 1
            elif B%2==1 and board[A][B]=='B': cnt += 1
    for A in range(a+1, a+8, 2):
        for B in range(b, b+8):
            if B%2==0 and board[A][B]=='B': cnt += 1
            elif B%2==1 and board[A][B]=='W': cnt += 1
    return cnt

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
board = []
for _ in range(N):
    temp = input().strip()
    board.append(temp)
minimal = 64
for a in range(N-7):
    for b in range(M-7):
        cnt = min(whitecheck(board, a, b), blackcheck(board, a, b))
        if cnt<minimal: minimal = cnt
print(minimal)


# 1436

import sys
input = sys.stdin.readline
N = int(input())
cnt = 0
num = 666
while cnt<N:
    if '666' in str(num): cnt += 1
    num += 1
print(num-1)


# 2839

import sys
input = sys.stdin.readline
N = int(input())
temp = N // 5
while True:
    cal = N - temp * 5
    if cal % 3 == 0:
        print(temp + int(cal / 3))
        exit(0)
    else:
        temp -= 1
        if temp < 0:
            print(-1)
            exit(0)


# -----------------------------------
# thirteen (정렬)


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
"""

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


