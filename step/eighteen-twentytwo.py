
"""
# -----------------------------------
# eighteen (동적 계획법 1)

# 24416

import sys
input = sys.stdin.readline
n = int(input())
# 3 -> 2 / 4 -> 2+1 / 5 -> 3+2 / 6 -> 5+3
a = [0, 1, 1]
for i in range(n-2):
    a.append(a[i+1]+a[i+2])
res1 = a[n]
res2 = n - 2
print(res1, res2)


# 9184

def findw(a, b, c):
    if a==0 and b==0 and c==0: return w[0][0][0]
    elif a<=0 or b<=0 or c<=0: return findw(0, 0, 0)
    elif a>20 or b>20 or c>20: return findw(20, 20, 20)
    elif w[a][b][c] != 0: return w[a][b][c]
    elif a < b < c: w[a][b][c] = findw(a,b,c-1) + findw(a,b-1,c-1) - findw(a,b-1,c)
    else: w[a][b][c] = findw(a-1,b,c) + findw(a-1,b-1,c) + findw(a-1,b,c-1) - findw(a-1,b-1,c-1)
    return w[a][b][c]

import sys
input = sys.stdin.readline
a, b, c = map(int, input().split())
w = []
for _ in range(21):
    temp1 = []
    for _ in range(21):
        temp2 = [0] * 21
        temp1.append(temp2)
    w.append(temp1)
w[0][0][0] = 1
while a != -1 or b != -1 or c != -1:
    res = findw(a, b, c)
    print("w(%d, %d, %d) = %d" % (a, b, c, res))
    a, b, c = map(int, input().split())


# 1904

import sys
input = sys.stdin.readline
N = int(input())
# 1 / 2 / 3 / 4 / 5 /  6 /  7
# 1 / 2 / 3 / 5 / 8 / 13 / 21
ans = [1, 2]
for i in range(N-1):
    ans.append((ans[i] + ans[i + 1]) % 15746)
print(ans[N-1])


# 9461

import sys
input = sys.stdin.readline
T = int(input())
P = [1, 1, 1, 2, 2]
for _ in range(T):
    N = int(input())
    if len(P) < N:
        for i in range(len(P)-1, N):
            P.append(P[i]+P[i-4])
    print(P[N-1])


# 1912

import sys
input = sys.stdin.readline
n = int(input())
ns = list(map(int, input().split()))
last = ns[0]
totalmax = last
ans = [ns[0]]
for i in range(1, n):
    last = max(last+ns[i], ns[i])
    ans.append(last)
    if totalmax<last: totalmax = last
print(totalmax)


# 1149

import sys
input = sys.stdin.readline
N = int(input())
house = []
for _ in range(N):
    rgb = list(map(int, input().split()))
    house.append(rgb)
cost = []
for _ in range(N):
    temp = [0, 0, 0]
    cost.append(temp)
for n in range(N):
    if n == 0: cost[0] = house[0]
    else:
        cost[n][0] = min(cost[n-1][1], cost[n-1][2]) + house[n][0]
        cost[n][1] = min(cost[n - 1][0], cost[n - 1][2]) + house[n][1]
        cost[n][2] = min(cost[n - 1][0], cost[n - 1][1]) + house[n][2]
print(min(cost[N-1][0], cost[N-1][1], cost[N-1][2]))


# 1932

import sys
input = sys.stdin.readline
N = int(input())
sum = []
for n in range(1, N+1):
    line = list(map(int, input().split()))
    temp = []
    if n == 1: temp.append(line[0])  # line 1 (one element)
    else:
        for i in range(n):
            if i == 0: temp.append(sum[n-2][0]+line[0])  # line n+1, element 1
            elif i == n-1: temp.append(sum[n-2][i-1]+line[i])  # element n+1
            else: temp.append(max(sum[n-2][i-1], sum[n-2][i])+line[i])
    sum.append(temp)
maxelement = 0
for j in range(N):
    if sum[N-1][j]>maxelement: maxelement = sum[N-1][j]
print(maxelement)


# 2579

import sys
input = sys.stdin.readline
N = int(input())
stair = []
for n in range(N):
    temp = int(input())
    stair.append(temp)
point = []  # [누적합(스텝1), 누적합(스텝2)]
for n in range(N):
    if n == 0: point.append([stair[0], 0])
    elif n == 1: point.append([stair[0]+stair[1], stair[1]])
    else:
        # 한 칸 오른 거라면, n-1번째가 두 칸 오른 거여야 함
        step1 = point[n-1][1] + stair[n]
        # 두 칸 오른 거라면, n-2번째가 한 칸 혹은 두 칸 오른 거여야 함
        step2 = max(point[n-2][0], point[n-2][1]) + stair[n]
        point.append([step1, step2])
print(max(point[N-1][0], point[N-1][1]))


# 1463

import sys
input = sys.stdin.readline
N = int(input())
cnt = [-1] * (N+1)
for i in range(1, N+1):
    if i == 1: cnt[i] = 0
    else:
        if i % 6 == 0: cnt[i] = min(cnt[i-1], cnt[i//3], cnt[i//2]) + 1
        elif i % 3 == 0: cnt[i] = min(cnt[i-1], cnt[i//3]) + 1
        elif i % 2 == 0: cnt[i] = min(cnt[i-1], cnt[i//2]) + 1
        else: cnt[i] = cnt[i-1] + 1
print(cnt[N])
"""

# 10844

import sys
input = sys.stdin.readline
N = int(input())








# https://www.acmicpc.net/step/16

