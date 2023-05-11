
# 단기간 성장

"""
# 12865

import sys
input = sys.stdin.readline
N, K = map(int, input().split())
things = [(0, 0)]
for _ in range(N):
    W, V = map(int, input().split())
    things.append((W, V))
dp = [[0 for _ in range(K + 1)] for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(1, K + 1):
        if j >= things[i][0]:
            dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - things[i][0]] + things[i][1])
        else:  # j < things[i][0]
            dp[i][j] = dp[i - 1][j]
print(dp[-1][-1])


# 1655

import sys
import heapq
input = sys.stdin.readline
N = int(input())
left = []
right = []
for _ in range(N):
    n = int(input())
    if len(left) == len(right): heapq.heappush(left, -n)
    else: heapq.heappush(right, n)
    if len(right) > 0 and -left[0] > right[0]:
        maxi = -heapq.heappop(left)
        mini = heapq.heappop(right)
        heapq.heappush(left, -mini)
        heapq.heappush(right, maxi)
    print(-left[0])


# 3197

import sys
from collections import deque
input = sys.stdin.readline
R, C = map(int, input().split())
c = [[0 for _ in range(C)] for _ in range(R)]
wc = [[0 for _ in range(C)] for _ in range(R)]
lake, swan = [], []  # 전체맵, 백조초기위치
q, q_temp, wq, wq_temp = deque(), deque(), deque(), deque()  # 백조큐, 물큐
move = [(0, 1), (0, -1), (1, 0), (-1, 0)]

for r in range(R):
    line = list(input().strip())
    lake.append(line)
    for cc in range(C):
        if line[cc] == "L":
            swan.append((r, cc))
            wq.append((r, cc))
        elif line[cc] == ".":
            wc[r][cc] = 1
            wq.append((r, cc))

q.append(swan[0])
lake[swan[0][0]][swan[0][1]], lake[swan[1][0]][swan[1][1]] = ".", "."
c[swan[0][0]][swan[0][1]] = 1
cnt = 0

def bfs():
    while len(q) > 0:
        y, x = q.popleft()
        if y == swan[1][0] and x == swan[1][1]: return True
        for i in range(4):
            ny = y + move[i][0]
            nx = x + move[i][1]
            if 0 <= ny < R and 0 <= nx < C and c[ny][nx] == 0:
                if lake[ny][nx] == ".":
                    q.append((ny, nx))
                else:  # lake[ny][nx] == "X"
                    q_temp.append((ny, nx))
                c[ny][nx] = 1
    return False

def melt():
    while len(wq) > 0:
        y, x = wq.popleft()
        if lake[y][x] == "X": lake[y][x] = "."
        for i in range(4):
            ny = y + move[i][0]
            nx = x + move[i][1]
            if 0 <= ny < R and 0 <= nx < C and wc[ny][nx] == 0:
                if lake[ny][nx] == ".":
                    wq.append((ny, nx))
                else:  # lake[ny][nx] == "X"
                    wq_temp.append((ny, nx))
                wc[ny][nx] = 1

while True:
    melt()
    if bfs() == True:
        print(cnt)
        break
    q, wq = q_temp, wq_temp  # change (next)
    q_temp, wq_temp = deque(), deque()  # reset
    cnt += 1


# 11401

import sys
input = sys.stdin.readline
N, K = map(int, input().split())
p = 1000000007
A, B = 1, 1
for i in range(1, N + 1): A = (A * i) % p  # n!
for i in range(1, K + 1): B = (B * i) % p  # k!
for i in range(1, N - K + 1): B = (B * i) % p  # (n-k)!

def cal(x, y):
    if y == 1: return x % p
    elif y % 2 == 0: return (cal(x, y//2) ** 2) % p
    else: return ((cal(x, y//2) ** 2) * x) % p

print((A * cal(B, p - 2)) % p)


# 10830

import sys
input = sys.stdin.readline
N, B = map(int, input().split())
A = []
for _ in range(N): A.append(list(map(int, input().split())))
for i in range(N):
    for j in range(N):
        A[i][j] = A[i][j] % 1000
# A를 B제곱한 결과

def cal(x):
    if x == 1:
        return A
    else:  # x % 2 == 0 and x % 2 == 1
        half = cal(x//2)
        a = []
        for i in range(N):
            line = []
            for j in range(N):
                temp = 0
                for k in range(N):
                    temp += half[i][k] * half[k][j]
                line.append(temp % 1000)
            a.append(line)
        if x % 2 == 0:
            return a
        else:  # x % 2 == 1
            b = []
            for i in range(N):
                line = []
                for j in range(N):
                    temp = 0
                    for k in range(N):
                        temp += a[i][k] * A[k][j]
                    line.append(temp % 1000)
                b.append(line)
            return b

ans = cal(B)
for i in range(N):
    print(' '.join(map(str, ans[i])))
"""

# 2933

import sys
input = sys.stdin.readline
R, C = map(int, input().split())
for _ in range(R):
    cs = input().strip()  # '.'는 빈 칸, 'x'는 미네랄
N = int(input())
height = list(map(int, input().split()))





# https://www.acmicpc.net/workbook/view/4349