
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
"""

# 11401

import sys
input = sys.stdin.readline







# https://www.acmicpc.net/workbook/view/4349