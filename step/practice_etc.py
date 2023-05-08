
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
"""

# 3197

import sys
sys.setrecursionlimit(10 ** 6)
input = sys.stdin.readline
inf = sys.maxsize
R, C = map(int, input().split())
visit = [[False for _ in range(C)] for _ in range(R)]
day = [[0 for _ in range(C)] for _ in range(R)]
bird = []
move = [(0, 1), (0, -1), (1, 0), (-1, 0)]
for r in range(R):
    line = input().strip()
    for c in range(C):
        if line[c] == "L":
            bird.append((r, c))
            visit[r][c] = True
        elif line[c] == ".":
            visit[r][c] = True

def ice(y, x):
    ans = inf
    for dy, dx in move:
        ny = y + dy
        nx = x + dx
        if 0 <= ny < R and 0 <= nx < C and visit[ny][nx] == True:
            ans = min(ans, day[ny][nx] + 1)
    day[y][x] = ans

for r in range(R):
    for c in range(C):
        if visit[r][c] == False:
            ice(r, c)
            visit[r][c] = True

def solving(y, x, maxi):
    if y == bird[1][0] and x == bird[1][1]:
        global mini
        mini = min(mini, maxi)
    else:
        for dy, dx in move:
            ny = y + dy
            nx = x + dx
            if 0 <= ny < R and 0 <= nx < C and visit[ny][nx] == False:
                visit[ny][nx] = True
                maxi = max(maxi, day[ny][nx])
                solving(ny, nx, maxi)
                visit[ny][nx] = False

mini = inf
visit = [[False for _ in range(C)] for _ in range(R)]
visit[bird[0][0]][bird[0][1]] = True
solving(bird[0][0], bird[0][1], 0)
print(mini)





# https://www.acmicpc.net/workbook/view/4349