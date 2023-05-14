
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


# 2933

import sys
from collections import deque
input = sys.stdin.readline
move = [(1, 0), (-1, 0), (0, 1), (0, -1)]
R, C = map(int, input().split())
maps = []
for _ in range(R):
    cs = [i for i in input().strip()]  # '.'는 빈 칸, 'x'는 미네랄
    maps.append(cs)
N = int(input())
height = list(map(int, input().split()))

def BFS(x_in, y_in):
    result = [(x_in, y_in)]  # 클러스터 구성원
    visit = [[False for _ in range(C)] for _ in range(R)]  # 방문 표시
    visit[y_in][x_in] = True
    q = deque([(x_in, y_in)])
    maps[y_in][x_in] = "."
    while len(q) > 0:
        x, y = q.popleft()
        for j in range(4):
            my = y + move[j][0]
            mx = x + move[j][1]
            if 0 <= my < R and 0 <= mx < C and maps[my][mx] == "x" and visit[my][mx] == False:
                result.append((mx, my))
                visit[my][mx] = True
                q.append((mx, my))
                maps[my][mx] = "."
    fall_num = 0
    flag = True
    while flag:
        fall_num += 1
        for x, y in result:
            my = y + fall_num
            if my == R or maps[my][x] == "x":
                flag = False
                break  # 더 이상 못 내려감
    for x, y in result:
        maps[y + fall_num - 1][x] = "x"

for idx in range(len(height)):
    target_y = R - height[idx]
    for target_x in range(C) if idx % 2 == 0 else range(C-1, -1, -1):
        if maps[target_y][target_x] == "x":
            maps[target_y][target_x] = "."
            for i in range(4):
                ny = target_y + move[i][0]
                nx = target_x + move[i][1]
                if 0 <= ny < R and 0 <= nx < C and maps[ny][nx] == "x":
                    BFS(nx, ny)
            break  # 막대가 미네랄 만남

for line in maps:
    print(''.join(map(str, line)))


# 2749

import sys
input = sys.stdin.readline
n = int(input())
num = 1000000
A = [[1, 1], [1, 0]]
# n번째 피보나치 수를 1,000,000으로 나눈 나머지를 출력
# 단, 0번째 피보나치 수는 0이고 1번째 피보나치 수는 1이고 2번째부터는 앞의 두 개 합
# 아주 큰 수의 피보나치를 빠르게 구하기 위해서는 분할정복을 이용한 '행렬의 거듭제곱'을 사용해야 함
# (Fn+1 Fn    = (1 1
#  Fn   Fn-1)    1 0) ^ n for 자연수 n

def cal(x):
    if x == 1:
        return A
    else:  # x % 2 == 0 and x % 2 == 1
        half = cal(x//2)
        a = [[0, 0], [0, 0]]
        a[0][0] = (half[0][0] * half[0][0] + half[0][1] * half[1][0]) % num
        a[1][0] = (half[1][0] * half[0][0] + half[1][1] * half[1][0]) % num
        a[0][1] = (half[0][0] * half[0][1] + half[0][1] * half[1][1]) % num
        a[1][1] = (half[1][0] * half[0][1] + half[1][1] * half[1][1]) % num
        if x % 2 == 0:
            return a
        else:  # x % 2 == 1
            b = [[0, 0], [0, 0]]
            b[0][0] = (a[0][0] * A[0][0] + a[0][1] * A[1][0]) % num
            b[1][0] = (a[1][0] * A[0][0] + a[1][1] * A[1][0]) % num
            b[0][1] = (a[0][0] * A[0][1] + a[0][1] * A[1][1]) % num
            b[1][1] = (a[1][0] * A[0][1] + a[1][1] * A[1][1]) % num
            return b

if n == 1: print(A[0][0])
else: print(cal(n - 1)[0][0])
"""

# 9376

import sys
input = sys.stdin.readline
T = int(input())
for _ in range(T):
    h, w = map(int, input().split())
    for _ in range(h):
        line = input().strip()  # 빈 공간 '.', 지나갈 수 없는 벽 '*', 문 '#', 죄수 '$'

    # 두 죄수를 탈옥시키기 위해서 열어야 하는 문의 최솟값 출력











# 0513 bad condition
# https://www.acmicpc.net/workbook/view/4349