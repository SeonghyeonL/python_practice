
"""
# -----------------------------------
# twentynine (최단 경로)


# 1753

import sys
input = sys.stdin.readline
V, E = map(int, input().split())
K = int(input())
connect = [[] for _ in range(V + 1)]
for _ in range(E):
    u, v, w = map(int, input().split())
    connect[u].append([v, w])
visit = [False] * (V + 1)
res = [float("inf")] * (V + 1)
res[K] = 0
idx = K
while True:
    visit[idx] = True
    minvalue = float("inf")
    nextidx = idx
    for i in connect[idx]:
        res[i[0]] = min(res[i[0]], res[idx] + i[1])
    for i in range(1, V + 1):
        if visit[i] == False and res[i] < minvalue:
            nextidx = i
            minvalue = res[i]
    if nextidx == idx or minvalue == float("inf"): break
    else: idx = nextidx
for i in range(1, V + 1):
    if res[i] != float("inf"): print(res[i])
    else: print("INF")


# 1504

import sys
input = sys.stdin.readline
N, E = map(int, input().split())
connect = [[] for _ in range(N + 1)]
for _ in range(E):
    a, b, c = map(int, input().split())
    connect[a].append([b, c])
    connect[b].append([a, c])
v1, v2 = map(int, input().split())

def cal(start, N):
    visit = [False] * (N + 1)
    res = [float("inf")] * (N + 1)
    res[start] = 0
    idx = start
    while True:
        visit[idx] = True
        minvalue = float("inf")
        nextidx = idx
        for i in connect[idx]:
            res[i[0]] = min(res[i[0]], res[idx] + i[1])
        for i in range(1, N + 1):
            if visit[i] == False and res[i] < minvalue:
                nextidx = i
                minvalue = res[i]
        if nextidx == idx or minvalue == float("inf"):
            break
        else:
            idx = nextidx
    return res

start1 = cal(1, N)
startv1 = cal(v1, N)
startv2 = cal(v2, N)
ans = min(start1[v1] + startv1[v2] + startv2[N], start1[v2] + startv2[v1] + startv1[N])
print(ans if ans < float("inf") else -1)


# 13549

import sys
import heapq
input = sys.stdin.readline
N, K = map(int, input().split())
result = [float("inf")] * 100001
result[N] = 0
heap = []

def cal(n, k):
    if k <= n:  # 작으면 -1 할 수밖에 없음
        return n - k
    else:
        heapq.heappush(heap, [0, n])  # 처음 위치
        while len(heap) > 0:
            n = heapq.heappop(heap)[1]  # 제일 작은 가중치를 갖는 위치
            for x in [n + 1, n - 1, n * 2]:
                if 0 <= x <= 100000:
                    if result[x] == float("inf"):
                        if x == n * 2:
                            result[x] = result[n]  # result[n] == w
                        else:
                            result[x] = result[n] + 1
                        heapq.heappush(heap, [result[x], x])
        return result[k]

print(cal(N, K))


# 9370

import sys
import heapq
input = sys.stdin.readline
T = int(input())
inf = sys.maxsize  # inf + inf = inf로 아래 test를 통과해 버릴 수 있음

def cal(start):
    heap = []
    heapq.heappush(heap, [0, start])
    res = [inf] * (n + 1)
    res[start] = 0
    while len(heap) > 0:
        now = heapq.heappop(heap)[1]
        for num, weight in connect[now]:
            if weight + res[now] < res[num]:
                res[num] = weight + res[now]
                heapq.heappush(heap, [weight + res[now], num])
    return res

for _ in range(T):
    n, m, t = map(int, input().split())
    s, g, h = map(int, input().split())
    connect = [[] for _ in range(n + 1)]
    for _ in range(m):
        a, b, d = map(int, input().split())
        connect[a].append([b, d])
        connect[b].append([a, d])
    x = []
    for _ in range(t): x.append(int(input()))
    starts = cal(s)
    startg = cal(g)
    starth = cal(h)
    ans = []
    for i in range(len(x)):
        if starts[g] + startg[h] + starth[x[i]] == starts[x[i]]\
                or starts[h] + starth[g] + startg[x[i]] == starts[x[i]]:
            ans.append(x[i])
    ans.sort()
    print(' '.join(map(str, ans)))
"""

# 11657

import sys
input = sys.stdin.readline






# https://www.acmicpc.net/step/26

