
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


# 11657 (벨만 포드)

import sys
input = sys.stdin.readline
inf = sys.maxsize
N, M = map(int, input().split())
connect = [[] for _ in range(N + 1)]
for _ in range(M):
    A, B, C = map(int, input().split())
    connect[A].append([B, C])
dist = [inf] * (N + 1)
dist[1] = 0

def bellmanFord():
    for i in range(N):
        for j in range(1, N + 1):
            for to, weight in connect[j]:
                if dist[j] != inf and dist[to] > dist[j] + weight:
                    dist[to] = dist[j] + weight
                    # N개의 노드가 있을 때, a에서 b까지 가는 최소값이
                    # a에서 바로 b로 일 수도 있지만 ( = 1 )
                    # 모든 노드를 거치고 b로 일 수도 있기 때문에 ( = N - 1 )
                    # N - 1 번 반복해야 함
                    if i == N - 1:  # 그리고 N번째는 -inf로 가는 거 있나 확인용
                        return False
    return True

res = bellmanFord()
if not res: print(-1)
else:
    for i in dist[2:]:
        print(i if i != inf else -1)


# 11404 (플로이드 워셜)

import sys
input = sys.stdin.readline
inf = sys.maxsize
n = int(input())
m = int(input())
dist = [[inf for _ in range(n + 1)] for _ in range(n + 1)]
for i in range(1, n + 1): dist[i][i] = 0
connect = [[] for _ in range(n + 1)]
for _ in range(m):
    a, b, c = map(int, input().split())
    connect[a].append([b, c])
    dist[a][b] = min(dist[a][b], c)  # 두 도시를 연결하는 노선은 하나가 아닐 수 있음
for i in range(1, n + 1):
    for j in range(1, n + 1):
        for k in range(1, n + 1):
            dist[j][k] = min(dist[j][k], dist[j][i] + dist[i][k])
for i in range(1, n + 1):
    for j in range(1, n + 1):
        print(dist[i][j] if dist[i][j] != inf else 0, end=" ")
    print()


# 1956

import sys
import heapq
inf = sys.maxsize
input = sys.stdin.readline
V, E = map(int, input().split())
connect = [[] for _ in range(V + 1)]
for _ in range(E):
    a, b, c = map(int, input().split())
    connect[a].append([b, c])
result = inf
for i in range(1, V + 1):
    distance = [inf] * (V + 1)
    heap = []
    for next, dist in connect[i]:
        distance[next] = dist
        heapq.heappush(heap, [distance[next], next])
    while len(heap) > 0:
        dist, now = heapq.heappop(heap)
        if now == i: break  # 목적지 도달
        elif distance[now] < dist: continue  # 기존 게 더 짧음
        else:
            for n, n_dist in connect[now]:
                cost = dist + n_dist
                if cost < distance[n]:
                    distance[n] = cost
                    heapq.heappush(heap, [cost, n])
    result = min(distance[i], result)
print(result if result != inf else -1)
"""

# -----------------------------------
# thirty (투 포인터)


# 3273

import sys
input = sys.stdin.readline
n = int(input())
a = list(map(int, input().split()))
x = int(input())
a.sort()
ans = 0
for i in range(n - 1):
    for j in range(i + 1, n):
        if a[i] + a[j] == x:
            ans += 1
print(ans)




# https://www.acmicpc.net/step/59

