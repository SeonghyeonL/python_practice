
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
start = 0
end = n - 1
while start < end:
    temp = a[start] + a[end]
    if temp == x:
        ans += 1
        start += 1
        end -= 1
    elif temp > x:
        end -= 1
    elif temp < x:
        start += 1
print(ans)


# 2470

import sys
input = sys.stdin.readline
N = int(input())
num = list(map(int, input().split()))
num.sort()
start = 0
end = N - 1
ans = [num[start], num[end]]
while start < end:
    temp = num[start] + num[end]
    if temp == 0:
        ans = [num[start], num[end]]
        break
    else:
        if abs(temp) < abs(ans[1] + ans[0]):
            ans = [num[start], num[end]]
        if temp > 0:
            end -= 1
        elif temp < 0:
            start += 1
print(ans[0], ans[1])


# 1806

import sys
input = sys.stdin.readline
inf = sys.maxsize
N, S = map(int, input().split())
array = list(map(int, input().split()))
sum = array[0]
ans = inf
start, end = 0, 0
while True:
    if sum >= S:
        sum -= array[start]
        ans = min(ans, end - start + 1)
        start += 1
    else:  # sum < S
        end += 1
        if end == N: break
        sum += array[end]
print(ans if ans != inf else 0)


# 1644

import sys
input = sys.stdin.readline
N = int(input())

prime = []
array = [True] * (N + 1)
array[0] = False
array[1] = False
for i in range(2, N + 1):
    if array[i] == True:
        prime.append(i)
        for j in range(2 * i, N + 1, i):
            array[j] = False

if len(prime) == 0:
    print(0)
    exit()

ans = 0
start, end = 0, 0
sum = prime[start]
while True:
    if sum == N:
        ans += 1
        end += 1
        if end == len(prime): break
        sum += prime[end]
    elif sum > N:
        sum -= prime[start]
        start += 1
    else:  # sum < N
        end += 1
        if end == len(prime): break
        sum += prime[end]
print(ans)


# 1450 (meet in the middle)

# 한 번에 연산하기 어려운 문제를 둘로 나누어 시간 단축
import sys
input = sys.stdin.readline
N, C = map(int, input().split())
w = list(map(int, input().split()))
aw = w[:N//2]
bw = w[N//2:]
asum = []
bsum = []

# 부분 집합의 합
def bruteforce(w_arr, sum_arr, idx, w_now):
    if idx == len(w_arr):
        sum_arr.append(w_now)
        return
    bruteforce(w_arr, sum_arr, idx + 1, w_now)
    bruteforce(w_arr, sum_arr, idx + 1, w_now + w_arr[idx])

bruteforce(aw, asum, 0, 0)
bruteforce(bw, bsum, 0, 0)
bsum.sort()  # binary search를 위해 오름차순 정렬

ans = 0
for i in asum:
    if i > C: continue
    elif i == C: ans += 1
    else:  # i < C
        start, end = 0, len(bsum)
        while start < end:  # binary search
            mid = (start + end) // 2
            if bsum[mid] + i <= C:
                start = mid + 1
            else:  # bsum[mid] + i > C
                end = mid
        ans += end
print(ans)


# -----------------------------------
# thirtyone (동적 계획법과 최단거리 역추적)


# 12852

import sys
inf = sys.maxsize
input = sys.stdin.readline
N = int(input())
find = [[inf, inf] for _ in range(N + 1)]
find[1] = [0, 0]

for i in range(2, N + 1):
    find[i][0] = find[i - 1][0] + 1
    find[i][1] = i - 1
    if i % 3 == 0 and find[i // 3][0] < find[i][0]:
        find[i][0] = find[i // 3][0] + 1
        find[i][1] = i // 3
    if i % 2 == 0 and find[i // 2][0] < find[i][0]:
        find[i][0] = find[i // 2][0] + 1
        find[i][1] = i // 2

print(find[N][0])
print(N, end=" ")
temp = find[N][1]
while temp > 0:
    print(temp, end=" ")
    temp = find[temp][1]


# 14002

import sys
input = sys.stdin.readline
N = int(input())
A = list(map(int, input().split()))
find = [[0, -1] for _ in range(N)]
find[0] = [1, -1]
maxlen = 1
maxidx = 0
for i in range(1, N):
    for j in range(i, -1, -1):
        if A[i] > A[j] and find[i][0] <= find[j][0]:
            find[i][0] = find[j][0] + 1
            find[i][1] = j
    if find[i][0] == 0: find[i][0] = 1  # 제일 작음
    if find[i][0] > maxlen:
        maxlen = find[i][0]
        maxidx = i
print(maxlen)
ans = []
ans.append(A[maxidx])
temp = find[maxidx][1]
while temp >= 0:
    ans.append(A[temp])
    temp = find[temp][1]
ans.reverse()
print(' '.join(map(str, ans)))


# 14003

import sys
input = sys.stdin.readline
N = int(input())
A = list(map(int, input().split()))

def binary_search(num):
    start = 0
    end = len(ans) - 1
    # 탈출 조건 및 리턴값 주의할 것
    while start + 1 <= end:
        mid = (start + end) // 2
        if num <= ans[mid]: end = mid
        else: start = mid + 1  # num > ans[mid]
    return end

ans = [A[0]]
ans_total = [(A[0], 0)]

for i in range(1, N):
    temp = A[i]
    if temp > ans[-1]:
        ans_total.append((temp, len(ans)))
        ans.append(temp)
    else:  # temp <= ans[-1]
        idx = binary_search(temp)
        ans[idx] = temp
        ans_total.append((temp, idx))

real_ans = []
idx = len(ans) - 1
# 뒤쪽을 기준으로 최대 길이 만족시키는 숫자들
for i in range(N - 1, -1, -1):
    if ans_total[i][1] == idx:
        real_ans.append(ans_total[i][0])
        idx -= 1
        if idx == -1: break
real_ans.reverse()

print(len(ans))
print(' '.join(map(str, real_ans)))


# 9252

import sys
input = sys.stdin.readline
A = input().strip()
B = input().strip()
LCS = []
for _ in range(len(B) + 1):
    temp = []
    for _ in range(len(A) + 1): temp.append([0, 0, 0])
    LCS.append(temp)
for i in range(len(B)):
    for j in range(len(A)):
        if B[i] == A[j]:
            LCS[i + 1][j + 1][0] = LCS[i][j][0] + 1
            LCS[i + 1][j + 1][1] = i
            LCS[i + 1][j + 1][2] = j
        else:  # B[i] != A[j]
            if LCS[i][j+1][0] >= LCS[i+1][j][0]:
                LCS[i + 1][j + 1][0] = LCS[i][j + 1][0]
                LCS[i + 1][j + 1][1] = i
                LCS[i + 1][j + 1][2] = j + 1
            else:  # LCS[i][j+1][0] < LCS[i+1][j][0]
                LCS[i + 1][j + 1][0] = LCS[i + 1][j][0]
                LCS[i + 1][j + 1][1] = i + 1
                LCS[i + 1][j + 1][2] = j
print(LCS[len(B)][len(A)][0])
if LCS[len(B)][len(A)][0] > 0:
    ans = []
    nowcnt = LCS[len(B)][len(A)][0]
    I = len(B)
    J = len(A)
    while nowcnt > 0:
        temp = LCS[I][J]
        if temp[1] == I - 1 and temp[2] == J - 1:
            ans.append(B[I - 1])
        I = temp[1]
        J = temp[2]
        nowcnt = LCS[I][J][0]
    ans.reverse()
    print(''.join(map(str, ans)))


# 2618

import sys
sys.setrecursionlimit(10 ** 6)
input = sys.stdin.readline
N = int(input())
W = int(input())
dp = [[-1] * (W + 2) for _ in range(W + 2)]  # 앞으로의 이동거리 중 최솟값
task = []
for _ in range(W): task.append(list(map(int, input().split())))
task1 = [[1, 1]] + task  # 경찰차1 위치
task2 = [[N, N]] + task  # 경찰차2 위치
ans = []

def solve(a, b):
    if max(a, b) == W: dp[a][b] = 0
    if dp[a][b] == -1:
        nextnum = max(a, b) + 1
        A = solve(nextnum, b) + abs(task1[a][0] - task1[nextnum][0]) \
            + abs(task1[a][1] - task1[nextnum][1])
        B = solve(a, nextnum) + abs(task2[b][0] - task2[nextnum][0]) \
            + abs(task2[b][1] - task2[nextnum][1])
        dp[a][b] = min(A, B)
    return dp[a][b]

def path(a, b):
    if max(a, b) == W: return 0
    nextnum = max(a, b) + 1
    A = dp[nextnum][b] + abs(task1[a][0] - task1[nextnum][0]) \
        + abs(task1[a][1] - task1[nextnum][1])
    B = dp[a][nextnum] + abs(task2[b][0] - task2[nextnum][0]) \
        + abs(task2[b][1] - task2[nextnum][1])
    if A <= B:
        print(1)
        path(nextnum, b)
    else:
        print(2)
        path(a, nextnum)

print(solve(0, 0))
path(0, 0)


# 13913

import sys
from collections import deque
inf = sys.maxsize
input = sys.stdin.readline
N, K = map(int, input().split())
maxi = max(N, K) + 1
dp = [[inf, -1] for _ in range(maxi + 1)]  # time, last
dp[N] = [0, -1]
move = [[2, 0], [1, 1], [1, -1]]

def bfs():
    q = deque([N])
    while len(q) > 0:
        x = q.popleft()
        if x == K:
            print(dp[x][0])
            path(x)
        else:
            for m in move:
                now = x * m[0] + m[1]
                if 0 <= now <= maxi and dp[now][0] > dp[x][0] + 1:
                    q.append(now)
                    dp[now][0] = dp[x][0] + 1
                    dp[now][1] = x

def path(i):
    ans = []
    temp = i
    while True:
        ans.append(temp)
        if temp == N: break
        temp = dp[temp][1]
    ans.reverse()
    print(' '.join(map(str, ans)))

bfs()


# 9019

import sys
from collections import deque
input = sys.stdin.readline
T = int(input())
for _ in range(T):
    A, B = map(int, input().split())
    # D -> double (단, 10000으로 나눈 나머지)
    # S -> n-1 (단, 0의 경우 9999로 대체)
    # L -> 자릿수를 왼쪽으로 회전
    # R -> 자릿수를 오른쪽으로 회전
    q = deque([[A, ""]])
    visit = [False] * 10000

    while len(q) > 0:
        num, path = q.popleft()
        visit[num] = True
        if num == B:
            print(path)
            break

        num_D = (2 * num) % 10000
        if visit[num_D] == False:
            q.append([num_D, path + "D"])
            visit[num_D] = True
        num_S = num - 1 if num > 0 else 9999
        if visit[num_S] == False:
            q.append([num_S, path + "S"])
            visit[num_S] = True
        num_L = (10 * num + (num // 1000)) % 10000
        if visit[num_L] == False:
            q.append([num_L, path + "L"])
            visit[num_L] = True
        num_R = ((num // 10) + 1000 * (num % 10)) % 10000
        if visit[num_R] == False:
            q.append([num_R, path + "R"])
            visit[num_R] = True


# 11779

import sys
from collections import deque
inf = sys.maxsize
input = sys.stdin.readline
n = int(input())  # 도시의 개수
m = int(input())  # 버스의 개수
bus = [[] for _ in range(n + 1)]
for _ in range(m):
    a, b, c = map(int, input().split())  # 출발지, 도착지, 비용
    bus[a].append([b, c])
A, B = map(int, input().split())  # 출발점, 도착점
dp = [(inf, 0) for _ in range(n + 1)]  # 비용, 이전 도시
dp[A] = (1, 0)

def bfs():
    q = deque([(A, 0)])
    while len(q) > 0:
        x, nowcost = q.popleft()
        if dp[x][0] <= nowcost: continue  # 시간 초과 방지 (찾았던 값이 더 작아서 지금 거 버리기)
        if x != B:
            for to, cost in bus[x]:  # x에서 출발하는 버스 중
                if dp[to][0] > dp[x][0] + cost:  # x를 거쳐서 가는 게 비용이 적으면
                    q.append((to, nowcost + cost))
                    dp[to] = (dp[x][0] + cost, x)

bfs()

print(dp[B][0] - 1)  # 최소 비용 (-1 ; A initial cost)

ans = []
cnt = 0
temp = B
while True:
    ans.append(temp)
    cnt += 1
    if temp == A: break
    temp = dp[temp][1]

print(cnt)  # 최소 비용 경로의 도시 개수 (출발, 도착 도시 포함)
ans.reverse()
print(' '.join(map(str, ans)))  # 최소 비용 방문 도시 순서대로


# 11780

import sys
inf = sys.maxsize
input = sys.stdin.readline
n = int(input())
m = int(input())

dp = [[inf] * (n + 1) for _ in range(n + 1)]  # i에서 j까지 가는 최소 비용
for i in range(1, n + 1): dp[i][i] = 0  # 시작 = 도착: 0 출력
for _ in range(m):
    a, b, c = map(int, input().split())  # 출발지, 도착지, 비용
    dp[a][b] = min(dp[a][b], c)

trace = [[0] * (n + 1) for _ in range(n + 1)]  # i에서 j로 갈 때 거치는 곳
for k in range(1, n + 1):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if dp[i][j] > dp[i][k] + dp[k][j]:
                dp[i][j] = dp[i][k] + dp[k][j]
                trace[i][j] = k

for i in range(1, n + 1):
    for j in range(1, n + 1):
        print(dp[i][j] if dp[i][j] != inf else 0, end=' ')
    print()

def find_trace(x, y):
    z = trace[x][y]
    if z == 0: return []
    else: return find_trace(x, z) + [z] + find_trace(z, y)

# 시작 = 도착 or 도시 i에서 j로 갈 수 없음 → "0을 출력"
for i in range(1, n + 1):
    for j in range(1, n + 1):
        if dp[i][j] == 0 or dp[i][j] == inf:  # 시작 = 도착 or 갈 수 없음
            print(0)  # print(len(path)) ; path is empty
        else:
            path = [i] + find_trace(i, j) + [j]
            print(len(path), end=' ')
            print(' '.join(map(str, path)))


# -----------------------------------
# thirtytwo (트리)


# 11725

import sys
from collections import deque
input = sys.stdin.readline
N = int(input())
connect = [[] for _ in range(N + 1)]
for _ in range(N - 1):
    a, b = map(int, input().split())
    connect[a].append(b)
    connect[b].append(a)
q = deque([1])  # 루트가 1
ans = [0] * (N + 1)  # 부모 노드 초기화

# bfs
while len(q) > 0:
    now = q.popleft()
    for next in connect[now]:
        if ans[next] == 0:
            ans[next] = now
            q.append(next)

for i in range(2, N + 1): print(ans[i])


# 1167

import sys
from collections import deque
input = sys.stdin.readline
V = int(input())
connect = [[] for _ in range(V + 1)]
for v in range(V):
    lst = list(map(int, input().split()))  # 번호, x번과 y거리 * a, -1
    for i in range(1, len(lst) - 2, 2):
        connect[lst[0]].append((lst[i], lst[i + 1]))  # 정점 순서대로 들어오는 거 아님 주의

def bfs(start):
    visit = [-1] * (V + 1)
    q = deque([start])
    visit[start] = 0
    maxi = [0, 0]  # 가장 먼 노드와 거리
    while len(q) > 0:
        temp = q.popleft()
        for node, length in connect[temp]:
            if visit[node] == -1:
                visit[node] = visit[temp] + length
                q.append(node)
                if maxi[1] < visit[node]:
                    maxi = [node, visit[node]]
    return maxi

node1, length1 = bfs(1)
_, ans = bfs(node1)
print(ans)


# 1967

import sys
from collections import deque
input = sys.stdin.readline
n = int(input())
connect = [[] for _ in range(n + 1)]
for _ in range(n - 1):
    a, b, c = map(int, input().split())  # 부모, 자식, 가중치
    connect[a].append((b, c))
    connect[b].append((a, c))

def bfs(start):
    visit = [-1] * (n + 1)
    q = deque([start])
    visit[start] = 0
    maxi = [0, 0]  # 가장 먼 노드와 거리
    while len(q) > 0:
        temp = q.popleft()
        for node, length in connect[temp]:
            if visit[node] == -1:
                visit[node] = visit[temp] + length
                q.append(node)
                if maxi[1] < visit[node]:
                    maxi = [node, visit[node]]
    return maxi

node1, length1 = bfs(1)
_, ans = bfs(node1)
print(ans)


# 1991

import sys
input = sys.stdin.readline
Anum = ord('A')
N = int(input())
connect = [(-1, -1) for _ in range(N)]
for _ in range(N):
    line = input().strip()
    a = ord(line[0]) - Anum
    b = ord(line[2]) - Anum if line[2] != '.' else -1
    c = ord(line[4]) - Anum if line[4] != '.' else -1
    connect[a] = (b, c)

def preorder(i):
    left, right = connect[i]
    print(chr(i + Anum), end="")
    if left != -1: preorder(left)
    if right != -1: preorder(right)

def inorder(i):
    left, right = connect[i]
    if left != -1: inorder(left)
    print(chr(i + Anum), end="")
    if right != -1: inorder(right)

def postorder(i):
    left, right = connect[i]
    if left != -1: postorder(left)
    if right != -1: postorder(right)
    print(chr(i + Anum), end="")

preorder(0)
print()
inorder(0)
print()
postorder(0)


# 2263

import sys
sys.setrecursionlimit(10 ** 6)
input = sys.stdin.readline
n = int(input())
inorder = list(map(int, input().split()))
postorder = list(map(int, input().split()))

nodenum = [0] * (n + 1)
for i in range(n): nodenum[inorder[i]] = i  # 시간초과 방지

def preorder(i_start, i_end, p_start, p_end):
    if i_start <= i_end and p_start <= p_end:
        root = postorder[p_end]
        print(root, end=" ")
        idx = nodenum[root]
        lefttreesize = idx - i_start
        preorder(i_start, idx - 1, p_start, p_start + lefttreesize - 1)
        preorder(idx + 1, i_end, p_start + lefttreesize, p_end - 1)

preorder(0, n - 1, 0, n - 1)


# 5639

import sys
sys.setrecursionlimit(10 ** 6)
input = sys.stdin.readline
preorder = []

while True:
    try:
        preorder.append(int(input()))
    except:
        break

def postorder(start, end):
    if start <= end:
        root = preorder[start]
        idx = end
        for i in range(start + 1, end + 1):
            if root < preorder[i]:
                idx = i - 1
                break
        postorder(start + 1, idx)
        postorder(idx + 1, end)
        print(root)

postorder(0, len(preorder) - 1)


# 4803 (유니온 파인드)

import sys
input = sys.stdin.readline

def find(x):  # 루트 찾기
    if x != connect[x]:  # 연결된 게 있음
        connect[x] = find(connect[x])
    return connect[x]

def union(x, y):
    x = find(x)  # 루트 노드
    y = find(y)  # 루트 노드
    connect[x] = y  # 연결시키기

case = 0
while True:
    case += 1  # 출력 위함
    n, m = map(int, input().split())
    if n == 0 and m == 0: break
    connect = [i for i in range(n + 1)]
    cycle = set()  # 중복 방지
    for _ in range(m):
        a, b = map(int, input().split())
        if find(a) == find(b):  # 두 정점이 이어짐으로써 사이클 생김
            cycle.add(connect[a])
            cycle.add(connect[b])
        elif connect[a] in cycle or connect[b] in cycle:  # 두 정점 중 하나라도 사이클에 포함
            cycle.add(connect[a])
            cycle.add(connect[b])
        union(a, b)

    for i in range(n + 1): find(i)  # 루트 갱신

    connect = set(connect)  # 중복 제거
    treecnt = sum([1 if i not in cycle else 0 for i in connect]) - 1

    if treecnt == 0: print("Case %d: No trees." % case)
    elif treecnt == 1: print("Case %d: There is one tree." % case)
    else: print("Case %d: A forest of %d trees." % (case, treecnt))


# -----------------------------------
# thirtythree (유니온 파인드)


# 1717

import sys
sys.setrecursionlimit(10 ** 6)
input = sys.stdin.readline
n, m = map(int, input().split())
connect = [i for i in range(n + 1)]

def find(x):  # 루트 찾기
    if x != connect[x]:  # 연결된 게 있음
        connect[x] = find(connect[x])
    return connect[x]

def union(x, y):
    x = find(x)  # 루트 노드
    y = find(y)  # 루트 노드
    if x == y: return
    elif x < y: connect[y] = x  # 연결시키기
    else: connect[x] = y  # 연결시키기
    # 루트 갱신 생략

for _ in range(m):
    x, a, b = map(int, input().split())

    if x == 0:
        union(a, b)

    elif x == 1:
        if find(a) == find(b): print("YES")
        else: print("NO")


# 1976

import sys
input = sys.stdin.readline
N = int(input())
M = int(input())
connect = [i for i in range(N + 1)]

def find(x):  # 루트 찾기
    if x != connect[x]:  # 연결된 게 있음
        connect[x] = find(connect[x])
    return connect[x]

def union(x, y):
    x = find(x)  # 루트 노드
    y = find(y)  # 루트 노드
    if x == y: return
    elif x < y: connect[y] = x  # 연결시키기
    else: connect[x] = y  # 연결시키기

for n in range(N):
    ns = list(map(int, input().split()))
    for i in range(N):
        if n < i and ns[i] == 1:
            union(n+1, i+1)

travel = list(map(int, input().split()))
if M == 1:
    print("YES")
    exit(0)
for m in range(1, M):
    if find(travel[m-1]) != find(travel[m]):
        print("NO")
        exit(0)
print("YES")


# 4195

import sys
input = sys.stdin.readline
T = int(input())

def find(x):  # 루트 찾기
    if x != root[x]:  # 연결된 게 있음
        root[x] = find(root[x])
    return root[x]


def union(x, y):
    x = find(x)  # 루트 노드
    y = find(y)  # 루트 노드
    if x != y:
        root[x] = root[y]  # x의 루트를 y의 루트로
        cnt[y] += cnt[x]  # y의 집합 개수 증가

for _ in range(T):
    F = int(input())
    root = dict()
    cnt = dict()

    for _ in range(F):
        a, b = input().strip().split()
        if a not in root:
            root[a] = a
            cnt[a] = 1
        if b not in root:
            root[b] = b
            cnt[b] = 1
        union(a, b)
        print(cnt[find(b)])


# 20040

import sys
input = sys.stdin.readline
n, m = map(int, input().split())
connect = [i for i in range(n)]

def find(x):  # 루트 찾기
    if x != connect[x]:  # 연결된 게 있음
        connect[x] = find(connect[x])
    return connect[x]


def union(x, y):
    x = find(x)  # 루트 노드
    y = find(y)  # 루트 노드
    if x == y:
        return
    elif x < y:
        connect[y] = x
    else:  # x > y
        connect[x] = y

for i in range(1, m + 1):
    a, b = map(int, input().split())
    if find(a) == find(b):
        print(i)
        exit()
    union(a, b)

print(0)  # m번 후에도 종료 안 된 경우


# -----------------------------------
# thirtyfour (최소 신장 트리)


# 9372 (N - 1)

import sys
input = sys.stdin.readline
T = int(input())

def dfs(x, cnt):
    visit[x] = True
    for i in connect[x]:
        if visit[i] == 0:
            cnt = dfs(i, cnt + 1)
    return cnt

for _ in range(T):
    N, M = map(int, input().split())
    connect = [[] for _ in range(N + 1)]
    for _ in range(M):
        a, b = map(int, input().split())
        connect[a].append(b)
        connect[b].append(a)
    visit = [False] * (N + 1)
    res = dfs(1, 0)
    print(res)


# 1197 (Kruskal Algorithm)

import sys
input = sys.stdin.readline
V, E = map(int, input().split())

edge = []
for _ in range(E):
    A, B, C = map(int, input().split())
    edge.append((A, B, C))
edge.sort(key=lambda x: x[2])  # C를 기준으로 정렬

parent = [i for i in range(V + 1)]

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    a = find(a)
    b = find(b)
    if a == b: return
    elif a < b: parent[b] = a
    else: parent[a] = b

answer = 0
for a, b, c in edge:
    p_a = find(a)
    p_b = find(b)
    if p_a != p_b:  # 사이클을 발생시키지 않을 때만 추가
        union(a, b)
        answer += c
print(answer)


# 4386

import sys
input = sys.stdin.readline
n = int(input())
star = []
for _ in range(n):
    x, y = map(float, input().split())
    star.append((x, y))
edge = []
for i in range(n):
    for j in range(i+1, n):
        dist = ((star[i][0] - star[j][0]) ** 2 + (star[i][1] - star[j][1]) ** 2) ** 0.5
        edge.append((i, j, dist))
edge.sort(key=lambda x: x[2])  # 거리 기준 정렬

parent = [i for i in range(n)]

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    a = find(a)
    b = find(b)
    if a == b: return
    elif a < b: parent[b] = a
    else: parent[a] = b

answer = 0
for a, b, c in edge:
    p_a = find(a)
    p_b = find(b)
    if p_a != p_b:  # 사이클을 발생시키지 않을 때만 추가
        union(a, b)
        answer += c
print(answer)


# 1774

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
star = []
for _ in range(N):
    X, Y = map(int, input().split())
    star.append((X, Y))
edge = []
for i in range(N):
    for j in range(i + 1, N):
        dist = ((star[i][0] - star[j][0]) ** 2 + (star[i][1] - star[j][1]) ** 2) ** 0.5
        edge.append((i + 1, j + 1, dist))
edge.sort(key=lambda x: x[2])  # 거리 기준 정렬

parent = [i for i in range(N + 1)]

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    a = find(a)
    b = find(b)
    if a == b: return
    elif a < b: parent[b] = a
    else: parent[a] = b

for _ in range(M):
    a, b = map(int, input().split())
    union(a, b)

answer = 0
for a, b, c in edge:
    p_a = find(a)
    p_b = find(b)
    if p_a != p_b:  # 사이클을 발생시키지 않을 때만 추가
        union(a, b)
        answer += c
print("{:.2f}".format(answer))


# 6497

import sys
input = sys.stdin.readline

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    a = find(a)
    b = find(b)
    if a == b: return
    elif a < b: parent[b] = a
    else: parent[a] = b

while True:
    m, n = map(int, input().split())
    if m == 0 and n == 0: break
    edge = []
    parent = [i for i in range(m)]
    total = 0
    for _ in range(n):
        x, y, z = map(int, input().split())
        total += z
        edge.append((x, y, z))
    edge.sort(key=lambda x: x[2])  # 거리 기준 정렬
    answer = 0
    for a, b, c in edge:
        p_a = find(a)
        p_b = find(b)
        if p_a != p_b:  # 사이클을 발생시키지 않을 때만 추가
            union(a, b)
            answer += c
    print(total - answer)


# 17472

import sys
from collections import deque
input = sys.stdin.readline
N, M = map(int, input().split())
map = [list(map(int, input().split())) for _ in range(N)]  # 0은 바다, 1은 땅
visited = [[False] * M for _ in range(N)]
move = [[0, 1], [0, -1], [1, 0], [-1, 0]]

def mark_island(y, x, m):
    q = deque([])
    q.append((y, x))
    map[y][x] = m
    visited[y][x] = True
    while len(q) > 0:
        b, a = q.popleft()
        for my, mx in move:
            ny, nx = my + b, mx + a
            if 0 <= ny < N and 0 <= nx < M and map[ny][nx] == 1 and visited[ny][nx] == False:
                map[ny][nx] = m
                visited[ny][nx] = True
                q.append((ny, nx))

mark = 1
for i in range(N):
    for j in range(M):
        if map[i][j] == 1 and visited[i][j] == False:
            mark_island(i, j, mark)
            mark += 1
mark -= 1

bridge = []  # 가능한 모든 다리

def find_bridge(y, x, now):
    q = deque([])
    for moving_idx in range(4):
        q.append((y, x, 0, move[moving_idx]))
    while len(q) > 0:
        b, a, cnt, direction = q.popleft()
        ny, nx = b + direction[0], a + direction[1]
        if map[b][a] not in (0, now):  # 도착했었음
            if cnt > 2:  # cnt에는 섬 도달도 포함되어 있음
                bridge.append((cnt - 1, now, map[b][a]))
        elif 0 <= ny < N and 0 <= nx < M and map[ny][nx] != now:  # 이동 가능 (도착 포함)
            q.append((ny, nx, cnt + 1, direction))

for i in range(N):
    for j in range(M):
        if map[i][j] != 0:
            find_bridge(i, j, map[i][j])

bridge.sort()  # 거리 기준 정렬
parent = [i for i in range(mark + 1)]

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    a = find(a)
    b = find(b)
    if a == b: return
    elif a < b: parent[b] = a
    else: parent[a] = b

island = 1
length = 0
for cost, a, b in bridge:
    if find(a) != find(b):
        island += 1
        union(a, b)
        length += cost

if length == 0 or island != mark: print(-1)
else: print(length)


# -----------------------------------
# thirtyfive (트리에서의 동적 계획법)


# 15681

import sys
sys.setrecursionlimit(10 ** 6)
input = sys.stdin.readline
N, R, Q = map(int, input().split())
connect = [[] for i in range(N + 1)]
for _ in range(N - 1):
    U, V = map(int, input().split())
    connect[U].append(V)
    connect[V].append(U)

visited = [False] * (N + 1)
child = [0] * (N + 1)

def find_child(x):
    visited[x] = True
    sum = 1
    for i in connect[x]:
        if visited[i] == False:
            find_child(i)
        sum += child[i]
    child[x] = sum

find_child(R)

for _ in range(Q):
    U = int(input())
    print(child[U])


# 2213

import sys
sys.setrecursionlimit(10 ** 6)
input = sys.stdin.readline
n = int(input())
w = [0] + list(map(int, input().split()))
connect = [[] for i in range(n + 1)]
for _ in range(n - 1):
    a, b = map(int, input().split())
    connect[a].append(b)
    connect[b].append(a)

visited = [False] * (n + 1)
dp = [[0, 0] for _ in range(n + 1)]
path = [[[], []] for _ in range(n + 1)]

def dfs(x):
    visited[x] = True
    #dp[x][0] = 0  # 포함 안 함
    dp[x][1] = w[x]  # 포함
    path[x][1].append(x)
    for i in connect[x]:
        if visited[i] == False:
            result = dfs(i)
            dp[x][0] += max(dp[i][0], dp[i][1])  # 현재를 불포함 -> 자식 포함/불포함
            if dp[i][0] >= dp[i][1]:  # 자식 불포함이 더 큰 경우
                path[x][0] += result[0]
            else:  # dp[i][0] < dp[i][1] → 자식 포함이 더 큰 경우
                path[x][0] += result[1]
            dp[x][1] += dp[i][0]  # 현재를 포함 -> 자식은 포함 불가
            path[x][1] += result[0]
    return path[x]

trace = dfs(1)

if dp[1][0] >= dp[1][1]:
    print(dp[1][0])
    trace[0].sort()
    print(' '.join(map(str, trace[0])))
else:
    print(dp[1][1])
    trace[1].sort()
    print(' '.join(map(str, trace[1])))


# 2533

import sys
sys.setrecursionlimit(10 ** 6)
input = sys.stdin.readline
N = int(input())
connect = [[] for _ in range(N + 1)]
for _ in range(N - 1):
    u, v = map(int, input().split())
    connect[u].append(v)
    connect[v].append(u)

visited = [False] * (N + 1)
dp = [[0, 0] for _ in range(N + 1)]

def dfs(x):
    visited[x] = True
    #dp[x][0] = 0  # 자신 포함 안 함 (default = 0)
    dp[x][1] = 1  # 자신 포함
    for i in connect[x]:
        if visited[i] == False:
            dfs(i)
            dp[x][0] += dp[i][1]  # 현재를 불포함 -> 자식 포함
            dp[x][1] += min(dp[i][0], dp[i][1])  # 현재를 포함 -> 자식 포함/불포함

dfs(1)
print(min(dp[1][0], dp[1][1]))


# 1949

import sys
sys.setrecursionlimit(10 ** 6)
input = sys.stdin.readline
N = int(input())
r = [0] + list(map(int, input().split()))
connect = [[] for _ in range(N + 1)]
for _ in range(N - 1):
    u, v = map(int, input().split())
    connect[u].append(v)
    connect[v].append(u)

visited = [False] * (N + 1)
dp = [[0, 0] for _ in range(N + 1)]

def dfs(x):
    visited[x] = True
    #dp[x][0] = 0  # 자신 포함 안 함 (default = 0)
    dp[x][1] = r[x]  # 자신 포함
    for i in connect[x]:
        if visited[i] == False:
            dfs(i)
            dp[x][0] += max(dp[i][0], dp[i][1])  # 현재를 불포함 -> 자식 포함
            dp[x][1] += dp[i][0]  # 현재를 포함 -> 자식 불포함

dfs(1)
print(max(dp[1][0], dp[1][1]))


# -----------------------------------
# thirtysix (기하 2)


# 2166

import sys
input = sys.stdin.readline
N = int(input())
xy = []
for _ in range(N):
    x, y = map(int, input().split())
    xy.append((x, y))
xy.append((xy[0][0], xy[0][1]))
ans = 0
for i in range(N):
    ans += xy[i][0] * xy[i + 1][1]
    ans -= xy[i][1] * xy[i + 1][0]
ans /= 2
print('{:.1f}'.format(abs(ans)))


# 11758

import sys
input = sys.stdin.readline
P = []
P.append(list(map(int, input().split())))
P.append(list(map(int, input().split())))
P.append(list(map(int, input().split())))
P.append([P[0][0], P[0][1]])
# 1: 반시계 / -1: 시계 / 0: 일직선
# 넓이 양수 / 음수 / 0
ans = 0
for i in range(3):
    ans += P[i][0] * P[i + 1][1]
    ans -= P[i][1] * P[i + 1][0]
if ans > 0: print(1)
elif ans < 0: print(-1)
else: print(0)


# 25308

import sys
from itertools import permutations
input = sys.stdin.readline
a = list(map(int, input().split()))
A = list(permutations(a, 8))
ans = 0

def fun(x, y):
    return (2 ** 0.5) * x * y / (x + y)

for i in A:
    if i[0] < fun(i[7], i[1]): continue
    okay = True
    for idx in range(1, 7):  # 1 ~ 6
        if i[idx] < fun(i[idx - 1], i[idx + 1]):
            okay = False
            break
    if okay == False: continue
    if i[7] < fun(i[0], i[6]): continue
    ans += 1

print(ans)


# 17386

import sys
input = sys.stdin.readline
x1, y1, x2, y2 = map(int, input().split())  # L1
x3, y3, x4, y4 = map(int, input().split())  # L2
# L1과 L2가 교차하면 1, 아니면 0을 출력

def ccw(x1, y1, x2, y2, x3, y3):
    ans = x1 * y2 - x2 * y1 + x2 * y3 - x3 * y2 + x3 * y1 - x1 * y3
    if ans > 0: return 1
    elif ans < 0: return -1
    else: return 0

result = 0
if ccw(x1, y1, x2, y2, x3, y3) * ccw(x1, y1, x2, y2, x4, y4) < 0:
    if ccw(x3, y3, x4, y4, x1, y1) * ccw(x3, y3, x4, y4, x2, y2) < 0:
        result = 1

print(result)


# 17387

import sys
input = sys.stdin.readline
x1, y1, x2, y2 = map(int, input().split())  # L1
x3, y3, x4, y4 = map(int, input().split())  # L2
# L1과 L2가 교차하면 1, 아니면 0을 출력

def ccw(x1, y1, x2, y2, x3, y3):
    ans = x1 * y2 - x2 * y1 + x2 * y3 - x3 * y2 + x3 * y1 - x1 * y3
    if ans > 0: return 1
    elif ans < 0: return -1
    else: return 0

result = 0
if ccw(x1, y1, x2, y2, x3, y3) * ccw(x1, y1, x2, y2, x4, y4) < 0 \
        and ccw(x3, y3, x4, y4, x1, y1) * ccw(x3, y3, x4, y4, x2, y2) < 0:
    result = 1
elif ccw(x1, y1, x2, y2, x3, y3) * ccw(x1, y1, x2, y2, x4, y4) <= 0 \
        and ccw(x3, y3, x4, y4, x1, y1) * ccw(x3, y3, x4, y4, x2, y2) <= 0:
    if min(x1, x2) <= max(x3, x4) and max(x1, x2) >= min(x3, x4) \
            and min(y1, y2) <= max(y3, y4) and min(y3, y4) <= max(y1, y2):
        result = 1

print(result)


# 20149

import sys
inf = sys.maxsize
input = sys.stdin.readline
x1, y1, x2, y2 = map(int, input().split())  # L1
x3, y3, x4, y4 = map(int, input().split())  # L2
# L1과 L2가 교차하면 1, 아니면 0을 출력

def ccw(x1, y1, x2, y2, x3, y3):
    ans = x1 * y2 - x2 * y1 + x2 * y3 - x3 * y2 + x3 * y1 - x1 * y3
    if ans > 0: return 1
    elif ans < 0: return -1
    else: return 0

result = 0
if ccw(x1, y1, x2, y2, x3, y3) * ccw(x1, y1, x2, y2, x4, y4) < 0 \
        and ccw(x3, y3, x4, y4, x1, y1) * ccw(x3, y3, x4, y4, x2, y2) < 0:
    result = 1
elif ccw(x1, y1, x2, y2, x3, y3) * ccw(x1, y1, x2, y2, x4, y4) <= 0 \
        and ccw(x3, y3, x4, y4, x1, y1) * ccw(x3, y3, x4, y4, x2, y2) <= 0:
    if min(x1, x2) <= max(x3, x4) and max(x1, x2) >= min(x3, x4) \
            and min(y1, y2) <= max(y3, y4) and min(y3, y4) <= max(y1, y2):
        result = 1

print(result)

def findfunc(x1, y1, x2, y2):
    if x1 == x2: a = inf
    else: a = (y1 - y2) / (x1 - x2)
    b = y1 - a * x1
    return a, b

if result == 1:
    a, b = findfunc(x1, y1, x2, y2)
    c, d = findfunc(x3, y3, x4, y4)
    if c == a:
        if min(x1, x2) == max(x3, x4) and a * min(x1, x2) + b == c * max(x3, x4) + d:
            x, y = min(x1, x2), a * min(x1, x2) + b
        elif max(x1, x2) == min(x3, x4) and a * max(x1, x2) + b == c * min(x3, x4) + d:
            x, y = min(x3, x4), c * min(x3, x4) + d
        else: exit()  # 교점 여러개
    else:
        x = (b - d) / (c - a)
        y = (b * c - a * d) / (c - a)
    print(x, y)


# 2162

import sys
input = sys.stdin.readline
N = int(input())
line = []
for _ in range(N):
    line.append(list(map(int, input().split())))  # x1, y1, x2, y2
connect = [[i, 1] for i in range(N)]

def ccw(x1, y1, x2, y2, x3, y3):
    ans = x1 * y2 - x2 * y1 + x2 * y3 - x3 * y2 + x3 * y1 - x1 * y3
    if ans > 0: return 1
    elif ans < 0: return -1
    else: return 0

def check(x1, y1, x2, y2, x3, y3, x4, y4):
    if ccw(x1, y1, x2, y2, x3, y3) * ccw(x1, y1, x2, y2, x4, y4) < 0 \
            and ccw(x3, y3, x4, y4, x1, y1) * ccw(x3, y3, x4, y4, x2, y2) < 0:
        return True
    elif ccw(x1, y1, x2, y2, x3, y3) * ccw(x1, y1, x2, y2, x4, y4) <= 0 \
            and ccw(x3, y3, x4, y4, x1, y1) * ccw(x3, y3, x4, y4, x2, y2) <= 0:
        if min(x1, x2) <= max(x3, x4) and max(x1, x2) >= min(x3, x4) \
                and min(y1, y2) <= max(y3, y4) and min(y3, y4) <= max(y1, y2):
            return True
    return False

def find(x):  # 루트 찾기
    if x != connect[x][0]:  # 연결된 게 있음
        connect[x][0] = find(connect[x][0])
    return connect[x][0]

def union(x, y):
    x = find(x)  # 루트 노드
    y = find(y)  # 루트 노드
    if x == y: return
    elif x < y:
        connect[y][0] = x  # 연결시키기
        connect[x][1] += connect[y][1]
    else:
        connect[x][0] = y  # 연결시키기
        connect[y][1] += connect[x][1]

for i in range(N - 1):
    for j in range(i + 1, N):
        x1, y1, x2, y2 = line[i]
        x3, y3, x4, y4 = line[j]
        if check(x1, y1, x2, y2, x3, y3, x4, y4):
            union(i, j)

g_cnt = 0
max_cnt = 0
for i in range(N):
    if connect[i][0] == i:
        g_cnt += 1
        max_cnt = max(max_cnt, connect[i][1])
print(g_cnt)
print(max_cnt)


# 7869

import sys
from math import pi, atan2
# atan: 출력 범위가 [-pi/2, pi/2] / atan2: 출력 범위가 [-pi, pi]
input = sys.stdin.readline
x1, y1, r1, x2, y2, r2 = map(float, input().split())
d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
if d >= r1 + r2:
    ans = 0
elif d <= abs(r1 - r2):
    ans = pi * (min(r1, r2) ** 2)
else:
    x = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    y = (r1 ** 2 - x ** 2) ** 0.5
    alpha = atan2(y, x)
    beta = atan2(y, (d - x))
    ans = (r1 ** 2) * alpha - x * y
    ans += (r2 ** 2) * beta - (d - x) * y
print('%.3f' % ans)


# 1069

import sys
input = sys.stdin.readline
X, Y, D, T = map(int, input().split())
# (X, Y) → (0, 0) / T초에 D만큼 일직선으로만 점프
d = (X ** 2 + Y ** 2) ** 0.5
if d >= D:
    ans = min(T * (d // D) + d % D, T * (d // D + 1), d)
else:
    ans = min(T + (D - d), 2 * T, d)
print(ans)
"""

# -----------------------------------
# thirtyseven (동적 계획법 3)


# 11723

import sys
input = sys.stdin.readline

# 0414 - coding test practice using other website





# https://www.acmicpc.net/step/31

