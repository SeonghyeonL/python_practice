
"""
# -----------------------------------
# thirtynine (위상 정렬)


# 2252

import sys
from collections import deque
input = sys.stdin.readline
N, M = map(int, input().split())
connect = [[] for _ in range(N + 1)]
cnt = [0] * (N + 1)
for _ in range(M):
    A, B = map(int, input().split())
    connect[A].append(B)
    cnt[B] += 1
q = deque([])
for i in range(1, N + 1):  # 초기
    if cnt[i] == 0: q.append(i)
while len(q) > 0:
    temp = q.popleft()
    print(temp, end=" ")
    for i in connect[temp]:
        cnt[i] -= 1
        if cnt[i] == 0: q.append(i)


# 3665

import sys
from collections import deque
input = sys.stdin.readline
T = int(input())
for _ in range(T):
    n = int(input())
    t = list(map(int, input().split()))
    m = int(input())
    connect = [[] for _ in range(n + 1)]
    rank = [0] * (n + 1)
    # 순위 변동 전
    for i in range(n):
        for j in range(i):
            connect[t[j]].append(t[i])
        rank[t[i]] = i
    # 순위 변동 중
    for _ in range(m):
        a, b = map(int, input().split())
        if a in connect[b]:  # b의 순위가 더 높았었음
            connect[b].remove(a)
            connect[a].append(b)
            rank[b] += 1
            rank[a] -= 1
        else:  # a의 순위가 더 높았었음
            connect[a].remove(b)
            connect[b].append(a)
            rank[a] += 1
            rank[b] -= 1
    # 순위 변동 후
    q = deque([])
    answer = []
    impossible = False
    for i in range(1, n + 1):
        if rank[i] == 0: q.append(i)
    if len(q) == 0: impossible = True
    while True:
        if len(q) == 0 and len(answer) == n:  # 다 찾음
            break
        if len(q) == 0 and len(answer) < n:  # 원소가 하나도 없음 = 사이클 발생
            impossible = True
            break
        if len(q) > 1:  # 원소가 2개 이상 = 순위 못 정함
            impossible = True
            break
        temp = q.popleft()
        answer.append(temp)
        for i in connect[temp]:
            rank[i] -= 1
            if rank[i] == 0: q.append(i)
    if impossible == True: print('IMPOSSIBLE')
    else: print(' '.join(map(str, answer)))


# 1766

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
connect = [[] for _ in range(N + 1)]
rank = [0] * (N + 1)
for _ in range(M):
    A, B = map(int, input().split())
    connect[A].append(B)
    rank[B] += 1
zero_list = []
for i in range(1, N + 1):
    if rank[i] == 0:
        zero_list.append(i)
while len(zero_list) > 0:
    zero_list.sort()
    temp = zero_list[0]
    zero_list = zero_list[1:]
    print(temp, end=" ")
    rank[temp] = -1
    for i in connect[temp]:
        rank[i] -= 1
        if rank[i] == 0: zero_list.append(i)


# -----------------------------------
# fourty (최소 공통 조상)


# 3584 (LCA)

import sys
input = sys.stdin.readline
T = int(input())
for _ in range(T):
    N = int(input())
    connect = [-1] * (N + 1)
    for _ in range(N - 1):
        A, B = map(int, input().split())
        connect[B] = A  # B의 부모가 A
    A, B = map(int, input().split())
    parent_A = [A]
    while True:
        if connect[A] == -1:
            break
        else:
            A = connect[A]
            parent_A.append(A)
    while True:
        if B in parent_A:
            print(B)
            break
        B = connect[B]


# 17435 (sparse table)

import sys
input = sys.stdin.readline
m = int(input())
f = [0] + list(map(int, input().split()))
dp = [[f[i]] for i in range(m + 1)]
# 1 ≤ m ≤ 200000 에서 log(m) = 17.609... < 18 이므로 18까지만 해도 됨
for i in range(1, 19):
    for j in range(1, m + 1):
        dp[j].append(dp[dp[j][i - 1]][i - 1])
Q = int(input())
for _ in range(Q):
    n, x = map(int, input().split())
    for i in range(18, -1, -1):
        if n >= 1 << i:  # 2^i보다 크면
            n -= 1 << i  # 2^i을 빼줌
            x = dp[x][i]  # 0이 될때까지 반복
    print(x)


# 11438

import sys
sys.setrecursionlimit(10 ** 6)
input = sys.stdin.readline
log = 17  # N, M ≤ 100000 에서 log(m) = 16.609... < 17
N = int(input())
connect = [[] for _ in range(N + 1)]
parent = [[0] * (log + 1) for _ in range(N + 1)]
visited = [False] * (N + 1)
d = [0] * (N + 1)  # 루트 노드로부터의 깊이

for _ in range(N - 1):
    A, B = map(int, input().split())
    connect[A].append(B)
    connect[B].append(A)

def dfs(x, depth):  # 루트 노드부터 시작해서 깊이 구하기
    visited[x] = True
    d[x] = depth
    for y in connect[x]:
        if visited[y] == False:
            parent[y][0] = x
            dfs(y, depth + 1)

# 전체 부모 관계 설정
dfs(1, 0)  # 루트 노드 = 1번 노드
for i in range(1, log + 1):
    for j in range(1, N + 1):
        parent[j][i] = parent[parent[j][i - 1]][i - 1]

def lca(a, b):  # a, b의 최소 공통 조상 찾기
    if d[a] > d[b]:
        a, b = b, a  # b가 더 깊도록

    for i in range(log, -1, -1):  # a와 b를 같은 depth로 만들기
        if d[b] - d[a] >= 1 << i:
            b = parent[b][i]

    if a == b: return a  # depth가 같을 때 a==b라면 최소 공통 조상 찾음

    for i in range(log, -1, -1):  # a!=b라면 depth 하나씩 줄이면서 확인
        if parent[a][i] != parent[b][i]:
            a = parent[a][i]
            b = parent[b][i]

    return parent[a][0]  # 최소 공통 조상 (= parent[b][0])

M = int(input())
for _ in range(M):
    A, B = map(int, input().split())
    print(lca(A, B))


# 3176

import sys
from collections import deque
input = sys.stdin.readline
inf = sys.maxsize
N = int(input())
log = 17  # 2 ≤ N ≤ 100000 → log(100000) = 16.6... > 17
connect = [[] for _ in range(N + 1)]
for _ in range(N - 1):
    A, B, C = map(int, input().split())  # A와 B 사이에 길이가 C인 도로가 있다
    connect[A].append((B, C))
    connect[B].append((A, C))

# DFS
q = deque([1])
parent = [[0, 0] for _ in range(N + 1)]
visited = [False] * (N + 1)
visited[1] = True
depth = [0] * (N + 1)
while len(q) > 0:
    temp = q.popleft()
    for b, c in connect[temp]:
        if visited[b] == False:
            q.append(b)
            depth[b] = depth[temp] + 1
            visited[b] = True
            parent[b] = [temp, c]

# 부모 노드, 가장 짧은 도로, 가장 긴 도로
DP = [[[0, 0, 0] for _ in range(log + 1)] for _ in range(N + 1)]
for i in range(1, N + 1):
    DP[i][0] = [parent[i][0], parent[i][1], parent[i][1]]  # 초기화
for j in range(1, log + 1):
    for i in range(1, N + 1):
        DP[i][j][0] = DP[DP[i][j - 1][0]][j - 1][0]
        DP[i][j][1] = min(DP[i][j - 1][1], DP[DP[i][j - 1][0]][j - 1][1])
        DP[i][j][2] = max(DP[i][j - 1][2], DP[DP[i][j - 1][0]][j - 1][2])

K = int(input())
for _ in range(K):
    D, E = map(int, input().split())  # D와 E를 연결하는 경로에서 가장 짧고, 가장 긴 도로 출력
    if depth[D] > depth[E]: D, E = E, D  # E가 더 깊도록

    mini = inf
    maxi = 0
    for i in range(log, -1, -1):  # D와 E를 같은 depth로 만들기
        if depth[E] - depth[D] >= 1 << i:
            mini = min(mini, DP[E][i][1])
            maxi = max(maxi, DP[E][i][2])
            E = DP[E][i][0]

    if D == E:  # depth가 같을 때 D==E라면 최소 공통 조상 찾음
        print(mini, maxi)
        continue

    for i in range(log, -1, -1):  # D!=E라면 depth 하나씩 줄이면서 확인
        if DP[D][i][0] != DP[E][i][0]:
            mini = min(mini, DP[D][i][1], DP[E][i][1])
            maxi = max(maxi, DP[D][i][2], DP[E][i][2])
            D = DP[D][i][0]
            E = DP[E][i][0]

    mini = min(mini, DP[D][0][1], DP[E][0][1])  # 최소 공통 조상
    maxi = max(maxi, DP[D][0][2], DP[E][0][2])  # 최소 공통 조상
    print(mini, maxi)


# 13511

import sys
from collections import deque
input = sys.stdin.readline
inf = sys.maxsize
N = int(input())
log = 17  # 2 ≤ N ≤ 100000 → log(100000) = 16.6... > 17
connect = [[] for _ in range(N + 1)]
for _ in range(N - 1):
    u, v, w = map(int, input().split())
    connect[u].append((v, w))
    connect[v].append((u, w))

# DFS
q = deque([1])
parent = [[0, 0] for _ in range(N + 1)]
visited = [False] * (N + 1)
visited[1] = True
depth = [0] * (N + 1)
while len(q) > 0:
    temp = q.popleft()
    for b, c in connect[temp]:
        if visited[b] == False:
            q.append(b)
            depth[b] = depth[temp] + 1
            visited[b] = True
            parent[b] = [temp, c]

# 부모 노드, 거리
DP = [[[0, 0] for _ in range(log + 1)] for _ in range(N + 1)]
for i in range(1, N + 1):
    DP[i][0] = [parent[i][0], parent[i][1]]  # 초기화
for j in range(1, log + 1):
    for i in range(1, N + 1):
        DP[i][j][0] = DP[DP[i][j - 1][0]][j - 1][0]
        DP[i][j][1] = DP[i][j - 1][1] + DP[DP[i][j - 1][0]][j - 1][1]

M = int(input())
for _ in range(M):
    query = list(map(int, input().split()))

    A, B = query[1], query[2]
    if depth[A] > depth[B]: A, B = B, A  # B가 더 깊도록

    for i in range(log, -1, -1):  # A와 B를 같은 depth로 만들기
        if depth[B] - depth[A] >= 1 << i:
            B = DP[B][i][0]

    if A == B:  # depth가 같을 때 A==B라면 최소 공통 조상 찾음
        root = A
    else:  # A!=B라면 depth 하나씩 줄이면서 확인
        for i in range(log, -1, -1):
            if DP[A][i][0] != DP[B][i][0]:
                A = DP[A][i][0]
                B = DP[B][i][0]
        root = DP[A][0][0]

    A, B = query[1], query[2]

    if query[0] == 1:  # u에서 v로 가는 경로의 비용
        cost = 0
        for i in range(log, -1, -1):
            if depth[A] - depth[root] >= 1 << i:
                cost += DP[A][i][1]
                A = DP[A][i][0]
            if depth[B] - depth[root] >= 1 << i:
                cost += DP[B][i][1]
                B = DP[B][i][0]
        print(cost)

    elif query[0] == 2:  # u에서 v로 가는 경로에 존재하는 정점 중에서 k번째 정점
        k = query[3]
        if k <= depth[A] - depth[root]:  # A의 k-1번째 조상 출력
            K = k - 1
            for i in range(log + 1):
                if K & 1 << i:
                    A = DP[A][i][0]
            print(A)
        else:  # 남은 거리를 B의 끝부터 계산
            K = depth[B] + depth[A] - 2 * depth[root] - (k - 1)
            for i in range(log, -1, -1):
                if K & 1 << i:
                    B = DP[B][i][0]
            print(B)


# -----------------------------------
# fourtyone (강한 연결 요소)


# 2150 (Kosaraju)

import sys
from collections import defaultdict, deque
sys.setrecursionlimit(10 ** 6)
input = sys.stdin.readline
V, E = map(int, input().split())
forward = defaultdict(list)
backward = defaultdict(list)
for _ in range(E):
    A, B = map(int, input().split())
    forward[A].append(B)
    backward[B].append(A)

def dfs(n):
    visited[n] = True
    for nxt in forward[n]:
        if visited[nxt] == False:
            dfs(nxt)
    stack.append(n)

visited = [False] * (V + 1)
stack = []
for i in range(1, V + 1):
    if visited[i] == False:
        dfs(i)

def reverseDfs(n, group):
    visited[n] = True
    group.append(n)
    for nxt in backward[n]:
        if visited[nxt] == False:
            group = reverseDfs(nxt, group)
    return group

visited = [False] * (V + 1)
answer = []
while len(stack) > 0:
    now = stack.pop()
    if visited[now] == True:
        continue
    answer.append(sorted(reverseDfs(now, [])))

print(len(answer))
for scc in sorted(answer):
    print(' '.join(map(str, scc)), end=" -1\n")


# 2150 (Tarjan)

import sys
from collections import defaultdict, deque
sys.setrecursionlimit(10 ** 6)
input = sys.stdin.readline
V, E = map(int, input().split())
connect = defaultdict(list)

for _ in range(E):
    A, B = map(int, input().split())
    connect[A].append(B)

def scc(now):
    global visitcnt
    global scccnt
    visitorder[now] = visitcnt
    visitcnt += 1
    stack.append(now)
    minidx = visitorder[now]
    for nxt in connect[now]:
        if visitorder[nxt] == -1:  # 방문하지 않은 노드
            minidx = min(minidx, scc(nxt))
        elif sccorder[nxt] == -1:  # 방문했지만 scc 번호가 없는 노드
            minidx = min(minidx, visitorder[nxt])
    if minidx == visitorder[now]:  # 가장 이른 노드가 자기 자신이라면
        while stack[-1] != now:
            sccorder[stack.pop()] = scccnt
        sccorder[stack.pop()] = scccnt
        scccnt += 1
    return minidx

visitcnt, scccnt = 0, 0
visitorder = [-1] * (V + 1)
sccorder = [-1] * (V + 1)
stack = []
for i in range(1, V + 1):
    if visitorder[i] == -1:
        scc(i)

print(scccnt)
ans = [[] for _ in range(scccnt)]
for i in range(1, V + 1):
    ans[sccorder[i]].append(i)
for scc in sorted(ans):
    print(' '.join(map(str, scc)), end=" -1\n")


# 4196

import sys
from collections import defaultdict
sys.setrecursionlimit(10 ** 6)
input = sys.stdin.readline
T = int(input())

def dfs(n):
    visited[n] = True
    for nxt in forward[n]:
        if visited[nxt] == False:
            dfs(nxt)
    stack.append(n)

def reverseDfs(n, group):
    global idx
    ids[n] = idx
    visited[n] = True
    group.append(n)
    for nxt in backward[n]:
        if visited[nxt] == False:
            group = reverseDfs(nxt, group)
    return group

for _ in range(T):
    N, M = map(int, input().split())
    forward = defaultdict(list)
    backward = defaultdict(list)
    for _ in range(M):
        x, y = map(int, input().split())
        forward[x].append(y)
        backward[y].append(x)

    visited = [False] * (N + 1)
    stack = []
    for i in range(1, N + 1):
        if visited[i] == False:
            dfs(i)

    idx = 0
    ids = [-1] * (N + 1)
    visited = [False] * (N + 1)
    answer = []
    while len(stack) > 0:
        now = stack.pop()
        if visited[now] == False:
            idx += 1
            answer.append(reverseDfs(now, []))

    scc_indegree = [0] * (idx + 1)
    for i in range(1, N + 1):
        for ed in forward[i]:
            if ids[i] != ids[ed]:
                scc_indegree[ids[ed]] += 1

    cnt = 0
    for i in range(1, len(scc_indegree)):
        if scc_indegree[i] == 0:
            cnt += 1

    print(cnt)


# 3977

import sys
from collections import defaultdict
sys.setrecursionlimit(10 ** 6)
input = sys.stdin.readline
T = int(input())

def dfs(n):
    visited[n] = True
    for nxt in forward[n]:
        if visited[nxt] == False:
            dfs(nxt)
    stack.append(n)

def reverseDfs(n, group):
    global idx
    ids[n] = idx
    visited[n] = True
    group.append(n)
    for nxt in backward[n]:
        if visited[nxt] == False:
            group = reverseDfs(nxt, group)
    return group

for t in range(T):
    N, M = map(int, input().split())
    forward = defaultdict(list)
    backward = defaultdict(list)
    for _ in range(M):
        A, B = map(int, input().split())  # A to B
        forward[A].append(B)
        backward[B].append(A)

    visited = [False] * N
    stack = []
    for i in range(N):
        if visited[i] == False:
            dfs(i)

    idx = -1
    ids = [-1] * N
    visited = [False] * N
    answer = []
    while len(stack) > 0:
        now = stack.pop()
        if visited[now] == False:
            idx += 1
            answer.append(reverseDfs(now, []))

    scc_indegree = [0] * (idx + 1)
    for i in range(N):
        for ed in forward[i]:
            if ids[i] != ids[ed]:
                scc_indegree[ids[ed]] += 1

    cnt = 0
    possible_list = []
    for i in range(len(scc_indegree)):
        if scc_indegree[i] == 0:  # 해당 아이디로 향하는 것은 아이디가 같거나 향하는 게 없음
            cnt += 1
            for j in answer[i]:
                possible_list.append(j)

    if cnt == 1: print('\n'.join(map(str, sorted(possible_list))))
    else: print("Confused")  # cnt == 0 or cnt > 1

    if t == T - 1: break  # last one
    else:  # t < T - 1
        print()
        _ = input()
"""

# 4013

import sys
input = sys.stdin.readline










# https://www.acmicpc.net/step/43