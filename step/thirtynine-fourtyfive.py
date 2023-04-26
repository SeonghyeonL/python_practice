
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
"""

# -----------------------------------
# fourty (최소 공통 조상)


# 3584 (LCA)

import sys
input = sys.stdin.readline







# https://www.acmicpc.net/step/40