
"""
# -----------------------------------
# twentythree (분할 정복)


# 2630

def cal(X, Y, len):
    done = True
    first = color[X][Y]
    if len == 1:
        cnt[first] += 1
        return
    for x in range(X, X+len):
        for y in range(Y, Y+len):
            if done == False: break
            elif color[x][y] != first: done = False
    if done:
        cnt[first] += 1
        return
    else:
        len //= 2
        cal(X, Y, len)
        cal(X, Y + len, len)
        cal(X + len, Y, len)
        cal(X + len, Y + len, len)
        return

import sys
input = sys.stdin.readline
N = int(input())
color = []
cnt = [0, 0]
for _ in range(N):
    temp = list(map(int, input().split()))
    color.append(temp)
cal(0, 0, N)
print(cnt[0])
print(cnt[1])


# 1992

def cal(X, Y, len):
    done = True
    first = image[X][Y]
    if len == 1:
        print(first, end="")
        return
    for x in range(X, X+len):
        for y in range(Y, Y+len):
            if done == False: break
            elif image[x][y] != first: done = False
    if done:
        print(first, end="")
    else:
        print("(", end="")
        len //= 2
        cal(X, Y, len)
        cal(X, Y + len, len)
        cal(X + len, Y, len)
        cal(X + len, Y + len, len)
        print(")", end="")

import sys
input = sys.stdin.readline
N = int(input())
image = []
for _ in range(N):
    temp = input().strip()
    image.append(temp)
cal(0, 0, N)


# 1780

def cal(X, Y, len):
    done = True
    first = paper[X][Y]
    if len == 1:
        cnt[first+1] += 1
        return
    for x in range(X, X+len):
        for y in range(Y, Y+len):
            if done == False: break
            elif paper[x][y] != first: done = False
    if done:
        cnt[first+1] += 1
        return
    else:
        len //= 3
        cal(X, Y, len)
        cal(X, Y + len, len)
        cal(X, Y + len * 2, len)
        cal(X + len, Y, len)
        cal(X + len, Y + len, len)
        cal(X + len, Y + len * 2, len)
        cal(X + len * 2, Y, len)
        cal(X + len * 2, Y + len, len)
        cal(X + len * 2, Y + len * 2, len)
        return

import sys
input = sys.stdin.readline
N = int(input())
paper = []
cnt = [0, 0, 0]  # -1, 0, 1
for _ in range(N):
    temp = list(map(int, input().split()))
    paper.append(temp)
cal(0, 0, N)
print(cnt[0])
print(cnt[1])
print(cnt[2])


# 1629

def cal(a, b, c):
    if b == 0: return 1
    elif b == 1: return a % c
    elif b % 2 == 0: return (cal(a, b//2, c) ** 2) % c
    else: return (((cal(a, b//2, c) ** 2) % c) * a) % c

import sys
input = sys.stdin.readline
A, B, C = map(int, input().split())
print(cal(A, B, C))


# 11401

def cal(b, p):
    if p == 1: return b % 1000000007
    elif p % 2 == 0: return (cal(b, p//2) ** 2) % 1000000007
    else: return (((cal(b, p//2) ** 2) % 1000000007) * b) % 1000000007

import sys
input = sys.stdin.readline
N, K = map(int, input().split())
# nCk = n*...*(n-k+1) / k*...*1 = n! / k!(n-k)! = A / B
# 1000000007로 나눈 나머지
# 페르마의 소정리: 소수 p, 정수 a에 대해 a^p mod p = a mod p (a^(p-1) = 1)
# nCk = (A * B^(-1)) % p = (A * B^(p-2)) % p
A, B = 1, 1
for i in range(1, N+1):
    A = (A * i) % 1000000007
for i in range(1, K+1):
    B = (B * i) % 1000000007
for i in range(1, N-K+1):
    B = (B * i) % 1000000007
print((A * cal(B, 1000000007-2)) % 1000000007)


# 2740

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
A = []
for _ in range(N):
    temp = list(map(int, input().split()))
    A.append(temp)
M, K = map(int, input().split())
B = []
for _ in range(M):
    temp = list(map(int, input().split()))
    B.append(temp)
for n in range(N):
    for k in range(K):
        temp = 0
        for m in range(M):
            temp += A[n][m] * B[m][k]
        print(temp, end=" ")
    print()


# 10830

def cal(b):
    if b == 1:
        n = len(A)  # n = N
        for i in range(n):
            for j in range(n):
                A[i][j] %= 1000
        return A
    else:
        n = len(A)  # n = N
        temp = cal(b//2)
        res = []
        for i in range(n):
            line = []
            for j in range(n):
                sum = 0
                for k in range(n):
                    sum = (sum + temp[i][k] * temp[k][j]) % 1000
                line.append(sum)
            res.append(line)
        if b % 2 == 0: return res
        else:
            realres = []
            for i in range(n):
                line = []
                for j in range(n):
                    sum = 0
                    for k in range(n):
                        sum = (sum + res[i][k] * A[k][j]) % 1000
                    line.append(sum)
                realres.append(line)
            return realres

import sys
input = sys.stdin.readline
N, B = map(int, input().split())
A = []
for _ in range(N):
    A.append(list(map(int, input().split())))
# 1000으로 나눈 나머지를 출력
answer = cal(B)
for i in range(N):
    print(' '.join(map(str, answer[i])))


# 11444

def cal(N, now):
    if N == 1: return now
    elif N % 2 == 0: return cal(N//2, multi(now, now))
    else: return multi(cal(N-1, now), now)

def multi(a, b):
    res = [[0, 0], [0, 0]]
    for i in range(2):
        for j in range(2):
            sum = 0
            for k in range(2):
                sum = (sum + a[i][k] * b[k][j]) % 1000000007
            res[i][j] = sum
    return res

import sys
input = sys.stdin.readline
n = int(input())
set = [[1, 1], [1, 0]]
# 1000000007으로 나눈 나머지를 출력
# 0 1 1 2 3 5 8 13 21
# 0 1 2 3 4 5 6  7  8
# [F(n+2)  = [1 1  [F(n+1)  → [F(n+1) F(n)    = [1 1  ^ n
#  F(n+1)]    1 0]  F(n)  ]    F(n)   F(n-1)]    1 0]
# F(n)을 찾기 위해서는 1,1;1,0을 n-1 거듭제곱 해야 함
if n<3: print(1)  # 이것 때문에 계속 틀림 주의!!!
else: print(cal(n-1, set)[0][0])


# 6549

def findmax(a, b):  # a는 왼쪽 끝, b는 오른쪽 끝
    if a == b: return hist[a]  # 너비 = 1, 높이 = hist[a]
    else:
        m = (a + b) // 2  # 가운데 기준
        bound_h = min(hist[m], hist[m + 1])
        boundmax = 2 * bound_h
        bound_l = m
        bound_r = m + 1
        width = 2
        while True:
            if (hist[bound_l] == 0 or bound_l == a) and (hist[bound_r] == 0 or bound_r == b):
                break
            elif hist[bound_l] == 0 or bound_l == a:
                if hist[bound_r + 1] < bound_h: bound_h = hist[bound_r + 1]
                bound_r += 1
            elif hist[bound_r] == 0 or bound_r == b:
                if hist[bound_l - 1] < bound_h: bound_h = hist[bound_l - 1]
                bound_l -= 1
            else:  # 오른쪽, 왼쪽 모두 여유가 있을 때
                if hist[bound_l - 1] < hist[bound_r + 1]:  # 더 큰 오른쪽 확장
                    if hist[bound_r + 1] < bound_h: bound_h = hist[bound_r + 1]
                    bound_r += 1
                else:  # 더 큰 왼쪽 확장
                    if hist[bound_l - 1] < bound_h: bound_h = hist[bound_l - 1]
                    bound_l -= 1
            width += 1
            boundmax = max(boundmax, bound_h * width)  # 가운데 기준 최대 넓이
        return max(findmax(a, m), findmax(m + 1, b), boundmax)  # 좌, 우, 가운데 중 최대

import sys
input = sys.stdin.readline
hist = list(map(int, input().split()))
while hist != [0]:
    print(findmax(1, len(hist) - 1))
    hist = list(map(int, input().split()))


# -----------------------------------
# twentyfour (이분 탐색)


# 1920

import sys
input = sys.stdin.readline
N = int(input())
A = list(map(int, input().split()))
A.sort()
M = int(input())
MS = list(map(int, input().split()))
for m in range(M):
    left = 0
    right = N - 1
    found = False
    while left <= right:
        now = int((left + right) / 2)
        if MS[m] == A[now]:
            found = True
            break
        elif MS[m] > A[now]:
            left = now + 1
        elif MS[m] < A[now]:
            right = now - 1
    if found: print(1)
    else: print(0)


# 10816

import sys
input = sys.stdin.readline
N = int(input())
card = list(map(int, input().split()))
card.sort()
cnt = {}  # dictionary
for i in card:
    if i in cnt: cnt[i] += 1
    else: cnt[i] = 1
M = int(input())
number = list(map(int, input().split()))
for m in range(M):
    left = 0
    right = N - 1
    found = False
    while left <= right:
        now = int((left + right) / 2)
        if number[m] == card[now]:
            found = True
            break
        elif number[m] > card[now]:
            left = now + 1
        elif number[m] < card[now]:
            right = now - 1
    if found:
        print(cnt.get(number[m]), end=" ")
    else:
        print(0, end=" ")


# 1654

import sys
input = sys.stdin.readline
K, N = map(int, input().split())
centi = []
for _ in range(K):
    centi.append(int(input()))
# parametric search -> 이 길이로 자르면 조건을 만족할 수 있는가
left = 1
right = max(centi)
ans = 0
while left <= right:
    mid = (left + right) // 2
    cnt = 0
    for i in range(K):
        cnt += centi[i] // mid
    if cnt >= N:
        left = mid + 1
        ans = mid
    else:
        right = mid - 1
print(ans)


# 2805

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
tree = list(map(int, input().split()))
low = 0
high = max(tree)
ans = 0
while low <= high:
    mid = (low + high) // 2
    total = 0
    for i in range(N):
        total += max(tree[i] - mid, 0)
    if total >= M:
        low = mid + 1
        ans = mid
    else:
        high = mid - 1
print(ans)


# 2110

import sys
input = sys.stdin.readline
N, C = map(int, input().split())
x = []
for _ in range(N):
    x.append(int(input()))
x.sort()
left = 1
right = x[-1] - x[0]
answer = 0
while left <= right:
    mid = (left + right) // 2
    cnt = 1  # 1번 집에는 반드시 설치해야 하는 듯
    lastidx = 0
    for i in range(1, N):
        if x[i] - x[lastidx] >= mid:
            cnt += 1
            lastidx = i
    if cnt >= C:
        answer = mid
        left = mid + 1
    else:
        right = mid - 1
print(answer)


# 1300

import sys
input = sys.stdin.readline
N = int(input())
k = int(input())
# 1 2  3  4 ; min(10//1,4)=4
# 2 4  6  8 ; min(10//2,4)=4
# 3 6  9 12 ; min(10//3,4)=3
# 4 8 12 16 ; min(10//4,4)=2
left = 1
right = N * N
ans = 0
while left <= right:
    mid = (left + right) // 2
    cnt = 0
    for i in range(1, N+1):
        cnt += min(mid // i, N)
    if cnt >= k:
        right = mid - 1
        ans = mid
    else:
        left = mid + 1
print(ans)


# 12015

import sys
input = sys.stdin.readline
N = int(input())
A = list(map(int, input().split()))
ans = [A[0]]
for i in range(1, N):
    if A[i] > ans[-1]: ans.append(A[i])
    else:
        #for j in range(len(ans)):
        #    if ans[j] >= A[i]: ans[j] = A[i]
        left = 0
        right = len(ans) - 1
        tempmid = 0
        while left <= right:
            mid = (left + right) // 2
            if ans[mid] >= A[i]:
                tempmid = mid
                right = mid - 1
            else:
                left = mid + 1
        ans[tempmid] = A[i]
print(len(ans))


# -----------------------------------
# twentyfive (우선순위 큐)


# 11279

import sys
import heapq  # 최소 힙
input = sys.stdin.readline
N = int(input())
# 최소 힙: 부모 노드의 키 < 자식 노드의 키
# 최대 힙: 부모 노드의 키 > 자식 노드의 키
heap = []
for _ in range(N):
    x = int(input())
    if x > 0: heapq.heappush(heap, (-x, x))  # 최대 힙으로 만들기 위해
    else:
        if len(heap) == 0: print(0)
        else: print(heapq.heappop(heap)[1])


# 1927

import sys
import heapq
input = sys.stdin.readline
N = int(input())
heap = []
for _ in range(N):
    x = int(input())
    if x > 0: heapq.heappush(heap, x)
    else:
        if len(heap) == 0: print(0)
        else: print(heapq.heappop(heap))


# 11286

import sys
import heapq
input = sys.stdin.readline
N = int(input())
heap = []
for _ in range(N):
    x = int(input())
    if x != 0: heapq.heappush(heap, (abs(x), x))
    else:
        if len(heap) == 0: print(0)
        else: print(heapq.heappop(heap)[1])


# -----------------------------------
# twentysix (동적 계획법 2)


# 11066

import sys
input = sys.stdin.readline
T = int(input())
for _ in range(T):
    K = int(input())
    file = list(map(int, input().split()))
    sum = [0, file[0]]
    for k in range(1, K): sum.append(sum[k]+file[k])
    findmin = [[0] * (K+1) for _ in range(K+1)]
    for size in range(1, K):
        for start in range(1, K+1-size):
            end = start + size  # 2 ~ K
            temp = float("inf")
            for k in range(start, end):
                temp = min(temp, findmin[start][k] + findmin[k+1][end] + sum[end] - sum[start - 1])
            findmin[start][end] = temp
    print(findmin[1][K])


# 11049

import sys
input = sys.stdin.readline
N = int(input())
rc = []
for _ in range(N):
    rc.append(list(map(int, input().split())))
findmin = [[0] * N for _ in range(N)]
for size in range(1, N):
    for start in range(N-size):
        end = start + size  # 1 ~ N-1
        temp = float("inf")
        for k in range(start, end):
            temp = min(temp, findmin[start][k] + findmin[k+1][end] + rc[start][0] * rc[k][1] * rc[end][1])
        findmin[start][end] = temp
print(findmin[0][N-1])


# 1520

import sys
input = sys.stdin.readline
M, N = map(int, input().split())
height = []
for _ in range(M):
    height.append(list(map(int, input().split())))
# DP → 전체 문제의 최적해가 부분 문제의 최적해로 쪼개질 수 있는가?
res = [[-1] * N for _ in range(M)]
udlr = [[-1, 0], [1, 0], [0, -1], [0, 1]]

def cal(x, y):
    if res[x][y] == -1:
        if x == M-1 and y == N-1:
            res[x][y] = 1
        else:
            res[x][y] = 0
            for i in range(4):
                nx = x + udlr[i][0]
                ny = y + udlr[i][1]
                if 0 <= nx <= M-1 and 0 <= ny <= N - 1:
                    if height[nx][ny] < height[x][y]:
                        res[x][y] += cal(nx, ny)
    return res[x][y]

print(cal(0, 0))


# 2629

import sys
input = sys.stdin.readline
N = int(input())
w = list(map(int, input().split()))
M = int(input())
q = list(map(int, input().split()))
res = [False] * 40001
res[0] = True

#def cal(now, idx):  # 현재 값, 현재 인덱스
#    if res[now] == 0: res[now] = 1
#    if idx <= N-2:
#        if now - w[idx + 1] >= 0: cal(now - w[idx + 1], idx + 1)
#        if w[idx + 1] - now >= 0: cal(w[idx + 1] - now, idx + 1)
#        if now + w[idx + 1] <= 40000: cal(now + w[idx + 1], idx + 1)
#        cal(now, idx + 1)

#cal(0, -1)

for i in range(N):
    temp = [False] * 40001
    for j in range(40001):
        if res[j]:
            temp[j] = True
            if j + w[i] <= 40000: temp[j + w[i]] = True
            if j - w[i] >= 0: temp[j - w[i]] = True
            if w[i] - j >= 0: temp[w[i] - j] = True
    res = temp

for i in range(M):
    if res[q[i]]: print('Y', end=" ")
    else: print('N', end=" ")


# 2293

import sys
input = sys.stdin.readline
n, k = map(int, input().split())
coin = []
for _ in range(n): coin.append(int(input()))
res = [0] * (k+1)
res[0] = 1
for i in range(n):
    for j in range(coin[i], k+1):
        if j - coin[i] >= 0:
            res[j] += res[j - coin[i]]
print(res[k])


# 7579

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
m = list(map(int, input().split()))
c = list(map(int, input().split()))
findM = [[0] * (sum(c) + 1) for _ in range(N + 1)]
ans = 10000
for i in range(1, N+1):
    for j in range(c[i-1]):  # 앞에도 채워야 함
        findM[i][j] = findM[i-1][j]
    for j in range(c[i-1], sum(c) + 1):
        findM[i][j] = max(findM[i-1][j], findM[i-1][j-c[i-1]]+m[i-1])
        if findM[i][j] >= M:
            if j < ans: ans = j
print(ans)
"""

# -----------------------------------
# twentyseven (스택 2)


# 9935

import sys
input = sys.stdin.readline





# https://www.acmicpc.net/step/51

