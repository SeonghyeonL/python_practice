
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
"""

# -----------------------------------
# twentyfour (이분 탐색)


# 1920

import sys
input = sys.stdin.readline





# https://www.acmicpc.net/step/29

