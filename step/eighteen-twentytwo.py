
"""
# -----------------------------------
# eighteen (동적 계획법 1)


# 24416

import sys
input = sys.stdin.readline
n = int(input())
# 3 -> 2 / 4 -> 2+1 / 5 -> 3+2 / 6 -> 5+3
a = [0, 1, 1]
for i in range(n-2):
    a.append(a[i+1]+a[i+2])
res1 = a[n]
res2 = n - 2
print(res1, res2)


# 9184

def findw(a, b, c):
    if a==0 and b==0 and c==0: return w[0][0][0]
    elif a<=0 or b<=0 or c<=0: return findw(0, 0, 0)
    elif a>20 or b>20 or c>20: return findw(20, 20, 20)
    elif w[a][b][c] != 0: return w[a][b][c]
    elif a < b < c: w[a][b][c] = findw(a,b,c-1) + findw(a,b-1,c-1) - findw(a,b-1,c)
    else: w[a][b][c] = findw(a-1,b,c) + findw(a-1,b-1,c) + findw(a-1,b,c-1) - findw(a-1,b-1,c-1)
    return w[a][b][c]

import sys
input = sys.stdin.readline
a, b, c = map(int, input().split())
w = []
for _ in range(21):
    temp1 = []
    for _ in range(21):
        temp2 = [0] * 21
        temp1.append(temp2)
    w.append(temp1)
w[0][0][0] = 1
while a != -1 or b != -1 or c != -1:
    res = findw(a, b, c)
    print("w(%d, %d, %d) = %d" % (a, b, c, res))
    a, b, c = map(int, input().split())


# 1904

import sys
input = sys.stdin.readline
N = int(input())
# 1 / 2 / 3 / 4 / 5 /  6 /  7
# 1 / 2 / 3 / 5 / 8 / 13 / 21
ans = [1, 2]
for i in range(N-1):
    ans.append((ans[i] + ans[i + 1]) % 15746)
print(ans[N-1])


# 9461

import sys
input = sys.stdin.readline
T = int(input())
P = [1, 1, 1, 2, 2]
for _ in range(T):
    N = int(input())
    if len(P) < N:
        for i in range(len(P)-1, N):
            P.append(P[i]+P[i-4])
    print(P[N-1])


# 1912

import sys
input = sys.stdin.readline
n = int(input())
ns = list(map(int, input().split()))
last = ns[0]
totalmax = last
ans = [ns[0]]
for i in range(1, n):
    last = max(last+ns[i], ns[i])
    ans.append(last)
    if totalmax<last: totalmax = last
print(totalmax)


# 1149

import sys
input = sys.stdin.readline
N = int(input())
house = []
for _ in range(N):
    rgb = list(map(int, input().split()))
    house.append(rgb)
cost = []
for _ in range(N):
    temp = [0, 0, 0]
    cost.append(temp)
for n in range(N):
    if n == 0: cost[0] = house[0]
    else:
        cost[n][0] = min(cost[n-1][1], cost[n-1][2]) + house[n][0]
        cost[n][1] = min(cost[n - 1][0], cost[n - 1][2]) + house[n][1]
        cost[n][2] = min(cost[n - 1][0], cost[n - 1][1]) + house[n][2]
print(min(cost[N-1][0], cost[N-1][1], cost[N-1][2]))


# 1932

import sys
input = sys.stdin.readline
N = int(input())
sum = []
for n in range(1, N+1):
    line = list(map(int, input().split()))
    temp = []
    if n == 1: temp.append(line[0])  # line 1 (one element)
    else:
        for i in range(n):
            if i == 0: temp.append(sum[n-2][0]+line[0])  # line n+1, element 1
            elif i == n-1: temp.append(sum[n-2][i-1]+line[i])  # element n+1
            else: temp.append(max(sum[n-2][i-1], sum[n-2][i])+line[i])
    sum.append(temp)
maxelement = 0
for j in range(N):
    if sum[N-1][j]>maxelement: maxelement = sum[N-1][j]
print(maxelement)


# 2579

import sys
input = sys.stdin.readline
N = int(input())
stair = []
for n in range(N):
    temp = int(input())
    stair.append(temp)
point = []  # [누적합(스텝1), 누적합(스텝2)]
for n in range(N):
    if n == 0: point.append([stair[0], 0])
    elif n == 1: point.append([stair[0]+stair[1], stair[1]])
    else:
        # 한 칸 오른 거라면, n-1번째가 두 칸 오른 거여야 함
        step1 = point[n-1][1] + stair[n]
        # 두 칸 오른 거라면, n-2번째가 한 칸 혹은 두 칸 오른 거여야 함
        step2 = max(point[n-2][0], point[n-2][1]) + stair[n]
        point.append([step1, step2])
print(max(point[N-1][0], point[N-1][1]))


# 1463

import sys
input = sys.stdin.readline
N = int(input())
cnt = [-1] * (N+1)
for i in range(1, N+1):
    if i == 1: cnt[i] = 0
    else:
        if i % 6 == 0: cnt[i] = min(cnt[i-1], cnt[i//3], cnt[i//2]) + 1
        elif i % 3 == 0: cnt[i] = min(cnt[i-1], cnt[i//3]) + 1
        elif i % 2 == 0: cnt[i] = min(cnt[i-1], cnt[i//2]) + 1
        else: cnt[i] = cnt[i-1] + 1
print(cnt[N])


# 10844

import sys
input = sys.stdin.readline
# 9 / 17 ; 10 12 21 23 32 34 43 45 54 56 65 67 76 78 87 89 '98'
# 32 ; '101' 121 123 210 212 232 234 ... 876 878 '898' 987 989
# 0 or 9 -> 1개만
N = int(input())
stair = []
for i in range(N+1):
    if i == 1: temp = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    else: temp = [0] * 10
    stair.append(temp)
for i in range(2, N+1):
    for j in range(10):
        if j == 0: stair[i][j] = stair[i-1][j+1]
        elif j == 9: stair[i][j] = stair[i-1][j-1]
        else: stair[i][j] = (stair[i-1][j-1] + stair[i-1][j+1]) % 1000000000
ans = 0
for i in range(1, 10):
    ans = (ans + stair[N][i]) % 1000000000
print(ans % 1000000000)


# 2156

import sys
input = sys.stdin.readline
N = int(input())
grape = [[0, 0, 0]]  # 0(not), 1(first), 2(last)
for n in range(N):
    temp = int(input())
    zero = max(grape[n][0], grape[n][1], grape[n][2])
    one = grape[n][0] + temp
    two = grape[n][1] + temp
    grape.append([zero, one, two])
print(max(grape[N][0], grape[N][1], grape[N][2]))


# 11053

import sys
input = sys.stdin.readline
N = int(input())
A = list(map(int, input().split()))
res = [0] * N
res[0] = 1
for i in range(1, N):
    for j in range(i):
        if A[j]<A[i] and res[j]>res[i]:
            res[i] = res[j]
    res[i] += 1
maxres = 0
for i in range(N):
    if res[i]>maxres: maxres = res[i]
print(maxres)


# 11054

import sys
input = sys.stdin.readline
N = int(input())
A = list(map(int, input().split()))
upres = [0] * N
upres[0] = 1
for i in range(1, N):
    for j in range(i):
        if A[j]<A[i] and upres[j]>upres[i]:
            upres[i] = upres[j]
    upres[i] += 1
downres = [0] * N
downres[N-1] = 1
for i in range(N-2, -1, -1):
    for j in range(N-1, i, -1):
        if A[j]<A[i] and downres[j]>downres[i]:
            downres[i] = downres[j]
    downres[i] += 1
maxres = 0
for i in range(N):
    if upres[i]+downres[i]-1>maxres:
        maxres = upres[i]+downres[i]-1
print(maxres)


# 2565

import sys
input = sys.stdin.readline
line = int(input())
match = []
for _ in range(line):
    temp = list(map(int, input().split()))
    match.append(temp)
match.sort()
res = [0] * line
res[0] = 1
for i in range(1, line):
    for j in range(i):
        if match[j][0]<match[i][0] and match[j][1]<match[i][1] and res[j]>res[i]:
            res[i] = res[j]
    res[i] += 1
maxres = 0
for i in range(line):
    if res[i]>maxres: maxres = res[i]
print(line-maxres)


# 9251

import sys
input = sys.stdin.readline
A = input().strip()
B = input().strip()
LCS = []
#     0 1 2 3 4 5 6
#     - A C A Y K P
# 0 - 0 0 0 0 0 0 0
# 1 C 0 0 1 1 1 1 1
# 2 A 0 1 1 2 2 2 2
# 3 P 0 1 1 2 2 2 3
# 4 C 0 1 2 2 2 2 3
# 5 A 0 2 2 3 3 3 3
# 6 K 0 2 2 3 3 4 4
for _ in range(len(B)+1):
    temp = [0] * (len(A)+1)
    LCS.append(temp)
for i in range(len(B)):
    for j in range(len(A)):
        if B[i] == A[j]: LCS[i+1][j+1] = LCS[i][j] + 1
        else: LCS[i+1][j+1] = max(LCS[i][j+1], LCS[i+1][j])
print(LCS[len(B)][len(A)])


# 12865

import sys
input = sys.stdin.readline
N, K = map(int, input().split())
thing = [[0, 0]]
for _ in range(N):
    temp = list(map(int, input().split()))  # W, V
    thing.append(temp)
DP = []
for _ in range(N+1):
    temp = [0] * (K+1)
    DP.append(temp)
# 6 13 / 4 8 / 3 6 / 5 12
# NK  0  1  2  3  4  5  6  7
# 0 | 0  0  0  0  0  0  0  0
# 1 | 0  0  0  0  0  0 13 13
# 2 | 0  0  0  0  8  8 13 13
# 3 | 0  0  0  6  8  8 13 14
# 4 | 0  0  0  6  8 12 13 14
for n in range(1, N+1):
    for k in range(K+1):
        if k < thing[n][0]: DP[n][k] = DP[n-1][k]
        else:
            DP[n][k] = max(DP[n-1][k], DP[n-1][k-thing[n][0]]+thing[n][1])
print(DP[N][K])


# -----------------------------------
# nineteen (누적 합)


# 11659

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
Ns = list(map(int, input().split()))
for n in range(1, N):
    Ns[n] += Ns[n-1]
for _ in range(M):
    i, j = map(int, input().split())
    if i == 1: print(Ns[j-1])
    else: print(Ns[j-1]-Ns[i-2])


# 2559

import sys
input = sys.stdin.readline
N, K = map(int, input().split())
Ns = list(map(int, input().split()))
for n in range(1, N):
    Ns[n] += Ns[n-1]
maxi = Ns[K-1]
for k in range(N-K):
    temp = Ns[k+K] - Ns[k]
    if temp > maxi: maxi = temp
print(maxi)


# 16139

import sys
input = sys.stdin.readline
S = input().strip()
q = int(input())
alpha = []  # a ~ z -> 26개
temp = [0] * 26
temp[ord(S[0])-ord('a')] += 1
alpha.append(temp)
for s in range(1, len(S)):
    temp = alpha[s-1].copy()
    temp[ord(S[s])-ord('a')] += 1
    alpha.append(temp)
for _ in range(q):
    a, l, r = map(str, input().strip().split())
    l = int(l)
    r = int(r)
    if l == 0: print(alpha[r][ord(a)-ord('a')])
    else: print(alpha[r][ord(a)-ord('a')]-alpha[l-1][ord(a)-ord('a')])


# 10986

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
cnt = [0] * 1000
sum = 0
A = list(map(int, input().split()))
for n in range(N):
    sum += A[n]
    sum %= M
    cnt[sum] += 1
ans = cnt[0]  # already remain is zero
for i in range(1000):
    ans += int(cnt[i] * (cnt[i] - 1) / 2)  # nC2 = n*(n-1)/2
print(ans)


# 11660

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
Ns = []
temp = [0] * (N+1)
Ns.append(temp)
for i in range(N):
    temp = list(map(int, input().split()))
    temp2 = [0]
    for j in range(N):
        temp2.append(Ns[i][j+1]+temp2[j]+temp[j]-Ns[i][j])
    Ns.append(temp2)
for _ in range(M):
    x1, y1, x2, y2 = map(int, input().split())
    print(Ns[x2][y2]-Ns[x1-1][y2]-Ns[x2][y1-1]+Ns[x1-1][y1-1])


# 25682

import sys
input = sys.stdin.readline
N, M, K = map(int, input().split())
board = []  # B로 시작하는 거 기준
temp = [0] * (M+1)
board.append(temp)
for i in range(N):
    s = input().strip()
    temp = [0]
    for j in range(M):
        if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
            if s[j] == 'B': temp.append(board[i][j+1]+temp[j]+0-board[i][j])
            else: temp.append(board[i][j+1]+temp[j]+1-board[i][j])
        else:
            if s[j] == 'W': temp.append(board[i][j+1]+temp[j]+0-board[i][j])
            else: temp.append(board[i][j+1]+temp[j]+1-board[i][j])
    board.append(temp)
maxi = 0
mini = M * N
for i in range(N-K+1):
    for j in range(M-K+1):
        cal = board[i+K][j+K]-board[i][j+K]-board[i+K][j]+board[i][j]
        if cal>maxi: maxi = cal
        if cal<mini: mini = cal
print(min(K*K-maxi, mini))


# -----------------------------------
# twenty (그리디 알고리즘)


# 11047

import sys
input = sys.stdin.readline
N, K = map(int, input().split())
A = []
for _ in range(N):
    A.append(int(input()))
cnt = 0
for i in range(N-1, -1, -1):
    if K == 0: continue
    cnt += K // A[i]
    K %= A[i]
print(cnt)


# 1931

import sys
input = sys.stdin.readline
N = int(input())
time = []
for _ in range(N):
    time.append(list(map(int, input().split())))
time.sort(key=lambda x : (x[1], x[0]))
cnt = 1
last = time[0]
for i in range(1, N):
    if time[i][0] >= last[1]:
        cnt += 1
        last = time[i]
print(cnt)


# 11399

import sys
input = sys.stdin.readline
N = int(input())
P = list(map(int, input().split()))
P.sort()
sum = P[0]
for n in range(1, N):
    P[n] += P[n-1]
    sum += P[n]
print(sum)


# 1541

import sys
input = sys.stdin.readline
S = input().strip()
start = 0
sum = 0
minus = False
for i in range(len(S)):
    if S[i] == '+' or S[i] == '-' or i == len(S)-1:
        if i == len(S)-1:
            i += 1
        if minus:
            sum -= int(S[start:i])
            start = i+1
        else:  # never
            sum += int(S[start:i])
            start = i+1
            if i < len(S)-1:
                if S[i] == '-': minus = True
print(sum)


# 13305

import sys
input = sys.stdin.readline
N = int(input())
length = list(map(int, input().split()))
price = list(map(int, input().split()))
minprice = 1000000000
sum = 0
for i in range(N-1):
    if price[i]<minprice: minprice = price[i]
    sum += minprice * length[i]
print(sum)


# -----------------------------------
# twentyone (스택)


# 10828

import sys
input = sys.stdin.readline
N = int(input())
stack = []
for _ in range(N):
    S = input().strip()
    if S == "pop":
        if len(stack) > 0:
            print(stack[-1])
            stack.pop()
        else:
            print(-1)
    elif S == "size":
        print(len(stack))
    elif S == "empty":
        if len(stack) == 0:
            print(1)
        else:
            print(0)
    elif S == "top":
        if len(stack) > 0:
            print(stack[-1])
        else:
            print(-1)
    else:
        push, X = map(str, S.split())
        stack.append(int(X))


# 10773

import sys
input = sys.stdin.readline
K = int(input())
stack = []
for _ in range(K):
    temp = int(input())
    if temp == 0:
        stack.pop()
    else:
        stack.append(temp)
sum = 0
for i in range(len(stack)):
    sum += stack[i]
print(sum)


# 9012

import sys
input = sys.stdin.readline
T = int(input())
for _ in range(T):
    S = input().strip()
    stack = []
    VPS = True
    for s in range(len(S)):
        if S[s] == "(":
            stack.append(S[s])
        else:
            if len(stack) == 0:
                VPS = False
                break
            else:
                stack.pop()
    if len(stack) != 0: VPS = False
    if VPS: print("YES")
    else: print("NO")


# 4949

import sys
input = sys.stdin.readline
S = input()
while len(S) > 2:
    stack = []
    balance = True
    needtocheck = False
    for s in range(len(S)-1):
        if S[s] in ["(", ")", "[", "]"]:
            needtocheck = True
        if S[s] == "(" or S[s] == "[":
            stack.append(S[s])
        elif S[s] == ")":
            if len(stack) > 0 and stack[-1] == "(":
                stack.pop()
            else:
                balance = False
                break
        elif S[s] == "]":
            if len(stack) > 0 and stack[-1] == "[":
                stack.pop()
            else:
                balance = False
                break
    if len(stack) != 0: balance = False
    if not needtocheck: print("yes")
    elif balance: print("yes")
    else: print("no")
    S = input()


# 1874

import sys
input = sys.stdin.readline
n = int(input())
stack = []
now = 0
ans = []
for _ in range(n):
    temp = int(input())
    if len(stack)==0 or stack[-1]<temp:
        for i in range(temp-now):
            now += 1
            stack.append(now)
            ans.append("+")
        stack.pop()
        ans.append("-")
    elif stack[-1]==temp:
        stack.pop()
        ans.append("-")
    else:  # stack[-1]>temp
        print("NO")
        exit(0)
for j in range(len(ans)): print(ans[j])


# -----------------------------------
# twentytwo (큐, 덱)


# 18258

import sys
from collections import deque
input = sys.stdin.readline
N = int(input())
queue = deque([])
for _ in range(N):
    S = input().strip()
    if S == "pop":
        if len(queue) > 0:
            print(queue[0])
            queue.popleft()
        else:
            print(-1)
    elif S == "size":
        print(len(queue))
    elif S == "empty":
        if len(queue) == 0:
            print(1)
        else:
            print(0)
    elif S == "front":
        if len(queue) > 0:
            print(queue[0])
        else:
            print(-1)
    elif S == "back":
        if len(queue) > 0:
            print(queue[-1])
        else:
            print(-1)
    else:
        push, X = map(str, S.split())
        queue.append(int(X))


# 2164

import sys
from collections import deque
input = sys.stdin.readline
N = int(input())
queue = deque([])
for i in range(1, N+1): queue.append(i)
while len(queue)>1:
    queue.popleft()
    if len(queue)==1:
        print(queue[0])
        exit(0)
    queue.append(queue[0])
    queue.popleft()
print(queue[0])


# 11866

import sys
from collections import deque
input = sys.stdin.readline
N, K = map(int, input().split())
queue = deque([])
ans = []
for i in range(1, N+1): queue.append(i)
while len(queue) > 1:
    for i in range(K-1):
        queue.append(queue[0])
        queue.popleft()
    ans.append(queue[0])
    queue.popleft()
ans.append(queue[0])
print("<%d" % ans[0], end="")
for i in range(1, len(ans)):
    print(", %d" % ans[i], end="")
print(">")


# 1966

import sys
from collections import deque
input = sys.stdin.readline
T = int(input())
for _ in range(T):
    N, M = map(int, input().split())
    important = list(map(int, input().split()))
    important2 = important.copy()
    important.sort(reverse=True)
    i_queue = deque(important2)
    queue = deque([])
    for i in range(N): queue.append(i)
    cnt = 0
    max_idx = 0
    while max_idx < N:
        if i_queue[0] == important[max_idx]:
            cnt += 1
            if queue[0] == M:
                print(cnt)
                break
            max_idx += 1
            i_queue.popleft()
            queue.popleft()
        else:
            i_queue.append(i_queue[0])
            i_queue.popleft()
            queue.append(queue[0])
            queue.popleft()


# 10866

import sys
from collections import deque
input = sys.stdin.readline
N = int(input())
queue = deque([])
for _ in range(N):
    S = input().strip()
    if S == "pop_front":
        if len(queue) > 0:
            print(queue[0])
            queue.popleft()
        else:
            print(-1)
    elif S == "pop_back":
        if len(queue) > 0:
            print(queue[-1])
            queue.pop()
        else:
            print(-1)
    elif S == "size":
        print(len(queue))
    elif S == "empty":
        if len(queue) == 0:
            print(1)
        else:
            print(0)
    elif S == "front":
        if len(queue) > 0:
            print(queue[0])
        else:
            print(-1)
    elif S == "back":
        if len(queue) > 0:
            print(queue[-1])
        else:
            print(-1)
    else:
        push, X = map(str, S.split())
        if push == "push_front":
            queue.appendleft(int(X))
        elif push == "push_back":
            queue.append(int(X))


# 1021

import sys
from collections import deque
input = sys.stdin.readline
N, M = map(int, input().split())
idx = list(map(int, input().split()))
queue = deque([])
for i in range(1, N+1): queue.append(i)
ans = 0
for i in range(M):
    cnt = 0
    while queue[0] != idx[i]:
        cnt += 1
        queue.append(queue[0])
        queue.popleft()
    cnt = min(cnt, len(queue)-cnt)
    queue.popleft()
    ans += cnt
print(ans)
"""

# 5430

import sys
from collections import deque
input = sys.stdin.readline
T = int(input())
for _ in range(T):
    p = input().strip()
    n = int(input())
    temp = input().strip()[1:-1]
    if len(temp) > 0: x = deque(list(map(int, temp.split(','))))
    else: x = deque([])
    left = True
    error = False
    for i in range(len(p)):
        if p[i] == "R":
            if left: left = False
            else: left = True
        elif p[i] == "D":
            if len(x) == 0:
                error = True
                break
            else:
                if left: x.popleft()
                else: x.pop()
    if error: print("error")
    else:
        print("[", end="")
        if len(x) != 0:
            if left:
                print(x[0], end="")
                for i in range(1, len(x)):
                    print(",%d" % x[i], end="")
            else:
                print(x[-1], end="")
                for i in range(len(x)-2, -1, -1):
                    print(",%d" % x[i], end="")
        print("]")


