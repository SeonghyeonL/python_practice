
"""
# -----------------------------------
# thirteen


# 2798

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
cards = list(map(int, input().split()))
max = 0
for i in range(len(cards)):
    for j in range(i+1, len(cards)):
        for k in range(j+1, len(cards)):
            if cards[i]+cards[j]+cards[k]>max and cards[i]+cards[j]+cards[k]<=M:
                max = cards[i]+cards[j]+cards[k]
print(max)


# 2231

import sys
input = sys.stdin.readline
N = int(input())
found = False
for i in range(N):
    sum = i
    temp2 = i
    if temp2 >= 10:
        while temp2>9:
            sum += temp2%10
            temp2 = temp2//10
    sum += temp2
    if sum == N:
        print(i)
        found = True
        break
if found==False: print(0)


# 7568

import sys
input = sys.stdin.readline
N = int(input())
people = []
for _ in range(N):
    x, y = map(int, input().split())
    people.append([x, y])
for n in range(N):
    cnt = 0
    for i in range(N):
        if i==n: continue
        if people[n][0]<people[i][0] and people[n][1]<people[i][1]: cnt += 1
    print(cnt+1, end=" ")


# 1018

def whitecheck(board, a, b):
    cnt = 0
    for A in range(a, a+8, 2):
        for B in range(b, b+8):
            if B%2==0 and board[A][B]=='B': cnt += 1
            elif B%2==1 and board[A][B]=='W': cnt += 1
    for A in range(a+1, a+8, 2):
        for B in range(b, b+8):
            if B%2==0 and board[A][B]=='W': cnt += 1
            elif B%2==1 and board[A][B]=='B': cnt += 1
    return cnt

def blackcheck(board, a, b):
    cnt = 0
    for A in range(a, a+8, 2):
        for B in range(b, b+8):
            if B%2==0 and board[A][B]=='W': cnt += 1
            elif B%2==1 and board[A][B]=='B': cnt += 1
    for A in range(a+1, a+8, 2):
        for B in range(b, b+8):
            if B%2==0 and board[A][B]=='B': cnt += 1
            elif B%2==1 and board[A][B]=='W': cnt += 1
    return cnt

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
board = []
for _ in range(N):
    temp = input().strip()
    board.append(temp)
minimal = 64
for a in range(N-7):
    for b in range(M-7):
        cnt = min(whitecheck(board, a, b), blackcheck(board, a, b))
        if cnt<minimal: minimal = cnt
print(minimal)


# 1436

import sys
input = sys.stdin.readline
N = int(input())
cnt = 0
num = 666
while cnt<N:
    if '666' in str(num): cnt += 1
    num += 1
print(num-1)


# -----------------------------------
# fourteen


# 10815

import sys
input = sys.stdin.readline
N = int(input())
card = list(map(int, input().split()))
card.sort()
M = int(input())
test = list(map(int, input().split()))
for m in range(M):
    start = 0
    end = N-1
    find = False
    while start<=end:
        i = int((start + end) / 2)
        if test[m]==card[i]:
            find = True
            break
        elif test[m]<card[i]: end = i-1
        elif test[m]>card[i]: start = i+1
    if find==True: print(1, end=" ")
    else: print(0, end=" ")


# 14425

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
S = []
cnt = 0
for _ in range(N):
    temp = input().strip()
    S.append(temp)
S.sort()
for _ in range(M):
    temp = input().strip()
    start = 0
    end = N - 1
    find = False
    while start <= end:
        i = int((start + end) / 2)
        if temp == S[i]:
            cnt += 1
            break
        elif temp < S[i]:
            end = i - 1
        elif temp > S[i]:
            start = i + 1
print(cnt)


# 1620

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
mon_1 = []
mon_2 = []
for n in range(N):
    temp = input().strip()
    mon_1.append((n+1, temp))  # num, name
    mon_2.append((temp, n+1))  # name, num
mon_2.sort()
for _ in range(M):
    temp = input().strip()
    if temp[0]>="1" and temp[0]<="9":   # number -> name
        temp = int(temp)
        print(mon_1[temp-1][1])
    else:                               # name -> number
        start = 0
        end = N-1
        while start <= end:
            i = int((start + end) / 2)
            if temp == mon_2[i][0]:
                print(mon_2[i][1])
                break
            elif temp < mon_2[i][0]:
                end = i - 1
            elif temp > mon_2[i][0]:
                start = i + 1


# 10816

import sys
input = sys.stdin.readline
N = int(input())
card = list(map(int, input().split()))
card2 = {}
for n in range(N):
    if card2.get(card[n])==None: card2[card[n]] = 1
    else: card2[card[n]] += 1
M = int(input())
test = list(map(int, input().split()))
for m in range(M):
    if card2.get(test[m])==None: print(0, end=" ")
    else: print(card2[test[m]], end=" ")


# 1764

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
hear = []
for _ in range(N):
    temp = input().strip()
    hear.append(temp)
hear.sort()
res = []
for _ in range(M):
    temp = input().strip()
    start = 0
    end = N - 1
    while start <= end:
        i = int((start + end) / 2)
        if temp == hear[i]:
            res.append(temp)
            break
        elif temp < hear[i]:
            end = i - 1
        elif temp > hear[i]:
            start = i + 1
res.sort()
print(len(res))
for i in range(len(res)): print(res[i])


# 1269

import sys
input = sys.stdin.readline
Anum, Bnum = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
A.sort()
B.sort()
AandB = 0
for bnum in range(Bnum):
    start = 0
    end = Anum - 1
    while start <= end:
        i = int((start + end) / 2)
        if B[bnum] == A[i]:
            AandB += 1
            break
        elif B[bnum] < A[i]:
            end = i - 1
        elif B[bnum] > A[i]:
            start = i + 1
print(Anum+Bnum-2*AandB)


# 11478

import sys
input = sys.stdin.readline
S = input().strip()
res = set()
for i in range(len(S)):
    for j in range(i, len(S)):
        res.add(S[i:j+1])
print(len(res))


# -----------------------------------
# fifteen


# 1085

import sys
input = sys.stdin.readline
x, y, w, h = map(int, input().split())
ans = min(x, y, abs(w-x), abs(h-y))
print(ans)


# 3009

import sys
input = sys.stdin.readline
x = []
y = []
for i in range(3):
    X, Y = map(int, input().split())
    if X in x: x.remove(X)
    else: x.append(X)
    if Y in y: y.remove(Y)
    else: y.append(Y)
print(x[0], y[0])


# 2477

import sys
input = sys.stdin.readline
K = int(input())
order = []
max_ver = 0
idx_ver = -1
max_hor = 0
idx_hor = -1
for i in range(6):
    a, b = map(int, input().split())
    if a == 1 or a == 2:
        if b>max_ver:
            max_ver = b
            idx_ver = i
    else:
        if b>max_hor:
            max_hor = b
            idx_hor = i
    order.append(b)
min1 = order[(idx_ver+3)%6]
min2 = order[(idx_hor+3)%6]
print(K*(max_ver*max_hor-min1*min2))


# 1002

import sys
input = sys.stdin.readline
T = int(input())
for _ in range(T):
    x1, y1, r1, x2, y2, r2 = map(int, input().split())
    if r1>=r2:
        R = r1
        r = r2
    else:   # r1<r2
        R = r2
        r = r1
    dis = ((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))**0.5
    if x1==x2 and y1==y2 and r1==r2: print(-1)
    elif dis+r<R or dis>r+R: print(0)
    elif dis+r==R or dis==r+R: print(1)
    elif dis+r>R and dis<R+r: print(2)


# 1004

def dis(x1, y1, x2, y2):
    return ((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))**0.5

import sys
input = sys.stdin.readline
T = int(input())
for _ in range(T):
    x1, y1, x2, y2 = map(int, input().split())
    n = int(input())
    res = 0
    for _ in range(n):
        cx, cy, r = map(int, input().split())
        one, two = False, False
        if dis(x1, y1, cx, cy)<r: one = True
        if dis(x2, y2, cx, cy)<r: two = True
        if one==True and two==False: res += 1
        elif one==False and two==True: res += 1
    print(res)
"""

# -----------------------------------
# sixteen


# 5086

import sys
input = sys.stdin.readline



