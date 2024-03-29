
"""
# -----------------------------------
# fourteen (집합과 맵)


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


# 7785

import sys
input = sys.stdin.readline
n = int(input())
company = dict()
for _ in range(n):
    name, log = input().strip().split()
    if log == "enter":
        company[name] = 1
    elif log == "leave":
        company.pop(name)
company = list(company.keys())
company.sort(reverse=True)
print('\n'.join(map(str, company)))


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
# fifteen (약수, 배수와 소수 2)


# 1934

import sys
input = sys.stdin.readline
T = int(input())
for _ in range(T):
    A, B = map(int, input().split())
    small = min(A, B)
    large = max(A, B)
    lst = []
    for i in range(1, int(small**0.5)+1):
        if small % i == 0:
            lst.append(i)
            if i != small / i:
                lst.append(small/i)
    lst.sort(reverse=True)
    for i in range(0, len(lst)):
        if large % lst[i] == 0:
            print(int(lst[i]*(A/lst[i])*(B/lst[i])))
            break


# 13241

import sys
input = sys.stdin.readline
A, B = map(int, input().split())
small = min(A, B)
large = max(A, B)
lst = []
for i in range(1, int(small**0.5)+1):
    if small % i == 0:
        lst.append(i)
        if i != small / i:
            lst.append(small/i)
lst.sort(reverse=True)
for i in range(0, len(lst)):
    if large % lst[i] == 0:
        print(int(lst[i]*(A/lst[i])*(B/lst[i])))
        break


# 1735

def findmax(a, b):
    small = min(a, b)
    large = max(a, b)
    temp = []
    for j in range(1, int(small ** 0.5) + 1):
        if small % j == 0:
            temp.append(j)
            if j != small / j:
                temp.append(small / j)
    temp.sort(reverse=True)
    for j in range(0, len(temp)):
        if large % temp[j] == 0:
            return temp[j]

import sys
input = sys.stdin.readline
A = list(map(int, input().split()))
B = list(map(int, input().split()))
found = findmax(A[1], B[1])
temp1 = int(found*(A[1]/found)*(B[1]/found))
temp2 = int(A[0]*(B[1]/found)+B[0]*(A[1]/found))
found2 = findmax(temp1, temp2)
temp1 = int(temp1/found2)
temp2 = int(temp2/found2)
print(temp2, temp1)


# 2485

def findmax(a, b):
    small = min(a, b)
    large = max(a, b)
    lst = []
    for j in range(1, int(small ** 0.5) + 1):
        if small % j == 0:
            lst.append(j)
            if j != small / j:
                lst.append(small / j)
    lst.sort(reverse=True)
    for j in range(0, len(lst)):
        if large % lst[j] == 0:
            return lst[j]

import sys
input = sys.stdin.readline
N = int(input())
tree = []
dis = []
for n in range(N):
    temp = int(input())
    tree.append(temp)
    if n > 0:
        dis.append(tree[n]-tree[n-1])
maxdis = 1000000000
for n in range(N-2):
    maxtemp = findmax(dis[n], dis[n+1])
    if maxtemp < maxdis: maxdis = maxtemp
res = 0
for n in range(N-1):
    res += dis[n]/maxdis - 1
print(int(res))


# 4134

def isprime(n):
    if n == 0 or n == 1:
        return False
    else:
        for i in range(2, int(n**0.5)+1):
            if n%i==0: return False
        return True

import sys
input = sys.stdin.readline
T = int(input())
for _ in range(T):
    n = int(input())
    while isprime(n) == False:
        n += 1
    print(n)


# 1929

def isprime(n):
    if n == 0 or n == 1:
        return False
    else:
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

import sys
input = sys.stdin.readline
M, N = map(int, input().split())
for mn in range(M, N+1):
    if isprime(mn):
        print(mn)


# 4948

def isprime(n):
    if n == 0 or n == 1:
        return False
    else:
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

import sys
input = sys.stdin.readline
n = int(input())
while n != 0:
    cnt = 0
    for m in range(n+1, 2*n+1):
        if isprime(m): cnt += 1
    print(cnt)
    n = int(input())


# 17103

import sys
input = sys.stdin.readline
T = int(input())
check = [False, False]
check += [True] * 1000000
for i in range(2, 1000001):
    if check[i] == True:
        for j in range(2*i, 1000001, i): check[j] = False
for _ in range(T):
    N = int(input())
    ans = 0
    for n in range(2, int(N/2)+1):
        if check[n] and check[N-n]: ans += 1
    print(ans)


# 13909

import sys
input = sys.stdin.readline
N = int(input())
ans = int(N ** 0.5)
print(ans)


# -----------------------------------
# sixteen (재귀) - 이후부터 수정 필요


# 10872

res = [1]
for i in range(12):
    res.append(0)

def fac(N):
    if res[N] != 0: return res[N]
    else: return fac(N-1)*N

import sys
input = sys.stdin.readline
N = int(input())
print(fac(N))


# 10870

res = [0, 1]
for i in range(20):
    res.append(-1)

def fib(N):
    if res[N] == -1: res[N] = fib(N-2)+fib(N-1)
    return res[N]

import sys
input = sys.stdin.readline
n = int(input())
print(fib(n))


# 25501

def recursion(s, l, r, cnt):
    cnt += 1
    if l >= r: return 1, cnt
    elif s[l] != s[r]: return 0, cnt
    else: return recursion(s, l+1, r-1, cnt)

def isPalindrome(s):
    return recursion(s, 0, len(s)-1, 0)

import sys
input = sys.stdin.readline
T = int(input())
for t in range(T):
    S = input().strip()
    temp = isPalindrome(S)
    print(temp[0], temp[1])


# 24060

def merge_sort(A, p, r, cnt):
    if p<r:
        q = int((p+r)/2)
        cnt1 = merge_sort(A, p, q, cnt)
        cnt2 = merge_sort(A, q+1, r, cnt1)
        cnt = merge(A, p, q, r, cnt2)
    return cnt

def merge(A, p, q, r, cnt):
    i, j = p, q+1
    temp = []
    while i<=q and j<=r:
        if A[i]<=A[j]:
            temp.append(A[i])
            i += 1
        else:
            temp.append(A[j])
            j += 1
    while i<=q:
        temp.append(A[i])
        i += 1
    while j<=r:
        temp.append(A[j])
        j += 1
    i, t = p, 0
    while i<=r:
        A[i] = temp[t]
        cnt += 1
        if cnt == K: print(temp[t])
        i += 1
        t += 1
    return cnt

import sys
input = sys.stdin.readline
N, K = map(int, input().split())
A = list(map(int, input().split()))
cnt = 0
cnt = merge_sort(A, 0, N-1, cnt)
if cnt < K: print(-1)


# 2447

def makestar(star, A, B, n):
    step = int(n/3)
    for a in range(A+step, A+2*step):
        for b in range(B+step, B+2*step):
            star[a][b] = 0
    if step>1:
        makestar(star, A, B, step)
        makestar(star, A, B+step, step)
        makestar(star, A, B+2*step, step)
        makestar(star, A+step, B, step)
        makestar(star, A+step, B+2*step, step)
        makestar(star, A+2*step, B, step)
        makestar(star, A+2*step, B+step, step)
        makestar(star, A+2*step, B+2*step, step)

import sys
input = sys.stdin.readline
N = int(input())
star = []
for i in range(N):
    temp = []
    for j in range(N):
        temp.append(1)
    star.append(temp)
makestar(star, 0, 0, N)
for i in range(N):
    for j in range(N):
        if star[i][j]==1: print("*", end="")
        else: print(" ", end="")
    print()


# 11729

def hanoi(now, to, temp, n):
    if n == 1:
        print(now, to)
    else:
        hanoi(now, temp, to, n-1)
        hanoi(now, to, temp, 1)
        hanoi(temp, to, now, n-1)

import sys
input = sys.stdin.readline
N = int(input())
sum = 1
for _ in range(1, N):
    sum = sum * 2 + 1
print(sum)
hanoi(1, 3, 2, N)


# -----------------------------------
# seventeen (백트래킹)


# 15649

def backtracking(num):
    if num == M:
        print(' '.join(map(str, result)))
    else:
        for i in range(1, 1+N):
            if visited[i] == False:
                visited[i] = True
                result.append(i)
                backtracking(num+1)
                visited[i] = False
                result.pop()

import sys
sys.setrecursionlimit(10**6)    # 재귀 깊이 제한 늘려서 런타임 에러 방지
input = sys.stdin.readline
N, M = map(int, input().split())
result = []
visited = [False] * (N+1)
backtracking(0)


# 15650

def backtracking(cnt, start):
    if cnt == M:
        print(' '.join(map(str, result)))
    else:
        for i in range(start, 1+N):
            if visited[i] == False:
                visited[i] = True
                result.append(i)
                backtracking(cnt+1, i+1)
                visited[i] = False
                result.pop()

import sys
sys.setrecursionlimit(10**6)
input = sys.stdin.readline
N, M = map(int, input().split())
result = []
visited = [False] * (N+1)
backtracking(0, 1)


# 15651

def backtracking(cnt):
    if cnt == M:
        print(' '.join(map(str, result)))
    else:
        for i in range(1, N+1):
            result.append(i)
            backtracking(cnt+1)
            result.pop()

import sys
sys.setrecursionlimit(10**6)
input = sys.stdin.readline
N, M = map(int, input().split())
result = []
backtracking(0)


# 15652

def backtracking(cnt, start):
    if cnt == M:
        print(' '.join(map(str, result)))
    else:
        for i in range(start, N+1):
            result.append(i)
            backtracking(cnt+1, i)
            result.pop()

import sys
sys.setrecursionlimit(10**6)
input = sys.stdin.readline
N, M = map(int, input().split())
result = []
backtracking(0, 1)


# 9663

def backtracking(col):
    if col == N+1:
        global total
        total += 1
    else:
        for i in range(1, 1+N):
            if row[i] == False:
                chess[col] = i
                row[i] = True
                check = True
                for j in range(1, col):
                    if col-j == abs(chess[col]-chess[j]):
                        check = False
                        break
                if check == True:
                    backtracking(col+1)
                row[i] = False
            chess[col] = 0

import sys
input = sys.stdin.readline
N = int(input())
chess = [0] * (N+1)
row = [False] * (N+1)
total = 0
# NxN 체스판 위에 퀸 N개이므로 한 열에 하나씩만 두어야 함
# chess에 저장하는 건 해당 열의 행 값 (0 제외, 1 ~ 15)
backtracking(1)
print(total)


# 2580

def checkrowcol(x, y, a):
    for b in range(9):
        if total[x][b] == a or total[b][y] == a:
            return False
    return True

def check3by3(x, y, a):
    x = (x // 3) * 3    # 0 1 2 / 3 4 5 / 6 7 8
    y = (y // 3) * 3
    for b in range(3):
        for c in range(3):
            if total[x+b][y+c] == a:
                return False
    return True

def backtracking(idx):
    if idx == len(zero):
        for a in range(9):
            print(' '.join(map(str, total[a])))
        exit(0)
    else:
        x = zero[idx][0]
        y = zero[idx][1]
        okay = True
        for a in range(10):
            if checkrowcol(x, y, a) and check3by3(x, y, a):
                total[x][y] = a
                backtracking(idx+1)
                total[x][y] = 0

import sys
input = sys.stdin.readline
zero = []   # [row, col]
total = []
for i in range(9):
    temp = list(map(int, input().split()))
    total.append(temp)
    for j in range(9):
        if temp[j] == 0:
            zero.append([i, j])
backtracking(0)


# 14888

def cal(idx, temp):
    if idx == N:
        global max
        global min
        if temp > max: max = temp
        if temp < min: min = temp
    else:
        if num[0] > 0:
            num[0] -= 1
            cal(idx+1, temp+A[idx])
            num[0] += 1
        if num[1] > 0:
            num[1] -= 1
            cal(idx + 1, temp - A[idx])
            num[1] += 1
        if num[2] > 0:
            num[2] -= 1
            cal(idx + 1, temp * A[idx])
            num[2] += 1
        if num[3] > 0:
            num[3] -= 1
            if temp < 0:
                cal(idx + 1, -1 * (abs(temp) // A[idx]))
            else:
                cal(idx + 1, abs(temp) // A[idx])
            num[3] += 1

import sys
input = sys.stdin.readline
N = int(input())
A = list(map(int, input().split()))
num = list(map(int, input().split()))
max = -1000000000
min = 1000000000
cal(1, A[0])
print(max)
print(min)


# 14889

def cal(idx):
    if idx == N:
        global min
        gradeA, gradeB = 0, 0
        for i in range(int(N/2)):
            for j in range(i+1, int(N/2)):
                gradeA += S[teamA[i]][teamA[j]] + S[teamA[j]][teamA[i]]
                gradeB += S[teamB[i]][teamB[j]] + S[teamB[j]][teamB[i]]
        if abs(gradeA-gradeB) < min: min = abs(gradeA-gradeB)
    else:
        if len(teamA) < N/2:
            teamA.append(idx)
            cal(idx+1)
            teamA.pop(-1)
        if len(teamB) < N/2:
            teamB.append(idx)
            cal(idx+1)
            teamB.pop(-1)

import sys
input = sys.stdin.readline
N = int(input())
S = []
for _ in range(N):
    temp = list(map(int, input().split()))
    S.append(temp)
min = 100
teamA = []
teamB = []
cal(0)
print(min)
"""


