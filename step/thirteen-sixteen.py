
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
"""

# -----------------------------------
# fourteen


# 10815

import sys
input = sys.stdin.readline



# https://www.acmicpc.net/step/49


