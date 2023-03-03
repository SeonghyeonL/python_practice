
"""
# 2292

import sys
input = sys.stdin.readline
N = int(input())
# 1(1) / 6(7) / 12(19) / 18(37)
room = 0
i = 0
while N > 0:
    if i == 0: N -= 1
    else: N -= 6 * i
    room += 1
    i += 1
print(room)


# 1193

import sys
input = sys.stdin.readline
X = int(input())
# 1(1) / 2(3) / 3(6) / 4(10)
i = 1
while X > 0:
    X -= i
    i += 1
i -= 1
X += i
if i%2==0: print("%d/%d" %(X, i-X+1))
else: print("%d/%d" %(i-X+1, X))


# 2869

import sys
input = sys.stdin.readline
A, B, V = map(int, input().split())
day = 1
if V > A:
    temp = (V-A)//(A-B)
    if (V-A)%(A-B)==0: day = temp + 1
    else: day = temp + 2
print(day)


# 10250

import sys
input = sys.stdin.readline
T = int(input())
for t in range(T):
    H, W, N = map(int, input().split())
    h = (N-1) % H + 1
    w = (N-1) // H + 1
    print(h*100+w)


# 2775

a = []
for i in range(15):
    list = []
    list.append(1)
    if i == 0:
        for j in range(2, 15):
            list.append(j)
    else:
        for j in range(2, 15):
            list.append(0)
    a.append(list)

def apart(k, n):
    if a[k][n] == 0:
        a[k][n] = apart(k, n-1) + apart(k-1, n)
    return a[k][n]

import sys
input = sys.stdin.readline
T = int(input())
for t in range(T):
    k = int(input())    # k층
    n = int(input())    # n호
    # 3층) 1 / 5 / 15 / 35 / 70
    # 2층) 1 / 4 / 10 / 20 / 35
    # 1층) 1 / 3 /  6 / 10 / 15
    # 0층) 1 / 2 /  3 /  4 /  5
    print(apart(k, n-1))


# 2839

import sys
input = sys.stdin.readline
N = int(input())
five = N//5
cant = False
while True:
    if (N-five*5)%3 == 0: break
    else: five -= 1
    if five<0:
        cant = True
        break
if cant: print(-1)
else: print(int(five+(N-five*5)/3))
"""

# 10757

import sys
input = sys.stdin.readline
A, B = map(int, input().split())
print(A+B)

