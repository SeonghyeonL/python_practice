
"""
# -----------------------------------
# missing


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



# -----------------------------------
# fourteen (기본 수학 1)



# 2355

import sys
input = sys.stdin.readline
A, B = map(int, input().split())
if A >= 0 and B >= 0:
    if A <= B: print(B * (B + 1) // 2 - (A - 1) * A // 2)
    else: print(A * (A + 1) // 2 - (B - 1) * B // 2)
elif A >= 0 and B < 0:
    print(A * (A + 1) // 2 - (- B) * (- B + 1) // 2)
elif A < 0 and B >= 0:
    print(B * (B + 1) // 2 - (- A) * (- A + 1) // 2)
else:
    A = - A
    B = - B
    if A <= B: print(- (B * (B + 1) // 2 - (A - 1) * A // 2))
    else: print(- (A * (A + 1) // 2 - (B - 1) * B // 2))




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



