
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
"""



