
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
"""

# 10250

import sys
input = sys.stdin.readline



# https://www.acmicpc.net/step/8
