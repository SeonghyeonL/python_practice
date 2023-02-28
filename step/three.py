
"""
# 2739

n = int(input())
for i in range(1, 10):
    print(n,'*',i,'=',n*i)


# 10950

t = int(input())
for i in range(t):
    a, b = map(int, input().split())
    print(a+b)


# 8393

n = int(input())
# n*(n+1)/2
print(int(n*(n+1)/2))


# 25304

x = int(input())
n = int(input())
sum = 0
for i in range(n):
    a, b = map(int, input().split())
    sum += a*b
if sum==x: print('Yes')
else: print('No')


# 25314

n = int(input())
res = ''
for i in range(int(n/4)):
    res += 'long '
res += 'int'
print(res)


# 15552

import sys
input = sys.stdin.readline
t = int(input())
for i in range(t):
    a, b = map(int, input().split())
    print(a+b)


# 11021

import sys
input = sys.stdin.readline
t = int(input())
for i in range(t):
    a, b = map(int, input().split())
    print("Case #%d: %d" %(i+1, a+b))


# 11022

import sys
input = sys.stdin.readline
t = int(input())
for i in range(t):
    a, b = map(int, input().split())
    print("Case #%d: %d + %d = %d" %(i+1, a, b, a+b))


# 2438

import sys
input = sys.stdin.readline
n = int(input())
for i in range(1, n+1):
    res = ""
    for j in range(i):
        res += "*"
    print(res)


# 2439

import sys
input = sys.stdin.readline
n = int(input())
for i in range(1, n+1):
    res = ""
    for j in range(n-i):
        res += " "
    for k in range(i):
        res += "*"
    print(res)


# 10952

import sys
input = sys.stdin.readline
a, b = map(int, input().split())
while a!=0 and b!=0:
    print(a+b)
    a, b = map(int, input().split())
"""

# 10951

import sys
input = sys.stdin.readline
while True:
    try:
        a, b = map(int, input().split())
        print(a+b)
    except:
        break

