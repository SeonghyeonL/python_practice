
"""
# -----------------------------------
# one (입출력과 사칙연산)


# 2557

print('Hello World!')


# 1000

ab = input()
a = int(ab[0])
b = int(ab[2])
print(a+b)


# 1001

ab = input()
a = int(ab[0])
b = int(ab[2])
print(a-b)


# 10998

ab = input()
a = int(ab[0])
b = int(ab[2])
print(a*b)


# 1008

ab = input()
a = int(ab[0])
b = int(ab[2])
print(a/b)


# 10869

a, b = map(int, input().split())
print(a + b)
print(a - b)
print(a * b)
print(a // b)
print(a % b)


# 10926

a = input()
print(a+"??!")


# 18108

a = int(input())
print(a-(2541-1998))


# 10430

a, b, c = map(int, input().split())
print((a+b)%c)
print(((a%c)+(b%c))%c)
print((a*b)%c)
print(((a%c)*(b%c))%c)


# 2588

a = int(input())
b = input()
b1, b2, b3 = int(b[2]), int(b[1]), int(b[0])
b = int(b)
print(a*b1)
print(a*b2)
print(a*b3)
print(a*b)


# 11382

a, b, c = map(int, input().split())
print(a+b+c)


# 10171

print('\\    /\\')
print(' )  ( \')')
print('(  /  )')
print(' \\(__)|')


# 10172

print('|\\_/|')
print('|q p|   /}')
print('( 0 )\"\"\"\\')
print('|\"^\"`    |')
print('||_/=\\\\__|')


# -----------------------------------
# two (조건문)


# 1330

a, b = map(int, input().split())
if a>b: print('>')
elif a<b: print('<')
elif a==b: print('==')


# 9498

a = int(input())
if a>=90: print('A')
elif a>=80: print('B')
elif a>=70: print('C')
elif a>=60: print('D')
else: print('F')


# 2753

# 윤년 1 / 아니면 0
# 윤년 = 4의 배수 O & 100의 배수 X or 400의 배수 O
y = int(input())
if y%400==0: print(1)
elif y%4==0 and y%100!=0: print(1)
else: print(0)


# 14681

x = int(input())
y = int(input())
if x>0 and y>0: print(1)
elif x<0 and y>0: print(2)
elif x<0 and y<0: print(3)
else: print(4)


# 2884

h, m = map(int, input().split())
m -= 45
if m<0:
    h -= 1
    m += 60
if h<0:
    h += 24
print(h, m)


# 2525

a, b = map(int, input().split())
c = int(input())
b += c
if b>59:
    a = a + b//60
    b = b%60
if a>23:
    a -= 24
print(a, b)

# 2480

a, b, c = map(int, input().split())
if a==b and b==c:
    print(10000+1000*a)
elif a==b:
    print(1000+100*a)
elif b==c:
    print(1000+100*b)
elif c==a:
    print(1000+100*c)
else:
    if a>b and a>c: print(100*a)
    elif b>c and b>a: print(100*b)
    else: print(100*c)


# -----------------------------------
# three (반복문)


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


