
"""
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
"""

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

