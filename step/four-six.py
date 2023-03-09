
"""
# -----------------------------------
# four


# 10807

import sys
input = sys.stdin.readline
n = int(input())
ns = list(map(int, input().split()))
v = int(input())
cnt = 0
for i in range(n):
    if ns[i]==v: cnt+=1
print(cnt)


# 10871

import sys
input = sys.stdin.readline
n, x = map(int, input().split())
a = list(map(int, input().split()))
for i in range(n):
    if a[i]<x: print(a[i], end=" ")


# 10818

import sys
input = sys.stdin.readline
n = int(input())
ns = list(map(int, input().split()))
min = ns[0]
max = ns[0]
for i in range(n):
    if ns[i]<min: min = ns[i]
    elif ns[i]>max: max = ns[i]
print(min, max)


# 2562

import sys
input = sys.stdin.readline
max = 0
idx = -1
for i in range(1, 10):
    temp = int(input())
    if temp>max:
        max = temp
        idx = i
print(max)
print(idx)


# 10810

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
b = []
for n in range(N):
    b.append(0)
for m in range(M):
    i, j, k = map(int, input().split())
    for idx in range(i-1, j):
        b[idx] = k
for n in range(N):
    print(b[n], end=" ")


# 10813

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
b = []
temp = 0
for n in range(N):
    b.append(n+1)
for m in range(M):
    i, j = map(int, input().split())
    temp = b[i-1]
    b[i-1] = b[j-1]
    b[j-1] = temp
for n in range(N):
    print(b[n], end=" ")


# 5597

import sys
input = sys.stdin.readline
b = []
temp = 0
for i in range(30):
    b.append(0)
for i in range(28):
    temp = int(input())
    b[temp-1] = 1
for i in range(30):
    if b[i]==0: print(i+1)


# 3052

import sys
input = sys.stdin.readline
a = []
temp = 0
cnt = 0
for i in range(42):
    a.append(0)
for i in range(10):
    temp = int(input())
    a[temp%42] = 1
for i in range(42):
    if a[i]==1: cnt += 1
print(cnt)


# 10811

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
a = []
temp = 0
for n in range(N):
    a.append(n+1)
for m in range(M):
    i, j = map(int, input().split())
    while i < j:
        temp = a[i-1]
        a[i-1] = a[j-1]
        a[j-1] = temp
        i += 1
        j -= 1
for n in range(N):
    print(a[n], end=" ")


# 1546

import sys
input = sys.stdin.readline
n = int(input())
score = list(map(int, input().split()))
max = -1
sum = 0
for i in range(n):
    if score[i]>max: max = score[i]
for i in range(n):
    sum += score[i] / max * 100
print(sum/n)


# -----------------------------------
# five


# 27866

import sys
input = sys.stdin.readline
S = input().strip()
i = int(input())
print(S[i-1])


# 2743

import sys
input = sys.stdin.readline
given = input().strip()
print(len(given))


# 9086

import sys
input = sys.stdin.readline
T = int(input())
for i in range(T):
    res = ""
    given = input().strip()
    res += given[0]
    res += given[len(given)-1]
    print(res)


# 11654

import sys
input = sys.stdin.readline
given = input().strip()
print(ord(given))


# 11720

import sys
input = sys.stdin.readline
N = int(input())
given = input().strip()
sum = 0
for i in range(N):
    sum += int(given[i])
print(sum)


# 10809

import sys
input = sys.stdin.readline
S = input().strip()
a = []
temp = 0
for i in range(26):
    a.append(-1)
for i in range(len(S)):
    temp = ord(S[i])-ord('a')
    if a[temp] == -1:
        a[temp] = i
for i in range(26):
    print(a[i], end=" ")


# 2675

import sys
input = sys.stdin.readline
T = int(input())
for t in range(T):
    S = input().strip()
    R = int(S[0])
    S = S[2:]
    res = ""
    for s in range(len(S)):
        for r in range(R):
            res += S[s]
    print(res)


# 1157

import sys
input = sys.stdin.readline
word = input().strip()
a = []
for i in range(26):
    a.append(0)
# a = 97 / A = 65
for i in range(len(word)):
    temp = ord(word[i])
    if temp >= 97: temp -= 32
    a[temp - 65] += 1
max = -1
idx = -1
for i in range(26):
    if a[i] > max:
        max = a[i]
        idx = i
multi = False
for i in range(idx+1, 26):
    if a[i] == max: multi = True
if multi: print("?")
else: print(chr(idx+65))


# 1152

import sys
input = sys.stdin.readline
line = list(input().strip().split())
print(len(line))


# 2908

import sys
input = sys.stdin.readline
line = input().strip()
a, b = "", ""
a += line[2]
a += line[1]
a += line[0]
b += line[6]
b += line[5]
b += line[4]
a, b = int(a), int(b)
if a>b: print(a)
else: print(b)


# 5622

import sys
input = sys.stdin.readline
# ABC/DEF/GHI/JKL/MNO/PQRS/TUV/WXYZ
#  3   4   5   6   7   8    9   10
word = input().strip()
sum = 0
for i in range(len(word)):
    if word[i]<='C': sum += 3
    elif word[i]<='F': sum += 4
    elif word[i]<='I': sum += 5
    elif word[i]<='L': sum += 6
    elif word[i]<='O': sum += 7
    elif word[i]<='S': sum += 8
    elif word[i]<='V': sum += 9
    else: sum += 10
print(sum)


# 11718 - 1

while True:
    try:
        print(input())
    except EOFError:
        break


# 11718 - 2

import sys
input = sys.stdin.readline
while line:=input().strip():
    print(line)


# -----------------------------------
# six


# 25083

print("         ,r\'\"7")
print("r`-_   ,\'  ,/")
print(" \\. \". L_r\'")
print("   `~\\/")
print("      |")
print("      |")


# 3003

import sys
input = sys.stdin.readline
a, b, c, d, e, f = map(int, input().split())
# 1 1 2 2 2 8
print(1-a, 1-b, 2-c, 2-d, 2-e, 8-f)


# 2444

import sys
input = sys.stdin.readline
N = int(input())
n = 2 * N - 1
for i in range(1, N+1): print(" "*(N-i)+"*"*(2*i-1))
for i in range(1, N): print(" "*i+"*"*(2*(N-i)-1))


# 10812

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
ns = []
for n in range(N):
    ns.append(n+1)
for m in range(M):
    i, j, k = map(int, input().split())
    front = ns[:i-1]
    btom = ns[i-1:k-1]
    mtoe = ns[k-1:j]
    back = ns[j:]
    ns = front+mtoe+btom+back
for n in range(N):
    print(ns[n], end=" ")


# 10988

import sys
input = sys.stdin.readline
word = input().strip()
ok = True
l = len(word)
for i in range(l//2):
    if word[i]!=word[l-i-1]:
        ok = False
        break
if ok: print(1)
else: print(0)


# 4344

import sys
input = sys.stdin.readline
C = int(input())
for c in range(C):
    line = list(map(int, input().split()))
    num = line[0]
    grade = line[1:]
    cnt = 0
    sum = 0
    for i in range(len(grade)): sum += grade[i]
    avg = sum / num
    for i in range(len(grade)):
        if grade[i]>avg: cnt += 1
    print("%.3f" %(cnt/num*100), end="")
    print("%")


# 2941

import sys
input = sys.stdin.readline
word = input().strip()
cnt = 0
for i in range(len(word)):
    if i<len(word)-1:
        if word[i:i+2] in ('c=', 'c-', 'd-', 'lj', 'nj', 's=', 'z='):
            continue
    if i<len(word)-2:
        if word[i:i+3]=='dz=':
            continue
    cnt += 1
print(cnt)


# 1316

import sys
input = sys.stdin.readline
N = int(input())
cnt = 0
for n in range(N):
    word = input().strip()
    tf = []
    for i in range(26): tf.append(False)
    mem = ""
    for i in range(len(word)):
        if i == 0:
            mem = word[0]
            tf[ord(word[0]) - ord('a')] = True
        elif mem != word[i]:
            if tf[ord(word[i])-ord('a')]: break
            else:
                mem = word[i]
                tf[ord(word[i])-ord('a')] = True
        if i == len(word) - 1: cnt += 1
print(cnt)
"""

# 25206

import sys
input = sys.stdin.readline
time_sum = 0
sum = 0
for i in range(20):
    name, time, grade = input().strip().split()
    int_time = int(time[0])
    time_sum += int_time
    if grade == 'A+': sum += 4.5 * int_time
    elif grade == 'A0': sum += 4.0 * int_time
    elif grade == 'B+': sum += 3.5 * int_time
    elif grade == 'B0': sum += 3.0 * int_time
    elif grade == 'C+': sum += 2.5 * int_time
    elif grade == 'C0': sum += 2.0 * int_time
    elif grade == 'D+': sum += 1.5 * int_time
    elif grade == 'D0': sum += 1.0 * int_time
    elif grade == 'P': time_sum -= int_time
print(sum / time_sum)


