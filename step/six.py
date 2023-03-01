
"""
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


