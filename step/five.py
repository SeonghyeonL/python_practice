
"""
# 11654

import sys
input = sys.stdin.readline
given = input().strip()
print(ord(given))


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
"""

# 11718 - 2

import sys
input = sys.stdin.readline
while line:=input().strip():
    print(line)
