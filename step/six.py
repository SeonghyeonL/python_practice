
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
"""

# 10812

import sys
input = sys.stdin.readline
N, M = map(int, input().split())
ns = ""
for n in range(N):
    ns += str(n+1)
for m in range(M):
    i, j, k = map(int, input().split())
    front = ns[:i-1]
    btom = ns[i-1:k-1]
    mtoe = ns[k-1:j]
    back = ns[j:]
    ns = front+mtoe+btom+back
for n in range(N):
    print(ns[n], end=" ")





