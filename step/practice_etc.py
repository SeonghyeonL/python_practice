
# 단기간 성장

"""
# 12865

import sys
input = sys.stdin.readline
N, K = map(int, input().split())
things = [(0, 0)]
for _ in range(N):
    W, V = map(int, input().split())
    things.append((W, V))
dp = [[0 for _ in range(K + 1)] for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(1, K + 1):
        if j >= things[i][0]:
            dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - things[i][0]] + things[i][1])
        else:  # j < things[i][0]
            dp[i][j] = dp[i - 1][j]
print(dp[-1][-1])
"""

# 1655

import sys
input = sys.stdin.readline
N = int(input())
for _ in range(N):
    n = int(input())








# https://www.acmicpc.net/workbook/view/4349