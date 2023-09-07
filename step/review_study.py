"""
#

line = [[2, -1, 4], [-2, -1, 4], [0, -1, 1], [5, -8, -12], [5, 8, 12]]

star = []
for i in range(len(line) - 1):
    A, B, C = line[i][0], line[i][1], line[i][2]
    for j in range(i + 1, len(line)):
        a, b, c = line[j][0], line[j][1], line[j][2]
        if A * b - B * a == 0: continue  # 평행 또는 일치
        x = (B * c - C * b) / (A * b - B * a)
        y = (C * a - A * c) / (A * b - B * a)
        if x == int(x) and y == int(y):
            star.append([int(x), int(y)])

max_x, min_x, max_y, min_y = star[0][0], star[0][0], star[0][1], star[0][1]
for x, y in star:
    max_x = max(max_x, x)
    min_x = min(min_x, x)
    max_y = max(max_y, y)
    min_y = min(min_y, y)

answer_a = [["." for _ in range(max_x - min_x + 1)] for _ in range(max_y - min_y + 1)]
for i in range(len(star)):
    answer_a[max_y - star[i][1]][star[i][0] - min_x] = "*"
answer = []
for i in range(len(answer_a)):
    now = ""
    for j in range(len(answer_a[i])):
        now += answer_a[i][j]
    answer.append(now)

print(answer)


#

babbling = ["aya", "yee", "u", "maa", "wyeoo"]
answer = 0
for bab in babbling:
    bab_list = list(bab)
    last = len(bab_list)
    idx = 0
    while True:
        if idx == last:
            answer += 1
            break
        if idx <= last - 3:
            if bab_list[idx] == "a" and bab_list[idx + 1] == "y" and bab_list[idx + 2] == "a":
                idx += 3
                continue
            elif bab_list[idx] == "w" and bab_list[idx + 1] == "o" and bab_list[idx + 2] == "o":
                idx += 3
                continue
        if idx <= last - 2:
            if bab_list[idx] == "y" and bab_list[idx + 1] == "e":
                idx += 2
                continue
            elif bab_list[idx] == "m" and bab_list[idx + 1] == "a":
                idx += 2
                continue
        break

print(answer)


#

def solution(A, B):
    for cnt in range(len(A)):
        if A == B:
            return cnt
        A = A[-1] + A[:-1]

    return -1


#

def solution(A, B):
    A, B = list(A), list(B)

    for cnt in range(len(A)):
        if A == B:
            return cnt

        A.insert(0, A.pop())

    return -1


#

from collections import deque

def solution(A, B):
    A, B = deque(A), deque(B)

    for cnt in range(len(A)):
        if A == B:
            return cnt

        A.rotate()

    return -1


#

def solution(A,B):
    BB = B*2
    return BB.find(A)


#

board = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]
n = len(board)
for i in range(n):
    for j in range(n):
        if board[i][j] == 1:
            if i - 1 >= 0 and board[i - 1][j] == 0: board[i - 1][j] = 2
            if i + 1 < n and board[i + 1][j] == 0: board[i + 1][j] = 2
            if j - 1 >= 0 and board[i][j - 1] == 0: board[i][j - 1] = 2
            if j + 1 < n and board[i][j + 1] == 0: board[i][j + 1] = 2
            if i - 1 >= 0 and j - 1 >= 0 and board[i - 1][j - 1] == 0:
                board[i - 1][j - 1] = 2
            if i - 1 >= 0 and j + 1 < n and board[i - 1][j + 1] == 0:
                board[i - 1][j + 1] = 2
            if i + 1 < n and j - 1 >= 0 and board[i + 1][j - 1] == 0:
                board[i + 1][j - 1] = 2
            if i + 1 < n and j + 1 < n and board[i + 1][j + 1] == 0:
                board[i + 1][j + 1] = 2
answer = 0
for i in range(n):
    for j in range(n):
        if board[i][j] == 0: answer += 1
print(answer)
"""

#


