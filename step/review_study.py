import sys

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