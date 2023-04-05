
n = 4
x1 = [1, 3, 3, 1]
y1 = [1, 1, 3, 3]
m = 12
x2 = [0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 1, 1]
y2 = [1, 2, 2, 3, 3, 2, 2, 1, 1, 0, 0, 1]

block1 = [[] for _ in range(4)]

for i in range(n):
    if i < n - 1: j = i + 1
    else: j = 0
    block1[0].append((x1[j]-x1[i], y1[j]-y1[i]))
    block1[1].append((-y1[j]+y1[i], x1[j]-x1[i]))
    block1[2].append((-x1[j] + x1[i], -y1[j] + y1[i]))
    block1[3].append((y1[j] - y1[i], -x1[j] + x1[i]))

block2 = []

for i in range(m):
    if i < m - 1: j = i + 1
    else: j = 0
    block2.append((x2[j]-x2[i], y2[j]-y2[i], abs(x2[j]-x2[i])+abs(y2[j]-y2[i])))

findlong = [[[[0 for _ in range(m+1)] for _ in range(n+1)] for _ in range(n)] for _ in range(4)]
maxlength = 0

for r in range(4):
    for i in range(n):
        nblock1 = block1[r][i:] + block1[r][:i]
        for a in range(1, n+1):
            for b in range(1, m+1):
                if nblock1[a-1][0] == block2[b-1][0] and nblock1[a-1][1] == block2[b-1][1]:
                    findlong[r][i][a][b] = findlong[r][i][a-1][b-1] + block2[b-1][2]
                    if findlong[r][i][a][b] > maxlength:
                        maxlength = findlong[r][i][a][b]

answer = maxlength
print(answer)