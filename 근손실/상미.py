import sys
input = sys.stdin.readline
from itertools import permutations

N, K = map(int, input().split())
added = list(map(int, input().split()))
addedC = list(permutations(added, N))
cnt = 0

for i in addedC:
    tmp = 0
    flag = True
    for n in range(N):
        tmp += i[n]
        tmp -= K
        if tmp < 0:
            flag = False
            break
    if flag:
        cnt += 1
print(cnt)