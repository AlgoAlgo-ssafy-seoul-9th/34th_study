# 1043_거짓말_lying
import sys
from collections import deque
input = sys.stdin.readline

N, M = map(int, input().split())

n, *n_lst = map(int, input().split())

ans = 0
know_the_truth = {}

for n in n_lst:
    if n not in know_the_truth.keys():
        know_the_truth[n] = True

adjL = [[] for _ in range(N+1)]
visited = [0]*(N+1)
party = []

for _ in range(M):
    p, *p_lst = map(int, input().split())
    party.append(p_lst)

    for i in range(p):
        for j in range(p):
            if i != j:
                adjL[p_lst[i]].append(p_lst[j])

q = deque()
for t in know_the_truth.keys():
    visited[t] = 1
    q.append(t)

while q:
    now = q.popleft()

    for node in adjL[now]:
        if not visited[node]:
            visited[node] = 1
            q.append(node)

for p in party:
    for node in p:
        if visited[node]:
            break
    else:
        ans += 1

print(ans)