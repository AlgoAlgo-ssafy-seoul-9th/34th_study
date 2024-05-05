# 18429_근손실_muscle loss
import sys
input = sys.stdin.readline

def bt(cnt, score):
    global ans
    if cnt == N:
        ans += 1
        return

    for i in range(N):
        if not visited[i] and score+exercise[i]-K >= 500:
            visited[i] = 1
            bt(cnt+1, score+exercise[i]-K)
            visited[i] = 0


N, K = map(int, input().split())
ans = 0

exercise = list(map(int, input().split()))
visited = [0]*(N+1)

bt(0, 500)

print(ans)