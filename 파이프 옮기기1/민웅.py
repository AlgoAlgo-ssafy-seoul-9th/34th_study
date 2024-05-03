# 17070_파이프옮기기1_pipe1
import sys
input = sys.stdin.readline

N = int(input())
field = [list(map(int, input().split())) for _ in range(N)]

ans = 0
if field[-1][-1] == 1:
    print(ans)
else:
    dp = [[[0, 0, 0] for _ in range(N)] for _ in range(N)]
    dp[0][1][0] = 1
    for i in range(N):
        for j in range(1, N):
            if j+1 < N and field[i][j+1] != 1:
                dp[i][j+1][0] += (dp[i][j][0] + dp[i][j][2])

            if i+1 < N and field[i+1][j] != 1:
                dp[i+1][j][1] += (dp[i][j][1] + dp[i][j][2])

            if i+1 < N and j+1 < N:
                if field[i+1][j+1] != 1 and field[i+1][j] != 1 and field[i][j+1] != 1:
                    dp[i+1][j+1][2] = sum(dp[i][j])

    print(sum(dp[-1][-1]))