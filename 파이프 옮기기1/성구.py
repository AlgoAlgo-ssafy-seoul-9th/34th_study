# 17070 파이프 옮기기 1
import sys
input = sys.stdin.readline


def main():
    N = int(input())
    home = [list(map(int, input().split())) for _ in range(N)]
    # [가로, 세로, 대각선]
    if home[-1][-1] or home[0][2]:
        print(0)
        return
    dp = [[[0,0,0] for _ in range(N-1)] for _ in range(N)]
    dp[0][0][0] = 1
    # 초기 dp 설정(1번째 라인 채우기)
    for i in range(2, N):
        if not home[0][i]:
            dp[0][i-1][0] += dp[0][i-2][0]

    # 현재 위치로 올 수 있는 위치 체크
    for i in range(1, N):
        for j in range(2, N):
            # 내 위치가 1이 아니면
            if not home[i][j]:
                # 가로 방향 올 수 있음(가로, 대각선)
                dp[i][j-1][0] += dp[i][j-2][0] + dp[i][j-2][2]
                # 세로 방향 올 수 있음(세로, 대각선)
                dp[i][j-1][1] += dp[i-1][j-1][1] + dp[i-1][j-1][2]
            
            # 내 위치와 위, 아래가  1이 아니명
            if not (home[i][j] or home[i-1][j] or home[i][j-1]):
                # 대각선 방향 올 수 있음(가로, 세로, 대각선)
                dp[i][j-1][2] += sum(dp[i-1][j-2])
    # debugs
    # [print(dp[o]) for o in range(N)]

    print(sum(dp[-1][-1]))
            
    return


if __name__ == "__main__":
    main()