def f2(N):
    for i in range(1, N+1):
        for j in range(2, N+1):
            if(rm[i][j] == 0):
                # 수평 이동으로 진입
                dp_left[i][j] = dp_left[i][j-1] + dp_diag[i][j-1] #왼쪽칸에 수평 또는 대각 진입
                # 대각선 이동 진입
                if(rm[i-1][j]==0 and rm[i][j-1]==0): #위와 왼쪽에 기둥이 없는 경우
                    dp_diag[i][j] = dp_diag[i-1][j-1]+dp_up[i-1][j-1]+dp_left[i-1][j-1]
                # 수직 이동 진입
                dp_up[i][j] = dp_up[i-1][j]+dp_diag[i-1][j]


    return dp_up[N][N]+ dp_diag[N][N]+dp_left[N][N]


N = int(input())
rm = [[0]*(N+1)]
for i in range(N):
    rm.append([0]+list(map(int, input().split())))
dp_up = [[0]*(N+1) for i in range(N+1)]
dp_left = [[0]*(N+1) for i in range(N+1)]
dp_diag = [[0]*(N+1) for i in range(N+1)]

dp_left[1][1] = 1
dp_up[1][1] = -1
cnt = 0

print(f2(N))
