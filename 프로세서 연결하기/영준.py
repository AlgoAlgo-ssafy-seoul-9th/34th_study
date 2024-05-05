di = [0, 1, 0, -1]
dj = [1, 0, -1, 0]
def connect(cn, d, N):   # cn 코어번호, d 방향, N 셀크기
    cnt = 0
    ni, nj = core[cn][0]+di[d], core[cn][1]+dj[d]    # 코어 옆 셀부터 전선 연결
    while 0<=ni<N and 0<=nj<N and cell[ni][nj]==0:
        ni, nj = ni+di[d], nj+dj[d]
    if ni<0 or ni==N or nj<0 or nj==N:              # 연결되면
        ni, nj = core[cn][0] + di[d], core[cn][1] + dj[d]  # 코어 옆 셀부터 전선 연결
        while 0 <= ni < N and 0 <= nj < N and cell[ni][nj] == 0:
            cnt += 1                    # 전선 길이
            cell[ni][nj] = 2
            ni, nj = ni + di[d], nj + dj[d]
    return cnt

def disconnect(cn, d, N):
    ni, nj = core[cn][0] + di[d], core[cn][1] + dj[d]  # 코어 옆 셀부터 전선 지우기
    while 0 <= ni < N and 0 <= nj < N:
        cell[ni][nj] = 0
        ni, nj = ni + di[d], nj + dj[d]

def mexynos(k, c, w):  # k 남은 코어수, c전원이 공급된 코어 수, w i-1까지 연결한 길이
    global min_wire
    global max_core
    if k==0:    # 더이상 남은 코어가 없는 경우
        if max_core < c:
            max_core = c
            min_wire = w
        elif max_core == c and min_wire > w:
            min_wire = w
    elif max_core > c+k:      # 남은 코어를 다 연결해도 부족하면 중단
        return
    else:
        for d in range(4):      # 4개 방향으로 연결시도
            tmp = connect(k-1, d, N)    # k-1번 코어, 전원 연결 성공한 경우 길이 리턴, 실패시 0
            if tmp:
                mexynos(k-1, c+1, w+tmp)
                disconnect(k-1, d, N)     # d방향 연결 삭제
            # else:
            #     mexynos(k - 1, c, w)  # 연결 못하는 경우가 여러번이면 중복되므로
        mexynos(k-1, c, w)      # 연결 안하는 경우 1회 추가

T = int(input())
for tc in range(1, T+1):
    N = int(input())
    cell = [list(map(int, input().split())) for _ in range(N)]
    core = []
    core_cnt = 0
    for i in range(1, N-1):     # 가장 자리 제외
        for j in range(1, N-1):
            if cell[i][j]:
                core.append((i, j))
                core_cnt += 1

    min_wire = 1000000
    max_core = 0
    mexynos(core_cnt, 0, 0)
    print(f'#{tc} {min_wire}')
