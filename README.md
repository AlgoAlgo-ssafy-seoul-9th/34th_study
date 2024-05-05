# 34th_study

알고리즘 스터디 34주차

<br/>

# 이번주 스터디 문제

<details markdown="1" open>
<summary>접기/펼치기</summary>

<br/>

## [근손실](https://www.acmicpc.net/problem/18429)

### [민웅](./근손실/민웅.py)

```py

```

### [상미](./근손실/상미.py)

```py

```

### [성구](./근손실/성구.py)

```py

```

### [영준](./근손실/영준.py)

```py

```

<br/>

## [전구와 스위치](https://www.acmicpc.net/problem/2138)

### [민웅](./전구와%20스위치/민웅.py)

```py

```

### [상미](./전구와%20스위치/상미.py)

```py

```

### [성구](./전구와%20스위치/성구.py)

```py

```

### [영준](./전구와%20스위치/영준.py)

```py

```

<br/>

## [거짓말](https://www.acmicpc.net/problem/1043)

### [민웅](./거짓말/민웅.py)

```py

```

### [상미](./거짓말/상미.py)

```py

```

### [성구](./거짓말/성구.py)

```py

```

### [영준](./거짓말/영준.py)

```py

```

<br/>

## [파이프 옮기기1](https://www.acmicpc.net/problem/17070)

### [민웅](./파이프%20옮기기1/민웅.py)

```py
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
```

### [상미](./파이프%20옮기기1/상미.py)

```py

```

### [성구](./파이프%20옮기기1/성구.py)

```py
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
```

### [영준](./파이프%20옮기기1/영준.py)

```py

```

<br/>

</details>

<br/><br/>

# 지난주 스터디 문제

<details markdown="1">
<summary>접기/펼치기</summary>

<br/>

## [프로세서 연결하기](https://swexpertacademy.com/main/code/problem/problemDetail.do?contestProbId=AV4suNtaXFEDFAUf)

### [민웅](./프로세서%20연결하기/민웅.py)

```py
# SWEA_프로세서연결하기
import sys
input = sys.stdin.readline
dxy = [(-1, 0), (0, 1), (1, 0), (0, -1)]

T = int(input())


def bt(idx, core_cnt, total_dis, picked_lines):
    global c_cnt, ans
    if idx == len(points):
        if core_cnt > c_cnt:
            ans = total_dis
            c_cnt = core_cnt
        elif core_cnt == c_cnt and total_dis < ans:
            ans = total_dis
        return

    sp = points[idx]

    for ep in lines[idx]:
        now_line = (sp, ep)
        for bl in picked_lines:
            if is_cross(now_line, bl):
                break
        else:
            picked_lines.append(now_line)
            bt(idx+1, core_cnt+1, total_dis + distance(sp, ep), picked_lines)
            picked_lines.pop()

    bt(idx+1, core_cnt, total_dis, picked_lines)


def is_cross(l1, l2):
    s1, e1 = l1
    s2, e2 = l2

    # 선분1이 수평
    if s1[0] == e1[0]:
        # 선분2도 수평
        if s2[0] == e2[0]:
            # 다른라인이면 안겹침
            if s1[0] != s2[0]:
                return False
            if s1[1] > e2[1] or e1[1] < s2[1]:
                return False
        # 선분2는 수직
        else:
            if min(s1[1], e1[1]) <= s2[1] <= max(s1[1], e1[1]) and min(s2[0], e2[0]) <= s1[0] <= max(s2[0], e2[0]):
                return True
            else:
                return False
    else:
        # 선분1 수직, 선분2 수평
        if s2[0] == e2[0]:
            if min(s1[0], e1[0]) <= s2[0] <= max(s1[0], e1[0]) and min(s2[1], e2[1]) <= s1[1] <= max(s2[1], e2[1]):
                return True
            else:
                return False
        # 선분2도 수직
        else:
            if s1[1] != s2[1]:
                return False
            if s1[0] > e2[0] or e1[0] < s2[0]:
                return False

    return True


def distance(s, e):
    return abs(e[0] - s[0]) + abs(e[1] - s[1])


for tc in range(1, T+1):
    N = int(input())
    cores = [list(map(int, input().split())) for _ in range(N)]

    points = []
    side_points = []
    c_cnt = 0
    ans = 0

    for i in range(N):
        for j in range(N):
            if cores[i][j]:
                if i == 0 or i == N-1 or j == 0 or j == N-1:
                    side_points.append((i, j))
                else:
                    points.append((i, j))

    lines = [[] for _ in range(len(points))]

    for i in range(len(points)):
        p = points[i]
        start = (p[0], p[1])
        for d in dxy:
            nx = p[0] + d[0]
            ny = p[1] + d[1]
            end = 0
            while 0 <= nx <= N-1 and 0 <= ny <= N-1:
                if cores[nx][ny] == 1:
                    end = 0
                    break
                end = (nx, ny)
                nx += d[0]
                ny += d[1]

            if end:
                lines[i].append(end)

    bt(0, 0, 0, [])

    # print(lines)
    # print(c_cnt, ans)
    print(f'#{tc} {ans}')
```

### [상미](./프로세서%20연결하기/상미.py)

```py

```

### [성구](./프로세서%20연결하기/성구.py)

```py

```

### [영준](./프로세서%20연결하기/영준.py)

```py

```

</details>

<br/><br/>

# 알고리즘 설명

<details markdown="1">
<summary>접기/펼치기</summary>

</details>
