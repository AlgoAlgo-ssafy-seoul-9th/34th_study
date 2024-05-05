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
```

### [상미](./근손실/상미.py)

```py

```

### [성구](./근손실/성구.py)

```py
# 18429 근손실
import sys
input = sys.stdin.readline


def main():
    N, K = map(int, input().split())
    w_list = list(map(int, input().split()))
    # 미리 K 다 빼 놓기
    for i in range(N):
        w_list[i] -= K

    # 다 돌면서 중간에 음수가 되면 넘김
    def dfs(start):
        stack = [(set([start]), w_list[start])]
        cnt = 0
        while stack:
            visited, w = stack.pop()
            if len(visited) == N:
                if w >= 0:
                    cnt += 1
                continue
            if w < 0:
                continue
            for i in range(N):
                if not i in visited:
                    stack.append((visited.union(set([i])), w+w_list[i]))

        return cnt
    
    # 모든 시작점에서 탐색
    ans = 0
    for i in range(N):
        ans += dfs(i)

    print(ans)
    return 


if __name__ == "__main__":
    main()
```

### [영준](./근손실/영준.py)

```py
def f(i, w):    # i 날짜, w 중량
    global ans
    if i==N:
         ans += 1
    elif w<500:
        return
    else:
        for j in range(i, N):
            kit[i],kit[j] = kit[j], kit[i]
            f(i+1, w+kit[i]-K)              # kit[i] 증가, K 만큼 감소
            kit[i], kit[j] = kit[j], kit[i]

N, K = map(int, input().split()) #  N개의 서로 다른 운동 키트, 하루가 지날 때마다 중량이 K만큼 감소
kit = list(map(int, input().split()))

ans = 0

f(0, 500)
print(ans)
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
```

### [상미](./거짓말/상미.py)

```py

```

### [성구](./거짓말/성구.py)

```py
# 1043 거짓말
import sys
input = sys.stdin.readline


def main():
    N, M = map(int, input().split())
    n, *is_know = map(int, input().split())
    
    parties = []
    for i in range(M):
        _, *members = map(int, input().split())
        parties.append(sorted(members))
    
    # 아는사람이 없으면 모든 파티에서 거짓말할 수 있음
    if n == 0:
        print(M)
    else:
        # union-find를 위한 부모 배열
        parent = [0] * (N+1)
        for i in range(1,N+1):
            parent[i] = i  
        
        # union
        def union(num1, num2):
            f1 = find(num1)
            f2 = find(num2)
            if f1 != f2:
                parent[f2] = f1
            return

        # find
        def find(num):
            tmp = num
            while tmp != parent[tmp]:
                tmp = parent[tmp]
            return  tmp

        # 첫번째 사람에게 모두 union
        for party in parties:
            for person in party[1:]:
                union(party[0],person)
        
        # set 을 통한 시간복잡도 절약
        known =set(is_know)
        for i in is_know:
            known.add(find(i))
        # 총 파티 수에서 빼는 방식
        cnt = M
        for party in parties:
            for i in party:
                # 한 사람이라도 진실을 알고 있으면 파티 수에서 빼고 중단
                if find(i) in known:
                    cnt -=1
                    break
        print(cnt)

         
    
    return


if __name__ == "__main__":
    main()

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
