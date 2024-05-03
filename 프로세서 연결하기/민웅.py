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