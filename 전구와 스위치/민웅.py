'''
8
00000000
11011000

7
1101000
1111111

4
1100
1000
'''

# 2138_전구와스위치_Bulb and Switch
import sys
sys.setrecursionlimit(10**6)
input = sys.stdin.readline

def bt(n, status, cnt):
    global ans
    if n == N-1:
        if status == wannabe and cnt < ans:
            ans = cnt
        status[n] = 1-status[n]
        status[n-1] = 1-status[n-1]
        if status == wannabe and cnt+1 < ans:
            ans = cnt+1
        return

    if cnt >= ans:
        return

    if status[n] == wannabe[n]:
        if status[n-1] != wannabe[n-1]:
            status[n-1] = 1 - status[n-1]
            status[n] = 1 - status[n]
            status[n+1] = 1 - status[n+1]
            bt(n+1, status, cnt+1)
        else:
            bt(n+1, status, cnt)
    else:
        if status[n-1] != wannabe[n-1]:
            status[n-1] = 1 - status[n-1]
            status[n] = 1 - status[n]
            status[n+1] = 1 - status[n+1]
            bt(n+1, status, cnt+1)
        else:
            bt(n+1, status, cnt)

N = int(input())

bulbs = list(map(int, input().strip()))
wannabe = list(map(int, input().strip()))

ans = 100001
tmp = [bulbs[i] for i in range(N)]

bt(1, bulbs, 0)
tmp[0] = 1-tmp[0]
tmp[1] = 1-tmp[1]
bt(1, tmp, 1)

if ans == 100001:
    print(-1)
else:
    print(ans)
