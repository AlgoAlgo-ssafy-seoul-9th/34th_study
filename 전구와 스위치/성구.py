# 2138 전구와 스위치
import sys
from collections import deque
input = sys.stdin.readline


def main():
    N = int(input())
    state = list(input().strip())
    target = input().strip()
    cnt = 0
    

    def switch(val):
        return "1" if val == "0" else "0"
    
    state0 = [0] * N
    state1 = [0] * N

    for i in range(N):
        state0[i] = state[i]
        state1[i] = state[i]
    

    def reverseState(idx, st):
        if idx == N-1:
            st[idx-1] = switch(st[idx-1])
            st[idx] = switch(st[idx])
        elif idx == 0:
            st[idx] = switch(st[idx])
            st[idx+1] = switch(st[idx+1])            
        else:
            st[idx-1] = switch(st[idx-1])
            st[idx] = switch(st[idx])
            st[idx+1] = switch(st[idx+1])


    reverseState(0, state0)

    cnt = [1,0]
    for i in range(1, N):
        if state0[i-1] != target[i-1]:
            reverseState(i, state0)
            cnt[0] += 1
        if state1[i-1] != target[i-1]:
            reverseState(i, state1)
            cnt[1] += 1
    else:
        ans = 100001
        if ''.join(state0) == target:
            ans = min(ans, cnt[0])
        if ''.join(state1) == target:
            ans = min(ans, cnt[1])
        
    print(ans if ans != 100001 else -1)
    
    return


if __name__ == "__main__":
    main()