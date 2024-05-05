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