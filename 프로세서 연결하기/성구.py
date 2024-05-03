# SW Expert Academy 프로세서 연결하기
from collections import defaultdict
# import sys
# sys.stdin = open('', 'r')


def main():
    
    def connectProccessor(N:int, meccinos:list) -> int:
        proccessors = []
        lines = defaultdict(list)
        connected = 0
        for i in range(1, N+1):
            for j in range(1, N+1):
                if meccinos[i][j]:  # 1일때, 프로세서가 있을때
                    if i in (0, N) or j in (0, N):
                        connected += 1
                        continue
                    proccessors.append((i,j))   # 프로세서 추가
                    # 벽으로 향하는 모든 방향 추가
                    lines[(i,j)].append((0,j))
                    lines[(i,j)].append((N-1, j))
                    lines[(i,j)].append((i,0))
                    lines[(i,j)].append((i, N-1))


        def isCrossing(s1, e1, s2, e2):
            if min(s1[1], e1[1]) <= s2[1] <= max(s1[1],e1[1]) and min(s2[0], e2[0]) <= s1[0] <= max(s2[0], e2[0]):
                return 1
            if min(s2[1], e2[1]) <= s1[1] <= max(s2[1],e2[1]) and min(s1[0], e1[0]) <= s2[0] <= max(s1[0], e1[0]):
                return 1
            return 0

        def bt(idx, cnt:list, line_cnt:list, line:list):
            
            if idx == len(proccessors):
                if line_cnt[0] < cnt[0]:
                    line_cnt = [cnt[0], cnt[1]]
                elif line_cnt[0] == cnt[0]:
                    line_cnt[1] = max(line_cnt[1], cnt[1])
                return line_cnt
            
            for i in range(4):
                di,dj = lines[proccessors[idx]][i]
                distance = abs(proccessors[idx][0]-di)+abs(proccessors[idx][1]-dj)
                tmp = line.copy
                tmp.append((idx, i))
                for p_idx, p_line in line:
                    prev_i, prev_j = lines[proccessors[p_idx]][p_line]
                    if isCrossing(proccessors[idx], (di,dj), proccessors[p_idx], (prev_i, prev_j)):
                        cand1 = bt(idx+1, [cnt[0], cnt[1]+distance], line_cnt, tmp)
                        break
                else:
                    cand2 = bt(idx+1, [cnt[0]+1, cnt[1]+distance], line_cnt, tmp)
                    

            return 
                    
        
        return
    
    for t in range(1, int(input())+1):
        N = int(input())
        meccinos = [ list(map(int, list(input().split()))) for _ in range(N)]
        print(f'#{t} {connectProccessor(N, meccinos)}')

    return 


if __name__ == "__main__":
    main()