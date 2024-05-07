'''
6 5
1 6
2 4 5
2 1 2
2 2 3
2 3 4
2 5 6
'''
def find_set(p, x):
    while p[x] != x:    # 대표원소가 아니면
        x = p[x]
    return x

def union(p, a, b):
    p[find_set(rep, b)] = p[find_set(rep, a)]

N, M = map(int, input().split())
people = list(map(int, input().split()))
party = [list(map(int, input().split())) for _ in range(M)]
rep = [i for i in range(N+1)]   # 대표원소

if people[0]==0:
    print(M)
else:
    cnt = 0
    for i in range(2, people[0]+1):     # 정직한 사람들 같은 집합으로
        union(rep, people[i-1], people[i])
    for i in range(M):  # 파티에 속한 사람들을 같은 집합으로
        for j in range(2, party[i][0]+1):
            union(rep, party[i][j-1], party[i][j])
    truth = []  #정직한 사람들의 대표원소 목록
    for i in range(1, people[0]+1):
        truth.append(find_set(rep, people[i]))
    for i in range(M):  # 파티에 참여한 사람이
        for j in range(1, party[i][0]+1):   # 진실을 아는 집합에 속하면
            if find_set(rep, party[i][j]) in truth:
                break
        else:   # 진실을 아는 사람이 없는 경우
            cnt += 1

    print(cnt)
