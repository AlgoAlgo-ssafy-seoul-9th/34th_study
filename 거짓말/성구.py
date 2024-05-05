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

'''
set 자료형을 이용한 풀이 - 실패
# 1043 거짓말
import sys
input = sys.stdin.readline


def main():
    N, M = map(int, input().split())
    n, *is_know = map(int, input().split())
    is_know = set(is_know)

    parties = []
    for i in range(M):
        _, *people = map(int, input().split())
        parties.append(set(people))

    if n == 0:
        print(M)
    else:
        cnt = M
        for party in parties:
            if is_know.intersection(party):
                is_know = is_know.union(party)
        for party in parties[::-1]:
            if is_know.intersection(party):
                is_know = is_know.union(party)
        for party in parties:
            if is_know.intersection(party):
                is_know = is_know.union(party)
        for party in parties:
            if is_know.intersection(party):
                cnt -=1

        print(cnt)
    
    return


if __name__ == "__main__":
    main()

'''
