import sys
input = sys.stdin.readline

N, M = map(int, input().split())
truth_ppl, *truth_nums = map(int, input().split())  # 진실을 아는 사람의 수와 번호
parties = []            # i번 파티에 오는 사람들 번호 parties[i]
for i in range(M):
    party_ppl, *party_nums = map(int, input().split())
    parties.append(party_nums)
ans = 0
lies = []   # 거짓말할 수 있는 파티들
becareful = set()

if truth_ppl == 0:  # 아무도 진실을 모르면 모든 파티 거짓말 가능
    ans = M
else:
    cnt = 0
    while cnt <= 50:
        for p in parties:
            if list(set(p).intersection(truth_nums)) or list(set(p).intersection(becareful)):
                for per in p:
                    becareful.add(per)
        cnt += 1

    for p in parties:
        if not list(set(p).intersection(list(becareful))):
            ans += 1
        
    
print(ans)