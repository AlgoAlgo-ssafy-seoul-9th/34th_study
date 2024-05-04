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
