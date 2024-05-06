import sys
input = sys.stdin.readline

def switch(n, arr):
    if n-1 >= 0: 
        if arr[n-1] == '0':
            arr[n-1] = '1'
        else:
            arr[n-1] = '0'
    if arr[n] == '0':
        arr[n] = '1'
    else:
        arr[n] = '0'
    if n+1 <= N-1:
        if arr[n+1] == '0':
            arr[n+1] = '1'
        else:
            arr[n+1] = '0'
    return arr

N = int(input())
before = list(str(input().strip()))
after = list(str(input().strip()))
cnt1, cnt2 = 0, 0
beforeC1 = before.copy()
beforeC2 = before.copy()
cnt = []

switch(0, beforeC1)     # 0번 전구 누름
cnt1 += 1
for i in range(1, N):
    if beforeC1[i-1] != after[i-1]:
        switch(i, beforeC1)
        cnt1 += 1
    else:
        continue

if beforeC1 == after:
    cnt.append(cnt1)

for i in range(1, N):   # 0번 전구 안 누름
    if beforeC2[i-1] != after[i-1]:
        switch(i, beforeC2)
        cnt2 += 1
    else:
        continue

if beforeC2 == after:
    cnt.append(cnt2)

if cnt:
    print(min(cnt))
else:
    print(-1)


'''
000
111  2
001  1
010  3

011  3
101  1
010  2

110  1
001  2
010  3
'''
'''
000
110  1
001  2  111    
010  3  100
100  1
'''