# 학습 로그

매일 종료 전에 짧게 기록합니다. 동료 코드나 AI의 구현 제안을 보기 전에는 내가 시도한 내용을 먼저 남기세요.

## Day 1 - 2026-11-27

### 오늘 구현한 것
- `Affine.forward`: `x` 저장 후 `x @ W + b`
- `ReLU.forward`: 0 이하는 0으로
- `SoftmaxWithLoss.forward`: softmax 후 cross entropy 평균

### 오늘 확인한 shape와 수식
- 입력/출력 shape: 
    1. `(B,784) → (B,H)`: Affine 1, `@ W1(784,H) + b1(H,)`로 784개 값을 H개로 변환
    2. `(B,H) → (B,H)`: ReLU, 0 이하를 0으로 바꾸며 크기는 유지
    3. `(B,H) → (B,10)`: Affine 2, `@ W2(H,10) + b2(10,)`로 숫자 0~9의 점수(logits) 출력
    4. `(B,10) → scalar`: SoftmaxWithLoss, 확률 `(B,10)`로 바꾸고 이미지별 loss의 평균 계산

- 손으로 확인한 계산: `x @ W + b`의 첫 칸(1행 1열)
    - 입력: `x=[[1,2],[3,4]]`, `W=[[1,0,-1],[2,1,0]]`, `b=[0.5,-1,0]`
    - 사용한 값: `x`의 첫 행 `[1,2]`, `W`의 첫 열 `[1,2]`, `b`의 첫 값 `0.5`
    - 계산: `1×1 + 2×2 + 0.5 = 5.5` (코드와 일치)

- 반드시 성립해야 하는 조건: 확률의 행 합은 1, logits가 모두 같으면 loss는 `ln(10)`

### 막힌 지점과 해결 과정
- 증상: `pytest`, `numpy`가 없다는 오류
- 내가 세운 가설:
- 시도 1: Python 3.14로 NumPy 계산 확인
- 시도 2: `python`에 패키지 설치를 확인하고 다시 실행
- 해결 또는 다음 확인 항목: forward 테스트 13개 통과

### 참고 기록
| 출처(교재·동료·AI) | 무엇을 참고했는가 | 내 코드에서 검증한 방법 |
| --- | --- | --- |
| AI (Claude Code) | forward 3개 코드 초안과 설명 참고 | 작은 배열 계산과 테스트로 확인 |

### 코드 없이 답할 질문 3개
1. `Affine.forward`는 왜 `x`를 저장할까?
    - backward에서 W의 기울기 `dW = x.T @ dout`을 계산할 때 필요
2. logits와 확률은 어떻게 다를까?
    - softmax를 거치기 전의 점수 / 거친 후의 값
3. logits가 모두 같으면 왜 loss가 `ln(10)`일까?
    - 10개 클래스의 확률이 모두 1/10이 되기 때문
    - loss = `-ln(정답 클래스의 확률)` = `-ln(1/10)` = `ln(10)`

## Day 2 - 2026-11-28

### 오늘 구현한 것
- `SoftmaxWithLoss.backward`: `(확률 - one_hot) / B`
- `ReLU.backward`: forward의 `mask` 위치는 gradient 0
- `Affine.backward`: `dW`, `db`, `dx` 계산

### 오늘 확인한 shape와 수식
- 입력/출력 shape (backward):
    1. `dlogits (B,10)`
    2. Affine 2: `dW2 (H,10)`, `db2 (10,)`, `dx (B,H)`
    3. ReLU: `(B,H) → (B,H)`
    4. Affine 1: `dW1 (784,H)`, `db1 (H,)`, `dx (B,784)`
- 예시 계산: 확률 `[0.5,0.5]`, 정답 0, B=1이면 `dlogits = [-0.5, 0.5]`
- 반드시 성립해야 하는 조건: `dW`, `db`, `dx`의 shape는 각각 `W`, `b`, `x`와 같고, `dlogits`의 합은 0에 가까움

### 막힌 지점과 해결 과정
- 증상: gradient check의 최대 상대 오차가 `0.00e+00`으로 나옴
- 내가 세운 가설:
- 시도 1: 제공 코드 확인 (차이가 `1e-8` 이하면 0으로 처리)
- 시도 2: 0 처리를 끄고 실행(`5.9e-09`), `db`를 일부러 틀리게 해 실패하는지 확인
- 해결 또는 다음 확인 항목: 검사가 정상 동작함, 다음은 Day 3

### 참고 기록
| 출처(교재·동료·AI) | 무엇을 참고했는가 | 내 코드에서 검증한 방법 |
| --- | --- | --- |
| AI (Claude Code) | backward 3개 코드 초안과 설명 참고 | selfcheck 26개와 gradient check 2개로 확인 |

### 코드 없이 답할 질문 3개
1. `dW`, `db`, `dx`의 shape는 각각 무엇과 같아야 할까?
2. ReLU backward에서 gradient가 0이 되는 위치는 어디일까?
3. gradient check는 어떤 두 gradient를 비교할까?
