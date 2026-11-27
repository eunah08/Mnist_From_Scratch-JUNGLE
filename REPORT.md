# REPORT - NumPy 신경망으로 손글씨 숫자 인식

코드를 길게 복사하지 말고, 구현할 때 판단한 내용과 검증 근거를 작성하세요.

## 0. 팀 정보

| 반 | 팀명 | 팀원 | 구현·리뷰 방식 |
| 301 | 13 | 윤종휘 |  |
| 301 | 13 | 조은아 | ai를 사용해 구현, 코드를 하나하나 적어가며 리뷰 |

## 1. 전체 학습 흐름

실제 실행한 batch의 shape를 채우세요.

Day 1: forward만 확인 (B=4, H=128, 랜덤 입력). backward와 update는 아직 실행 전입니다.

```text
x_batch                       (B, 784)
-> Affine 1                   (B, H)
-> ReLU                       (B, H)
-> Affine 2 / logits          (B, 10)
-> SoftmaxWithLoss            scalar

loss
-> dlogits                    (B, 10)
-> Affine 2 backward          dW2 / db2 / dx
-> ReLU backward              (B, H)
-> Affine 1 backward          dW1 / db1 / dx
-> optimizer update
```

`Forward -> Loss -> Backward -> Update`가 이 순서여야 하는 이유를 설명하세요.

## 2. 핵심 구현

### 2.1 Affine

- `x @ W + b`에서 각 배열의 shape: `x (B, in)`, `W (in, out)`, `b (out,)` → 출력 `(B, out)`
- `dW = x.T @ dout`이 되는 이유:
- `db`를 batch 방향으로 합산하는 이유:
- forward 입력을 저장해야 하는 이유: backward에서 `dW`를 계산할 때 필요합니다.

### 2.2 ReLU

- forward에서 저장한 mask: `x <= 0`인 위치가 `True`입니다.
- backward에서 gradient가 0이 되는 위치와 이유:

### 2.3 SoftmaxWithLoss

- logits와 확률의 차이: logits는 점수, 확률은 softmax를 거친 0~1 값(행 합 1)입니다.
- max-shift를 적용한 이유: `exp` overflow를 막습니다. 확률은 변하지 않습니다.
- 모든 logits가 같을 때 loss가 `ln(10)`인 이유: 확률이 모두 `1/10`이라 `-ln(1/10) = ln(10)`입니다.
- gradient가 `(확률 - 정답 one-hot) / batch_size`가 되는 의미:

### 2.4 SGD와 학습 루프

- gradient 반대 방향으로 이동하는 이유:
- 파라미터를 제자리에서 갱신해야 하는 이유:
- 한 미니배치에서 실행한 네 단계:

## 3. 검증 결과

| 검증 | 실행 조건 | 결과 | 무엇을 확인했는가 |
| --- | --- | --- | --- |
| 작은 배열 단위 테스트 | 각 함수 | forward 13개 통과 | 값과 shape |
| 초기 loss | `ln(10)`과 비교 | 2.3026 (MNIST 아님) | 출력층 forward |
| gradient check | 최대 상대 오차 |  | backward 수학적 일치 |
| 학습 Smoke | 인공 데이터 |  ->  | 네 단계 연결과 parameter 변화 |
| 전체 테스트 | `pytest tests/ -v` |  | 전체 불변조건 |

인공 데이터 정확도 검사는 학습 연결을 확인하는 테스트일 뿐 MNIST 성능 목표가 아닙니다.

## 4. MNIST 실행 관찰

| 항목 | 값 |
| --- | --- |
| hidden size |  |
| optimizer / learning rate |  |
| epochs / batch size |  |
| 파라미터 수 |  |
| 첫 epoch loss |  |
| 마지막 epoch loss |  |
| 관찰한 정확도 |  |
| 실행 시간 |  |

정확도는 실행 상태를 관찰하기 위한 값이며 제출 통과 기준이 아닙니다. 학습 전후 예측이 어떻게 달라졌는지 한두 문장으로 적으세요.

## 5. 오류와 해결 과정

| 증상 | 처음 세운 가설 | 확인한 최소 입력 | 실제 원인 | 수정 후 검증 |
| --- | --- | --- | --- | --- |
| `No module named pytest` / `numpy` |  | import 확인 | 패키지 미설치 | 설치 확인 후 13개 통과 |
|  |  |  |  |  |

## 6. 참고와 AI 사용 기록

| 출처 | 질문·참고 내용 | 내가 먼저 시도한 것 | 직접 검증한 방법 |
| --- | --- | --- | --- |
| AI (Claude Code) | forward 3개 코드 초안 |  | ai가 작성한 초안을 바탕으로 직접 검사용 계산과 테스트 실행 |

## 7. 회고

- 이제 코드와 shape으로 설명할 수 있는 개념:
- 아직 설명하기 어려운 개념:
- 다시 구현한다면 먼저 확인할 것:
