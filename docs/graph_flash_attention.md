# Graph Flash Attention 적용 내용

이 변경은 `AttentionLayer`의 attention 수식, layer 수, head 수, hidden dimension, edge 생성 방식, radius, max neighbor, loss, optimizer, batch config를 바꾸지 않습니다.

바뀌는 부분은 하나입니다.

```text
기존: PyG MessagePassing이 edge별 [E, H, D] 중간 tensor를 크게 생성
변경: target별 neighbor edge를 FlashAttention varlen cross-attention으로 계산
```

## 목적

pretrain 중 `map-map`, `map-agent`, `agent-agent`, `temporal` graph attention에서 생기는 edge별 중간값 저장량을 줄입니다.

## 구현 위치

- `src/smart/layers/graph_flash_attention.py`
- `src/smart/layers/graph_flash_attention_flash.py`
- `src/smart/layers/attention_layer.py`

`AttentionLayer`는 기존처럼 같은 입력을 받고 같은 output shape을 반환합니다. 내부에서만 `graph_flash_attention(...)`을 호출합니다.

## 수식 유지

기존 attention 수식은 유지됩니다.

```text
score = query · (key + relation_key)
attention = softmax(score)
output = attention × (value + relation_value)
```

## 메모리 절감 원리

기존 방식은 edge 전체에 대해 아래 값을 한 번에 만들 수 있습니다.

```text
q_i, k_j, v_j, relation_key, relation_value, score, attention, weighted_value
```

변경 방식은 target node 하나를 query 길이 1인 sequence로 보고, 그 target의 neighbor edge를 variable-length key/value sequence로 넘깁니다. FlashAttention은 score와 attention weight를 큰 activation으로 저장하지 않고 output만 만듭니다.

backward에서는 forward 때 만든 edge별 key/value/relation tensor를 저장해 두지 않고, target chunk 단위로 다시 계산합니다. 그래서 한 layer 전체의 edge activation이 backward 전까지 쌓이는 문제를 피합니다.

## 주의사항

CUDA 학습에는 `flash-attn`이 필요합니다. CUDA에서 FlashAttention을 불러올 수 없으면 기존 PyG 경로로 조용히 fallback하지 않고 즉시 에러를 냅니다. CPU에서는 unit test용 reference path만 사용합니다.
