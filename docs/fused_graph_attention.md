# FlashAttention Graph Attention for CAT-K `AttentionLayer`

## 목표

이 변경의 목표는 다음 네 가지를 동시에 만족하는 것입니다.

```text
같은 train_batch_size 유지
GPU 메모리 사용량 감소
학습 속도 유지 또는 향상
네트워크 구조/config/attention 수식 유지
```

## 변경 전 문제

기존 PyG `MessagePassing` attention은 edge 단위로 여러 중간 tensor를 만듭니다.

```text
k_j
v_j
relation_key
relation_value
attention_score
attention_weight
weighted_value
```

edge 수가 커질수록 이 중간값이 CUDA 메모리를 크게 차지합니다.

## `5e4297e` chunk 방식의 한계

이전 memory-saving 구현은 edge를 chunk로 나눠 처리했습니다.

```text
큰 중간 tensor 저장 감소
대신 edge 여러 번 재방문
relation projection 반복 계산
PyTorch kernel 호출 반복 증가
```

그래서 peak memory는 줄었지만 step time이 느려질 수 있었습니다.

## 이번 구현 방향

이번 구현은 PyTorch chunk loop를 production 경로에서 제거하고, CUDA에서는 FlashAttention varlen kernel을 사용합니다.

핵심 구조:

```text
1. edge를 target node 기준으로 정렬
2. target별 edge 시작 위치를 metadata로 저장
3. 여러 AttentionLayer가 같은 metadata 재사용
4. 각 target node를 query 길이 1인 variable-length cross-attention 문제로 변환
5. FlashAttention kernel 안에서 softmax와 value 합산 처리
6. backward는 edge tensor를 chunk 단위로 다시 만들어 현재 chunk의 gradient만 계산
```

attention 수식은 그대로 유지합니다.

```text
score = q · (k + relation_key)
attention = softmax(score)
output = attention · (v + relation_value)
```

## 적용 위치

- `src/smart/layers/graph_flash_attention.py`
- `src/smart/layers/graph_flash_attention_flash.py`
- `src/smart/layers/attention_layer.py`
- `src/smart/modules/map_decoder.py`
- `src/smart/modules/flow_agent_decoder.py`는 `tools/apply_fused_graph_attention_patch.py`로 metadata 재사용 패치를 적용합니다.

## 필수 조건

CUDA 학습에서는 FlashAttention wheel이 필요합니다.

```bash
python -m pip install -r install/requirements.txt
```

FlashAttention을 불러올 수 없으면 기존 PyG attention으로 조용히 fallback하지 않고 즉시 에러를 냅니다.

## 검증

CPU reference 수식 검증:

```bash
python -m pytest tests/test_graph_flash_attention.py -q
```

CUDA/FlashAttention 환경에서는 같은 테스트 안의 smoke test가 자동으로 활성화됩니다.

실제 pretrain에서는 아래를 확인해야 합니다.

```text
peak CUDA memory
samples/sec 또는 step time
초기 loss curve
validation metric drift
```

## 제한

이 구현은 forward activation으로 edge별 relation key/value, attention score, attention weight, weighted value를 저장하지 않습니다. backward에서는 현재 chunk의 relation key/value만 다시 만들고, 해당 chunk가 끝나면 해제합니다.

실제 속도는 GPU, dtype, scene density, degree 분포에 따라 달라질 수 있습니다. 따라서 최종 판단은 H100/A100/V100 각각에서 동일 batch size 기준으로 peak memory와 step time을 측정해야 합니다.
