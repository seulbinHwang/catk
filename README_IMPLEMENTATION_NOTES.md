# CAT-K Graph Attention Memory Optimization

## 포함 파일

- `src/smart/layers/graph_flash_attention.py`
- `src/smart/layers/attention_layer.py`
- `tests/test_graph_flash_attention.py`
- `docs/graph_flash_attention.md`
- `tools/apply_graph_flash_attention_readme_update.py`

## 적용 방법

레포지토리 루트에서 이 zip의 파일을 같은 경로로 복사합니다.
그 다음 README 섹션을 자동 삽입합니다.

```bash
python tools/apply_graph_flash_attention_readme_update.py
```

검증:

```bash
python -m pytest tests/test_graph_flash_attention.py -q
```

## 검증 상태

로컬 CPU 환경에서 `tests/test_graph_flash_attention.py`는 통과했습니다.
전체 CAT-K 학습 검증은 이 환경에 `torch_geometric`, `torch_cluster`, CUDA GPU가 없어 실행하지 못했습니다.

## 중요한 제한

이 zip의 구현은 PyTorch 연산 기반의 exact memory-saving graph attention입니다.
별도 Triton/CUDA custom kernel은 포함하지 않았습니다.
따라서 edge별 큰 중간 tensor 저장량은 줄이지만, 실제 학습 속도는 데이터 크기와 GPU에 따라 달라질 수 있습니다.
