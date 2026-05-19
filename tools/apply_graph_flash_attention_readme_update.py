from __future__ import annotations

from pathlib import Path


README_SECTION = """\
### Graph Flash Attention 기반 AttentionLayer 메모리 절감

`AttentionLayer`는 더 이상 PyG `MessagePassing`으로 edge별 attention 중간값을 한 번에 크게 만들지 않습니다.
대신 `src/smart/layers/graph_flash_attention.py`의 exact graph attention 경로를 사용합니다.

유지되는 항목:

- 네트워크 layer 수
- hidden dimension
- attention head 수
- head dimension
- map-agent / agent-agent / map-map edge 생성 방식
- radius와 max neighbor config
- relation embedding
- attention 수식
- loss, optimizer, batch 설정

바뀌는 항목은 `AttentionLayer` 내부 attention 계산 방식뿐입니다.
기존 수식은 그대로 유지합니다.

```text
score = query · (key + relation_key)
attention = softmax(score)
output = attention × (value + relation_value)
```

메모리 절감은 edge 전체에 대한 `[E, H, D]` 크기의 key/value/relation 중간값을 오래 저장하지 않는 방식으로 이뤄집니다.
edge를 target node 기준으로 정렬하고, chunk 단위로 score, softmax 통계, value 합을 계산합니다.
학습 중 attention dropout은 기존처럼 적용되며, backward는 필요한 값을 chunk 단위로 다시 계산합니다.

이 구현은 PyTorch 연산 기반의 exact memory-saving graph attention입니다.
별도 Triton/CUDA custom kernel은 포함하지 않았으므로 CUDA memory는 줄어들 수 있지만, 실제 학습 속도는 scene 크기와 GPU에 따라 달라질 수 있습니다.

"""


def insert_readme_section(readme_path: Path) -> bool:
    """README.md에 graph attention 설명 섹션을 추가합니다.

    Args:
        readme_path: 수정할 README.md 경로입니다.

    Returns:
        실제로 내용을 추가했으면 True, 이미 같은 섹션이 있으면 False입니다.
    """
    text = readme_path.read_text(encoding="utf-8")
    title = "### Graph Flash Attention 기반 AttentionLayer 메모리 절감"
    if title in text:
        return False

    marker = "### Fast WOSAC Metric"
    if marker not in text:
        raise RuntimeError(f"README insertion marker not found: {marker}")

    updated = text.replace(marker, README_SECTION + marker, 1)
    readme_path.write_text(updated, encoding="utf-8")
    return True


def main() -> None:
    """현재 작업 디렉터리의 README.md를 수정합니다."""
    readme_path = Path("README.md")
    if not readme_path.exists():
        raise FileNotFoundError("README.md not found. Run this script at the repository root.")
    changed = insert_readme_section(readme_path)
    if changed:
        print("README.md updated with Graph Flash Attention section.")
    else:
        print("README.md already contains Graph Flash Attention section.")


if __name__ == "__main__":
    main()
