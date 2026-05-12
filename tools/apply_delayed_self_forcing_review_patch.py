from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def patch_flow_agent_decoder() -> None:
    """flow agent context encoder가 지연 시점의 신호 시간 정보를 읽게 수정합니다.

    Args:
        없음.

    Returns:
        None.

    설명:
        지연 self-forcing에서는 2초, 4초, 6초 뒤 시점을 새 현재로 봅니다.
        이때 신호 정보도 그만큼 오래된 정보로 봐야 합니다. 기존 코드는 agent 사전에
        들어간 시간 정보를 map-agent 관계 계산에 넘기지 않았습니다. 이 함수는 해당
        인자를 한 줄 추가합니다.
    """
    path = ROOT / "src/smart/modules/flow_agent_decoder.py"
    text = path.read_text(encoding="utf-8")
    new = (
        '            light_type=map_feature.get("light_type"),\n'
        '            light_time_delta_norm=tokenized_agent.get("light_time_delta_norm"),\n'
        '        )'
    )
    if new in text:
        print("[skip] flow_agent_decoder.py already passes light_time_delta_norm")
        return

    old = (
        '            light_type=map_feature.get("light_type"),\n'
        '        )'
    )
    if old not in text:
        raise RuntimeError(
            "Could not find the build_map2agent_edge call to patch in "
            "src/smart/modules/flow_agent_decoder.py"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("[ok] patched src/smart/modules/flow_agent_decoder.py")


def patch_readme() -> None:
    """README에 review fix 요약을 중복 없이 추가합니다.

    Args:
        없음.

    Returns:
        None.
    """
    path = ROOT / "README.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    marker = "### Delayed-Window Self-Forcing review fix"
    if marker in text:
        print("[skip] README.md already contains review fix note")
        return
    note = """

### Delayed-Window Self-Forcing review fix

`6184acb4` 구현은 지연 시작 schedule과 앞구간 gradient 차단은 맞게 들어가 있습니다.
추가 review patch는 지연 시점을 더 정확한 현재 context로 만들기 위해 두 가지를 보강합니다.

- 지연 시점의 `ctx_sampled_idx`를 자기 생성 0.5초 chunk 기준으로 다시 고릅니다.
- 지연 시점의 traffic-light 시간 차를 D초 기준으로 다시 계산해서 map-agent 관계 입력에 넘깁니다.

이 보강은 새 loss, random window, 혼합 window를 추가하지 않습니다.
"""
    path.write_text(text.rstrip() + note, encoding="utf-8")
    print("[ok] patched README.md")


def main() -> None:
    patch_flow_agent_decoder()
    patch_readme()


if __name__ == "__main__":
    main()
