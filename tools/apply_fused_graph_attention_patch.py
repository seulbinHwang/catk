from __future__ import annotations

from pathlib import Path


FLOW_IMPORT = "from src.smart.layers.graph_flash_attention import build_graph_attention_metadata\n"

README_SECTION = """
### Fused Graph Attention for `AttentionLayer`

Pretrain에서 CUDA peak memory를 줄이기 위해 `AttentionLayer`는 PyG `MessagePassing` 기반 edge-wise attention 대신 CAT-K 전용 fused graph attention을 사용합니다.

이 변경은 네트워크 구조와 config를 바꾸지 않습니다.

- `hidden_dim`, `num_heads`, `head_dim`, layer 수는 그대로입니다.
- map-map / map-agent / temporal / agent-agent edge 생성 방식은 그대로입니다.
- radius, `max_num_neighbors`, relation embedding, loss, optimizer, batch size는 그대로입니다.
- attention 수식은 `softmax(q · (k + relation_key)) · (v + relation_value)` 그대로입니다.

변경되는 것은 `AttentionLayer` 내부 계산 방식뿐입니다. CUDA에서는 Triton kernel이 target node 기준으로 정렬된 edge를 읽고, score 계산, softmax, value 합산, backward 계산을 GPU kernel 안에서 처리합니다. 따라서 edge별 score, attention weight, weighted value를 큰 tensor로 오래 저장하지 않습니다.

필수 조건:

```bash
python -m pip install -r install/requirements.txt
```

`install/requirements.txt`에는 PyTorch 2.4.1과 맞는 `triton==3.0.0`이 포함되어 있습니다. CUDA tensor에서 Triton을 불러올 수 없으면 기존 PyG 경로로 조용히 fallback하지 않고 즉시 에러를 냅니다. CPU에서는 unit test용 reference path만 사용합니다.

검증:

```bash
python -m pytest tests/test_graph_flash_attention.py -q
```

실제 pretrain에서 확인할 항목:

- 같은 `data.train_batch_size` 기준 peak CUDA memory
- `train/step_time` 또는 samples/sec
- 초기 수천 step의 loss curve
- validation metric drift

""".strip()


def _replace_once(text: str, old: str, new: str, file_path: Path) -> str:
    """문자열을 정확히 한 번만 교체합니다.

    Args:
        text: 원본 파일 내용입니다.
        old: 찾을 문자열입니다.
        new: 바꿀 문자열입니다.
        file_path: 에러 메시지에 표시할 파일 경로입니다.

    Returns:
        교체된 파일 내용입니다.

    Raises:
        RuntimeError: 찾을 문자열이 없거나 여러 번 등장할 때 발생합니다.
    """
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one match in {file_path}, found {count}: {old[:120]!r}")
    return text.replace(old, new, 1)


def _patch_flow_agent_decoder(repo_root: Path) -> None:
    """`flow_agent_decoder.py`에서 edge metadata를 여러 layer가 재사용하게 만듭니다.

    Args:
        repo_root: CAT-K repository root입니다.
    """
    path = repo_root / "src/smart/modules/flow_agent_decoder.py"
    text = path.read_text()
    if FLOW_IMPORT not in text:
        text = _replace_once(
            text,
            "from src.smart.layers.fourier_embedding import FourierEmbedding\n",
            "from src.smart.layers.fourier_embedding import FourierEmbedding\n" + FLOW_IMPORT,
            path,
        )

    marker = """        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(\n            pos_pl=map_feature[\"position\"],\n            orient_pl=map_feature[\"orientation\"],\n            pos_a=pos_a, # ctx_sampled_pos\n            head_a=head_a,  # ctx_sampled_heading\n            head_vector_a=head_vector_a, # ctx_sampled_heading\n            mask=mask,\n            batch_s=batch_s_pl2a,\n            batch_pl=map_feature[\"batch\"],\n            light_type=map_feature.get(\"light_type\"),\n        )\n\n        feat_map = map_feature[\"pt_token\"]\n        for i in range(self.num_layers):\n            feat_a = feat_a.flatten(0, 1)\n            feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)\n            feat_a = feat_a.view(n_agent, n_step, -1).transpose(0, 1).flatten(0, 1)\n            feat_a = self.pt2a_attn_layers[i]((feat_map, feat_a), r_pl2a, edge_index_pl2a)\n            feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)\n            feat_a = feat_a.view(n_step, n_agent, -1).transpose(0, 1)\n        return feat_a\n"""
    replacement = """        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(\n            pos_pl=map_feature[\"position\"],\n            orient_pl=map_feature[\"orientation\"],\n            pos_a=pos_a, # ctx_sampled_pos\n            head_a=head_a,  # ctx_sampled_heading\n            head_vector_a=head_vector_a, # ctx_sampled_heading\n            mask=mask,\n            batch_s=batch_s_pl2a,\n            batch_pl=map_feature[\"batch\"],\n            light_type=map_feature.get(\"light_type\"),\n        )\n\n        t_metadata = build_graph_attention_metadata(\n            edge_index=edge_index_t,\n            num_dst_nodes=n_agent * n_step,\n        )\n        r_t = t_metadata.reorder_edge_features(r_t)\n        edge_index_t = t_metadata.sorted_edge_index\n        pl2a_metadata = build_graph_attention_metadata(\n            edge_index=edge_index_pl2a,\n            num_dst_nodes=n_agent * n_step,\n        )\n        r_pl2a = pl2a_metadata.reorder_edge_features(r_pl2a)\n        edge_index_pl2a = pl2a_metadata.sorted_edge_index\n        a2a_metadata = build_graph_attention_metadata(\n            edge_index=edge_index_a2a,\n            num_dst_nodes=n_agent * n_step,\n        )\n        r_a2a = a2a_metadata.reorder_edge_features(r_a2a)\n        edge_index_a2a = a2a_metadata.sorted_edge_index\n\n        feat_map = map_feature[\"pt_token\"]\n        for i in range(self.num_layers):\n            feat_a = feat_a.flatten(0, 1)\n            feat_a = self.t_attn_layers[i](\n                feat_a,\n                r_t,\n                edge_index_t,\n                attention_metadata=t_metadata,\n                r_is_sorted=True,\n            )\n            feat_a = feat_a.view(n_agent, n_step, -1).transpose(0, 1).flatten(0, 1)\n            feat_a = self.pt2a_attn_layers[i](\n                (feat_map, feat_a),\n                r_pl2a,\n                edge_index_pl2a,\n                attention_metadata=pl2a_metadata,\n                r_is_sorted=True,\n            )\n            feat_a = self.a2a_attn_layers[i](\n                feat_a,\n                r_a2a,\n                edge_index_a2a,\n                attention_metadata=a2a_metadata,\n                r_is_sorted=True,\n            )\n            feat_a = feat_a.view(n_step, n_agent, -1).transpose(0, 1)\n        return feat_a\n"""
    if marker in text:
        text = _replace_once(text, marker, replacement, path)

    cache_marker = """        edge_index_a2a, r_a2a = self.build_interaction_edge(\n            pos_a=pos_window,\n            head_a=head_window,\n            head_vector_a=head_vector_window,\n            batch_s=batch_s_a2a,\n            mask=valid_window,\n        )\n\n        feat_map = map_feature[\"pt_token\"]\n        feat_a_t_dict: Dict[int, torch.Tensor] = {}\n        feat_a_now = feat_a[:, -1].clone()\n        for i in range(self.num_layers):\n            temporal_feat = feat_a if i == 0 else feat_a_t_dict[i]\n            temporal_feat = self.t_attn_layers[i](\n                temporal_feat.flatten(0, 1),\n                r_t,\n                edge_index_t,\n            ).view(n_agent, n_step, -1)\n            temporal_feat = temporal_feat.transpose(0, 1).flatten(0, 1)\n            temporal_feat = self.pt2a_attn_layers[i]((feat_map, temporal_feat), r_pl2a, edge_index_pl2a)\n            temporal_feat = self.a2a_attn_layers[i](temporal_feat, r_a2a, edge_index_a2a)\n            temporal_feat = temporal_feat.view(n_step, n_agent, -1).transpose(0, 1)\n            feat_a_now = temporal_feat[:, -1]\n            if i + 1 < self.num_layers:\n                feat_a_t_dict[i + 1] = temporal_feat\n"""
    cache_replacement = """        edge_index_a2a, r_a2a = self.build_interaction_edge(\n            pos_a=pos_window,\n            head_a=head_window,\n            head_vector_a=head_vector_window,\n            batch_s=batch_s_a2a,\n            mask=valid_window,\n        )\n\n        t_metadata = build_graph_attention_metadata(\n            edge_index=edge_index_t,\n            num_dst_nodes=n_agent * n_step,\n        )\n        r_t = t_metadata.reorder_edge_features(r_t)\n        edge_index_t = t_metadata.sorted_edge_index\n        pl2a_metadata = build_graph_attention_metadata(\n            edge_index=edge_index_pl2a,\n            num_dst_nodes=n_agent * n_step,\n        )\n        r_pl2a = pl2a_metadata.reorder_edge_features(r_pl2a)\n        edge_index_pl2a = pl2a_metadata.sorted_edge_index\n        a2a_metadata = build_graph_attention_metadata(\n            edge_index=edge_index_a2a,\n            num_dst_nodes=n_agent * n_step,\n        )\n        r_a2a = a2a_metadata.reorder_edge_features(r_a2a)\n        edge_index_a2a = a2a_metadata.sorted_edge_index\n\n        feat_map = map_feature[\"pt_token\"]\n        feat_a_t_dict: Dict[int, torch.Tensor] = {}\n        feat_a_now = feat_a[:, -1].clone()\n        for i in range(self.num_layers):\n            temporal_feat = feat_a if i == 0 else feat_a_t_dict[i]\n            temporal_feat = self.t_attn_layers[i](\n                temporal_feat.flatten(0, 1),\n                r_t,\n                edge_index_t,\n                attention_metadata=t_metadata,\n                r_is_sorted=True,\n            ).view(n_agent, n_step, -1)\n            temporal_feat = temporal_feat.transpose(0, 1).flatten(0, 1)\n            temporal_feat = self.pt2a_attn_layers[i](\n                (feat_map, temporal_feat),\n                r_pl2a,\n                edge_index_pl2a,\n                attention_metadata=pl2a_metadata,\n                r_is_sorted=True,\n            )\n            temporal_feat = self.a2a_attn_layers[i](\n                temporal_feat,\n                r_a2a,\n                edge_index_a2a,\n                attention_metadata=a2a_metadata,\n                r_is_sorted=True,\n            )\n            temporal_feat = temporal_feat.view(n_step, n_agent, -1).transpose(0, 1)\n            feat_a_now = temporal_feat[:, -1]\n            if i + 1 < self.num_layers:\n                feat_a_t_dict[i + 1] = temporal_feat\n"""
    if cache_marker in text:
        text = _replace_once(text, cache_marker, cache_replacement, path)
    path.write_text(text)


def _patch_readme(repo_root: Path) -> None:
    """README에 fused graph attention 설명을 추가합니다.

    Args:
        repo_root: CAT-K repository root입니다.
    """
    path = repo_root / "README.md"
    text = path.read_text()
    if "### Fused Graph Attention for `AttentionLayer`" in text:
        return
    marker = "\n### Fast WOSAC Metric\n"
    if marker not in text:
        raise RuntimeError("Could not find README insertion marker: ### Fast WOSAC Metric")
    text = text.replace(marker, "\n" + README_SECTION + "\n\n" + marker.lstrip(), 1)
    path.write_text(text)


def main() -> None:
    """현재 repository에 fused graph attention 보조 패치를 적용합니다."""
    repo_root = Path.cwd()
    _patch_flow_agent_decoder(repo_root)
    _patch_readme(repo_root)
    print("Applied fused graph attention flow_agent_decoder and README patches.")


if __name__ == "__main__":
    main()
