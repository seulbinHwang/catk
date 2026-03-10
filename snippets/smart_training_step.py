def training_step(self, data: dict, batch_idx: int) -> Tensor:
    """학습 step을 수행한다.

    이 버전은 Flow-Planner 스타일에 맞춰 두 손실만 기록한다.

    - ``train/flow``
    - ``train/consistency``

    Returns:
        스칼라 total loss.
    """
    tokenized_map, tokenized_agent = self.token_processor(data)
    map_feature = self.encoder.encode_map(tokenized_map)

    total_loss = 0.0
    total_flow = 0.0
    total_consistency = 0.0
    n_terms = 0

    if self.use_closed_loop_finetune:
        outputs = self.encoder.closed_loop_train(
            map_feature=map_feature,
            tokenized_agent=tokenized_agent,
            agent_raw=data["agent"],
            unroll_steps=self.closed_loop_unroll,
        )
        for pred in outputs:
            loss_mask = self._loss_mask_train(pred, data)
            loss, log_dict = self._compute_single_loss(pred, loss_mask)

            total_loss = total_loss + loss
            total_flow = total_flow + log_dict["flow"]
            total_consistency = total_consistency + log_dict["consistency"]
            n_terms += 1
    else:
        for anchor_step in self._select_train_anchor_steps():
            pred = self.encoder.forward_from_map(
                map_feature=map_feature,
                tokenized_agent=tokenized_agent,
                agent_raw=data["agent"],
                anchor_step=anchor_step,
            )
            loss_mask = self._loss_mask_train(pred, data)
            loss, log_dict = self._compute_single_loss(pred, loss_mask)

            total_loss = total_loss + loss
            total_flow = total_flow + log_dict["flow"]
            total_consistency = total_consistency + log_dict["consistency"]
            n_terms += 1

    total_loss = total_loss / max(n_terms, 1)

    self.log("train/loss", total_loss, on_step=True, batch_size=1)
    self.log("train/flow", total_flow / max(n_terms, 1), on_step=True, batch_size=1)
    self.log("train/consistency", total_consistency / max(n_terms, 1), on_step=True, batch_size=1)
    return total_loss
