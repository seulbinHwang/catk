from __future__ import annotations

import copy
import math
from contextlib import nullcontext
from typing import Any, Dict

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from src.smart.model.smart_flow import SMARTFlow
from src.smart.modules.self_forced_gan_cache import (
    TeacherRolloutCache,
    build_current_pose_from_data,
    pack_flat_agent_tensor,
    pack_rollout_prediction_to_set,
)
from src.smart.modules.self_forced_gan_critic import (
    SelfForcedGANDiscriminator,
    add_rollout_pose_perturbation,
    frozen_parameters,
    relativistic_discriminator_loss,
    relativistic_generator_loss,
)
from src.smart.modules.self_forced_trainable_range import apply_self_forced_unfrozen_range


def _cfg(config: object | None, key: str, default: Any) -> Any:
    """설정 객체에서 값을 안전하게 읽습니다.

    Args:
        config: OmegaConf DictConfig, dict, 일반 객체 또는 None입니다.
        key: 읽을 key입니다.
        default: 값이 없을 때 사용할 기본값입니다.

    Returns:
        Any: 읽은 값입니다.
    """
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if callable(getter):
        value = getter(key, default)
    elif isinstance(config, dict):
        value = config.get(key, default)
    else:
        value = getattr(config, key, default)
    return default if value is None else value


class SMARTFlowGAN(SMARTFlow):
    """Set-level GAN 기반 closed-loop fine-tuning 모델입니다.

    Args:
        model_config: 기존 ``SMARTFlow`` 설정에 ``self_forced_gan`` 섹션을 추가한 설정입니다.

    설명:
        기존 pretrained SMARTFlow generator를 유지하고, offline teacher open-loop cache와
        student closed-loop rollout set을 작은 discriminator로 맞춥니다. Generator는
        flow decoder 중심으로만 update하고, scene encoder는 pretrained 표현을 재사용합니다.
    """

    def __init__(self, model_config) -> None:
        super().__init__(model_config)
        self.self_forced_gan_config = getattr(model_config, "self_forced_gan", None)
        self.self_forced_gan_enabled = bool(
            self.self_forced_gan_config is not None
            and _cfg(self.self_forced_gan_config, "enabled", False)
        )
        if not self.self_forced_gan_enabled:
            return

        self.automatic_optimization = False
        self.strict_loading = False
        self.gan_start_epoch = int(_cfg(self.self_forced_gan_config, "start_epoch", 0))
        self.gan_rollout_set_size = int(_cfg(self.self_forced_gan_config, "rollout_set_size", 16))
        self.gan_teacher_cache_size = int(_cfg(self.self_forced_gan_config, "teacher_cache_size", 32))
        self.gan_cache_root = str(_cfg(self.self_forced_gan_config, "teacher_cache_root", ""))
        if not self.gan_cache_root:
            raise ValueError("model.model_config.self_forced_gan.teacher_cache_root must be set.")

        self.gan_effective_scene_batch = int(
            _cfg(self.self_forced_gan_config, "effective_scene_batch", 64)
        )
        self.gan_warmup_scene_exposure = int(
            _cfg(self.self_forced_gan_config, "warmup_scene_exposure", 64000)
        )
        self.gan_warmup_min_updates = int(_cfg(self.self_forced_gan_config, "warmup_min_updates", 500))
        self.gan_warmup_max_updates = int(_cfg(self.self_forced_gan_config, "warmup_max_updates", 1500))
        self.gan_discriminator_lr = float(_cfg(self.self_forced_gan_config, "discriminator_lr", 5.0e-6))
        self.gan_student_lr = float(_cfg(self.self_forced_gan_config, "student_lr", self.lr))
        self.gan_r1_weight = float(_cfg(self.self_forced_gan_config, "r1_weight", 0.1))
        self.gan_r2_weight = float(_cfg(self.self_forced_gan_config, "r2_weight", 0.1))
        self.gan_position_sigma = float(_cfg(self.self_forced_gan_config, "position_sigma", 0.01))
        self.gan_yaw_sigma = float(_cfg(self.self_forced_gan_config, "yaw_sigma", 0.01))
        self.gan_gradient_clip_val = float(_cfg(self.self_forced_gan_config, "gradient_clip_val", 1.0))
        self.gan_manual_accumulate_grad_batches = int(
            _cfg(self.self_forced_gan_config, "manual_accumulate_grad_batches", 1)
        )
        if self.gan_manual_accumulate_grad_batches < 1:
            raise ValueError("self_forced_gan.manual_accumulate_grad_batches must be >= 1.")
        self.gan_checkpoint_discriminator = bool(
            _cfg(self.self_forced_gan_config, "checkpoint_discriminator", False)
        )
        self.gan_resample_fake_for_generator = bool(
            _cfg(self.self_forced_gan_config, "resample_fake_for_generator", False)
        )
        self.gan_ema_weight = float(_cfg(self.self_forced_gan_config, "ema_weight", 0.99))
        self.gan_ema_start_step = int(_cfg(self.self_forced_gan_config, "ema_start_step", 50))
        self.gan_student_unfrozen_range = str(
            _cfg(self.self_forced_gan_config, "student_unfrozen_range", "full_flow_decoder")
        )
        self.self_forced_sampling = _cfg(
            self.self_forced_gan_config,
            "sampling",
            self.validation_rollout_sampling,
        )
        self.self_forced_detach_block_transition = bool(
            _cfg(self.self_forced_gan_config, "detach_block_transition", False)
        )
        self.self_forced_use_stop_motion = bool(
            _cfg(self.self_forced_gan_config, "use_stop_motion", False)
        )

        position_type_scale = tuple(
            float(value) for value in getattr(model_config, "wosac_distribution_type_scale")
        )
        self.gan_teacher_cache = TeacherRolloutCache(
            self.gan_cache_root,
            n_teacher_rollout=self.gan_teacher_cache_size,
            rollout_set_size=self.gan_rollout_set_size,
        )
        self.gan_discriminator = SelfForcedGANDiscriminator(
            hidden_dim=128,
            n_rollout=self.gan_rollout_set_size,
            n_step=self.flow_window_steps,
            position_type_scale=position_type_scale,
            interaction_radius_m=float(model_config.decoder.a2a_radius),
        )
        self.gan_generator_ema = copy.deepcopy(self.encoder)
        self.gan_generator_ema.requires_grad_(False)
        self.gan_generator_ema.eval()
        self.register_buffer(
            "gan_generator_update_count",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "gan_generator_ema_ready",
            torch.zeros((), dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "gan_discriminator_update_count",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "gan_warmup_update_count",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.gan_resolved_warmup_updates = self._resolve_gan_warmup_updates()
        apply_self_forced_unfrozen_range(self.encoder, self.gan_student_unfrozen_range)

    def _resolve_gan_warmup_updates(self) -> int:
        """scene exposure 기준 warmup update 수를 계산합니다.

        Returns:
            int: deterministic discriminator warmup update 수입니다.
        """
        effective_batch = max(1, int(self.gan_effective_scene_batch))
        raw_updates = math.ceil(float(self.gan_warmup_scene_exposure) / float(effective_batch))
        return int(
            min(
                max(raw_updates, int(self.gan_warmup_min_updates)),
                int(self.gan_warmup_max_updates),
            )
        )

    def _is_gan_active(self) -> bool:
        """현재 epoch에서 GAN fine-tuning을 사용할지 판단합니다.

        Returns:
            bool: GAN fine-tuning 활성 여부입니다.
        """
        return bool(self.self_forced_gan_enabled and int(self.current_epoch) >= self.gan_start_epoch)

    def _is_gan_warmup_active(self) -> bool:
        """현재 step이 discriminator warmup 구간인지 판단합니다.

        Returns:
            bool: warmup 활성 여부입니다.
        """
        return bool(int(self.gan_warmup_update_count.item()) < int(self.gan_resolved_warmup_updates))

    def _copy_online_generator_to_gan_ema(self) -> None:
        """online generator weight를 GAN EMA generator에 복사합니다."""
        self.gan_generator_ema.load_state_dict(self.encoder.state_dict())
        self.gan_generator_ema.requires_grad_(False)
        self.gan_generator_ema.eval()

    @torch.no_grad()
    def _update_gan_generator_ema_after_step(self) -> None:
        """GAN generator optimizer step 뒤 EMA를 갱신합니다."""
        self.gan_generator_update_count.add_(1)
        if int(self.gan_generator_update_count.item()) < int(self.gan_ema_start_step):
            return
        if not bool(self.gan_generator_ema_ready.item()):
            self._copy_online_generator_to_gan_ema()
            self.gan_generator_ema_ready.fill_(True)
            return
        ema_weight = float(self.gan_ema_weight)
        online_state = self.encoder.state_dict()
        ema_state = self.gan_generator_ema.state_dict()
        for name, ema_value in ema_state.items():
            online_value = online_state[name].detach().to(device=ema_value.device)
            if torch.is_floating_point(ema_value):
                ema_value.mul_(ema_weight).add_(online_value.to(dtype=ema_value.dtype), alpha=1.0 - ema_weight)
            else:
                ema_value.copy_(online_value.to(dtype=ema_value.dtype))
        self.gan_generator_ema.eval()

    def _get_eval_generator(self):
        """validation/test에서 쓸 generator를 반환합니다."""
        if self.self_forced_gan_enabled and bool(self.gan_generator_ema_ready.item()):
            return self.gan_generator_ema
        return super()._get_eval_generator()

    def _build_gan_batch_context(
        self,
        data,
        tokenized_agent: Dict[str, Tensor],
        rollout_cache: Dict[str, object],
        map_feature: Dict[str, Tensor],
    ) -> Dict[str, Tensor | int]:
        """discriminator에 필요한 padded scene context를 만듭니다.

        Args:
            data: 학습 batch입니다.
            tokenized_agent: eval mode agent token입니다.
            rollout_cache: encoder rollout cache입니다.
            map_feature: frozen pretrained scene encoder가 만든 map token/geometry입니다.

        Returns:
            Dict[str, Tensor | int]: discriminator 입력 context입니다.
        """
        scenario_ids = data["scenario_id"]
        batch_size = len(scenario_ids)
        agent_batch = data["agent"]["batch"].to(device=self.device)
        n_per_scene = torch.bincount(agent_batch, minlength=batch_size)
        n_max_agent = int(n_per_scene.max().item()) if batch_size > 0 else 0
        current_pose_flat = build_current_pose_from_data(
            data,
            num_historical_steps=self.num_historical_steps,
        ).to(device=self.device)
        current_pose = pack_flat_agent_tensor(
            current_pose_flat,
            agent_batch,
            batch_size=batch_size,
            n_max_agent=n_max_agent,
        )
        agent_type_flat = tokenized_agent.get("type", data["agent"]["type"]).to(device=self.device)
        agent_type = pack_flat_agent_tensor(
            agent_type_flat.long(),
            agent_batch,
            batch_size=batch_size,
            n_max_agent=n_max_agent,
        ).long()
        current_valid_flat = data["agent"]["valid_mask"][:, self.num_historical_steps - 1].to(device=self.device)
        valid_mask = pack_flat_agent_tensor(
            current_valid_flat.bool(),
            agent_batch,
            batch_size=batch_size,
            n_max_agent=n_max_agent,
        ).bool()
        feat_a_now = rollout_cache["feat_a_now"]
        if not torch.is_tensor(feat_a_now):
            raise TypeError("rollout_cache['feat_a_now'] must be a Tensor.")
        agent_context = pack_flat_agent_tensor(
            feat_a_now.detach(),
            agent_batch,
            batch_size=batch_size,
            n_max_agent=n_max_agent,
        )
        map_batch = map_feature["batch"].to(device=self.device)
        n_per_map = torch.bincount(map_batch, minlength=batch_size)
        n_max_map = int(n_per_map.max().item()) if batch_size > 0 and int(map_batch.numel()) > 0 else 0
        map_context = pack_flat_agent_tensor(
            map_feature["pt_token"].detach().to(device=self.device),
            map_batch,
            batch_size=batch_size,
            n_max_agent=n_max_map,
        )
        map_position = pack_flat_agent_tensor(
            map_feature["position"].detach().to(device=self.device),
            map_batch,
            batch_size=batch_size,
            n_max_agent=n_max_map,
        )
        map_orientation = pack_flat_agent_tensor(
            map_feature["orientation"].detach().to(device=self.device),
            map_batch,
            batch_size=batch_size,
            n_max_agent=n_max_map,
        )
        map_valid_flat = torch.ones_like(map_batch, dtype=torch.bool, device=self.device)
        map_valid_mask = pack_flat_agent_tensor(
            map_valid_flat,
            map_batch,
            batch_size=batch_size,
            n_max_agent=n_max_map,
        ).bool()
        return {
            "batch_size": batch_size,
            "n_max_agent": n_max_agent,
            "n_max_map": n_max_map,
            "agent_batch": agent_batch,
            "current_pose": current_pose,
            "agent_type": agent_type,
            "valid_mask": valid_mask,
            "agent_context": agent_context,
            "map_context": map_context,
            "map_position": map_position,
            "map_orientation": map_orientation,
            "map_valid_mask": map_valid_mask,
        }

    def _load_gan_real_set(self, data, context: Dict[str, Tensor | int]) -> Tensor:
        """offline teacher cache에서 real rollout set을 읽습니다.

        Args:
            data: 학습 batch입니다.
            context: ``_build_gan_batch_context`` 반환값입니다.

        Returns:
            Tensor: teacher rollout set입니다. shape은 ``[B, K, 20, N, 4]`` 입니다.
        """
        real_pose, teacher_valid, _ = self.gan_teacher_cache.load_batch(
            scenario_ids=data["scenario_id"],
            batch_agent_id=data["agent"]["id"].to(device=self.device),
            batch_agent_type=data["agent"]["type"].to(device=self.device),
            batch_agent_batch=context["agent_batch"],
            n_max_agent=int(context["n_max_agent"]),
            device=self.device,
            dtype=context["current_pose"].dtype,
        )
        context["valid_mask"] = context["valid_mask"] & teacher_valid
        return real_pose

    def _sample_gan_fake_set(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        context: Dict[str, Tensor | int],
    ) -> Tensor:
        """student closed-loop rollout set K개를 생성합니다.

        Args:
            tokenized_map: eval mode map token입니다.
            tokenized_agent: eval mode agent token입니다.
            context: discriminator context입니다.

        Returns:
            Tensor: fake rollout set입니다. shape은 ``[B, K, 20, N, 4]`` 입니다.
        """
        fake_items: list[Tensor] = []
        for _ in range(int(self.gan_rollout_set_size)):
            rollout = self._run_self_forced_rollout(tokenized_map, tokenized_agent)
            pred_traj = rollout["pred_traj_10hz"][:, : self.flow_window_steps, :]
            pred_head = rollout["pred_head_10hz"][:, : self.flow_window_steps]
            fake_pose = pack_rollout_prediction_to_set(
                pred_traj=pred_traj,
                pred_head=pred_head,
                batch_index=context["agent_batch"],
                batch_size=int(context["batch_size"]),
                n_max_agent=int(context["n_max_agent"]),
            )
            fake_items.append(fake_pose)
        return torch.stack(fake_items, dim=1).contiguous()

    def _gan_forward_discriminator(self, rollout_pose: Tensor, context: Dict[str, Tensor | int]) -> Tensor:
        """discriminator forward를 실행합니다.

        Args:
            rollout_pose: rollout set입니다. shape은 ``[B, K, 20, N, 4]`` 입니다.
            context: discriminator context입니다.

        Returns:
            Tensor: logit입니다. shape은 ``[B, 1]`` 입니다.
        """
        return self.gan_discriminator(
            rollout_pose,
            current_pose=context["current_pose"],
            agent_type=context["agent_type"],
            valid_mask=context["valid_mask"],
            agent_context=context["agent_context"],
            map_context=context["map_context"],
            map_position=context["map_position"],
            map_orientation=context["map_orientation"],
            map_valid_mask=context["map_valid_mask"],
        )

    def _gan_forward_discriminator_for_backward(
        self,
        rollout_pose: Tensor,
        context: Dict[str, Tensor | int],
    ) -> Tensor:
        """Run discriminator forward with optional activation checkpointing."""
        if not self.gan_checkpoint_discriminator or not torch.is_grad_enabled():
            return self._gan_forward_discriminator(rollout_pose, context)

        def forward_fn(pose: Tensor) -> Tensor:
            return self._gan_forward_discriminator(pose, context)

        return checkpoint(forward_fn, rollout_pose, use_reentrant=False)

    def _compute_gan_finite_difference_regularizer(
        self,
        rollout_pose: Tensor,
        context: Dict[str, Tensor | int],
        *,
        base_logit: Tensor | None = None,
    ) -> Tensor:
        """finite-difference 방식 discriminator smoothness penalty를 계산합니다.

        Args:
            rollout_pose: teacher 또는 student rollout set입니다. shape은
                ``[B, K, 20, N, 4]`` 입니다.
            context: discriminator context입니다.
            base_logit: 이미 계산된 ``D(rollout_pose)``입니다. 같은 D update 안에서
                base score를 공유할 때 사용합니다.

        Returns:
            Tensor: finite-difference smoothness penalty scalar입니다.
        """
        scale = self.gan_discriminator.position_type_scale
        current_pose = context["current_pose"]
        agent_type = context["agent_type"]
        rollout_pose = rollout_pose.detach()
        base = (
            self._gan_forward_discriminator_for_backward(rollout_pose, context)
            if base_logit is None
            else base_logit
        )
        perturbed = add_rollout_pose_perturbation(
            rollout_pose,
            current_pose=current_pose,
            agent_type=agent_type,
            position_type_scale=scale,
            position_sigma=self.gan_position_sigma,
            yaw_sigma=self.gan_yaw_sigma,
        )
        perturbed_logit = self._gan_forward_discriminator_for_backward(perturbed, context)
        return ((perturbed_logit - base) / max(self.gan_position_sigma, 1.0e-6)).square().mean()

    def _gan_should_step_accumulated_optimizer(self, batch_idx: int) -> bool:
        """manual optimization에서 gradient accumulation step 경계를 판단합니다."""
        accumulate = max(1, int(self.gan_manual_accumulate_grad_batches))
        if (int(batch_idx) + 1) % accumulate == 0:
            return True
        return bool(getattr(self.trainer, "is_last_batch", False))

    def _gan_is_accumulation_start(self, batch_idx: int) -> bool:
        """현재 microbatch가 accumulation window 시작인지 판단합니다."""
        accumulate = max(1, int(self.gan_manual_accumulate_grad_batches))
        return int(batch_idx) % accumulate == 0

    def _gan_backward_sync_context(self, should_step: bool):
        """manual accumulation backward context입니다.

        V100x4x2에서는 DDP ``no_sync``가 non-step microbatch의 bucket memory peak를
        키워 rank별 OOM을 만들 수 있어 sync는 매 microbatch 유지합니다.
        """
        return nullcontext()

    def _training_step_self_forced_gan(self, data, batch_idx):
        """set-level GAN fine-tuning 한 step을 실행합니다.

        Args:
            data: 학습 batch입니다.
            batch_idx: batch index입니다.

        Returns:
            Tensor: logging용 detached loss입니다.
        """
        tokenized_map, tokenized_agent = self._build_eval_tokenized_inputs(data)
        map_feature = self.encoder.encode_map(tokenized_map)
        rollout_cache = self.encoder.prepare_training_rollout_cache(tokenized_agent, map_feature)
        context = self._build_gan_batch_context(data, tokenized_agent, rollout_cache, map_feature)
        real_pose = self._load_gan_real_set(data, context)

        warmup_active = self._is_gan_warmup_active()
        if warmup_active or self.gan_resample_fake_for_generator:
            with torch.no_grad():
                fake_pose_for_d = self._sample_gan_fake_set(tokenized_map, tokenized_agent, context)
        else:
            fake_pose_for_d = self._sample_gan_fake_set(tokenized_map, tokenized_agent, context)

        generator_optimizer, discriminator_optimizer = self.optimizers()
        accumulate = max(1, int(self.gan_manual_accumulate_grad_batches))
        accumulation_start = self._gan_is_accumulation_start(batch_idx)
        step_accumulated = self._gan_should_step_accumulated_optimizer(batch_idx)
        r1_log = real_pose.new_zeros(())
        r2_log = real_pose.new_zeros(())

        self.toggle_optimizer(discriminator_optimizer)
        try:
            if accumulation_start:
                discriminator_optimizer.zero_grad(set_to_none=True)

            real_pose_for_d = real_pose.detach()
            fake_pose_for_d_detached = fake_pose_for_d.detach()
            real_logit = self._gan_forward_discriminator_for_backward(real_pose_for_d, context)
            fake_logit = self._gan_forward_discriminator_for_backward(fake_pose_for_d_detached, context)
            d_adv = relativistic_discriminator_loss(real_logit, fake_logit)
            real_logit_for_margin = real_logit.detach()
            fake_logit_for_margin = fake_logit.detach()
            d_total = d_adv
            d_loss = d_adv.detach()

            if self.gan_r1_weight > 0.0:
                r1 = self._compute_gan_finite_difference_regularizer(
                    real_pose_for_d,
                    context,
                    base_logit=real_logit,
                )
                r1_log = r1.detach()
                d_total = d_total + self.gan_r1_weight * r1
                d_loss = d_loss + self.gan_r1_weight * r1_log
            if self.gan_r2_weight > 0.0:
                r2 = self._compute_gan_finite_difference_regularizer(
                    fake_pose_for_d_detached,
                    context,
                    base_logit=fake_logit,
                )
                r2_log = r2.detach()
                d_total = d_total + self.gan_r2_weight * r2
                d_loss = d_loss + self.gan_r2_weight * r2_log
            with self._gan_backward_sync_context(step_accumulated):
                self._manual_backward_without_autocast(d_total / accumulate)
            del real_logit, fake_logit, d_adv, d_total, real_pose_for_d, fake_pose_for_d_detached

            if step_accumulated:
                self._clip_and_step_with_optional_scaler(
                    discriminator_optimizer,
                    gradient_clip_val=self.gan_gradient_clip_val,
                    gradient_clip_algorithm="norm",
                )
                self.gan_discriminator_update_count.add_(1)
                if warmup_active:
                    self.gan_warmup_update_count.add_(1)
        finally:
            if step_accumulated:
                discriminator_optimizer.zero_grad(set_to_none=True)
            self.untoggle_optimizer(discriminator_optimizer)

        if warmup_active:
            total_loss = d_loss.detach()
            g_loss = real_pose.new_zeros(())
        else:
            if self.gan_resample_fake_for_generator:
                del fake_pose_for_d
                fake_pose = self._sample_gan_fake_set(tokenized_map, tokenized_agent, context)
            else:
                fake_pose = fake_pose_for_d
            self.toggle_optimizer(generator_optimizer)
            try:
                if accumulation_start:
                    generator_optimizer.zero_grad(set_to_none=True)
                with frozen_parameters(self.gan_discriminator):
                    real_logit_for_g = self._gan_forward_discriminator(real_pose.detach(), context).detach()
                    fake_logit_for_g = self._gan_forward_discriminator_for_backward(fake_pose, context)
                    g_loss = relativistic_generator_loss(real_logit_for_g, fake_logit_for_g)
                with self._gan_backward_sync_context(step_accumulated):
                    self._manual_backward_without_autocast(g_loss / accumulate)
                if step_accumulated:
                    self._clip_and_step_with_optional_scaler(
                        generator_optimizer,
                        gradient_clip_val=self.gan_gradient_clip_val,
                        gradient_clip_algorithm="norm",
                    )
                    self._update_gan_generator_ema_after_step()
            finally:
                if step_accumulated:
                    generator_optimizer.zero_grad(set_to_none=True)
                self.untoggle_optimizer(generator_optimizer)
            total_loss = g_loss.detach()

        margin = (real_logit_for_margin - fake_logit_for_margin).mean()
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/gan/d_loss", d_loss.detach(), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/gan/g_loss", g_loss.detach(), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/gan/d_margin", margin, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/gan/r1", r1_log, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/gan/r2", r2_log, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/gan/warmup_active", float(warmup_active), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/gan/warmup_updates", float(self.gan_resolved_warmup_updates), on_step=False, on_epoch=True, sync_dist=False, batch_size=1)
        self.log("train/gan/manual_accumulate_grad_batches", float(accumulate), on_step=False, on_epoch=True, sync_dist=False, batch_size=1)
        self.log("train/gan/optimizer_step_boundary", float(step_accumulated), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
        return total_loss

    def training_step(self, data, batch_idx):
        """GAN fine-tuning 또는 기존 SMARTFlow training_step을 실행합니다."""
        if self.self_forced_gan_enabled and self._is_gan_active():
            return self._training_step_self_forced_gan(data, batch_idx)
        return super().training_step(data, batch_idx)

    def configure_optimizers(self):
        """GAN fine-tuning용 optimizer 두 개를 만듭니다."""
        if not self.self_forced_gan_enabled:
            return super().configure_optimizers()
        apply_self_forced_unfrozen_range(self.encoder, self.gan_student_unfrozen_range)
        generator_params = [param for param in self.encoder.parameters() if param.requires_grad]
        if not generator_params:
            raise RuntimeError("No trainable generator parameters found for GAN fine-tuning.")
        discriminator_params = [param for param in self.gan_discriminator.parameters() if param.requires_grad]
        if not discriminator_params:
            raise RuntimeError("No trainable discriminator parameters found for GAN fine-tuning.")
        generator_optimizer = torch.optim.AdamW(
            generator_params,
            lr=self.gan_student_lr,
            betas=(0.0, 0.999),
            weight_decay=0.0,
        )
        discriminator_optimizer = torch.optim.AdamW(
            discriminator_params,
            lr=self.gan_discriminator_lr,
            betas=(0.0, 0.99),
            weight_decay=0.01,
        )
        return [generator_optimizer, discriminator_optimizer]

    def on_fit_start(self) -> None:
        """fit 시작 시 validation budget, EMA, warmup 정보를 준비합니다."""
        super().on_fit_start()
        if not self.self_forced_gan_enabled:
            return
        self.gan_resolved_warmup_updates = self._resolve_gan_warmup_updates()
        if not bool(self.gan_generator_ema_ready.item()):
            self._copy_online_generator_to_gan_ema()
        if getattr(self.trainer, "is_global_zero", True):
            print(
                "[self_forced_gan] resolved_warmup_updates="
                f"{self.gan_resolved_warmup_updates}, "
                f"effective_scene_batch={self.gan_effective_scene_batch}, "
                f"manual_accumulate_grad_batches={self.gan_manual_accumulate_grad_batches}, "
                f"checkpoint_discriminator={self.gan_checkpoint_discriminator}, "
                f"resample_fake_for_generator={self.gan_resample_fake_for_generator}, "
                f"critic_trainable_params={self.gan_discriminator.count_trainable_parameters()}",
                flush=True,
            )
