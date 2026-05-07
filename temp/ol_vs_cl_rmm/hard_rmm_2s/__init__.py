# Hard-RMM 2초 변형 — non-differentiable hard RMM (`src/smart/metrics/`) 의 복사본을
# T_pred=20 (2초) horizon 으로 작동하게 최소 수정.
#
# 원본 hard RMM 은 8초 (n_simulation_steps=80) 가정으로 짜여 있어, short-horizon
# (예: OL/CL 2초 비교) 입력에 그대로 못 흘린다.  본 디렉토리는 그 코드를
# **`src/` 를 건드리지 않고** 복사해 다음 두 점만 변경:
#
#   1. `n_steps_override` 인자 추가:
#      ``cfg.n_simulation_steps`` 대신 사용해 GT future slicing / sim_traj
#      length 가정을 모두 ``n_steps_override`` (또는 None 이면 cfg) 로 통일.
#   2. ``compute_metric_features`` 안의 ``eval_logged`` / ``logged_full`` 의
#      시간축을 ``simulated.x`` 와 매치하도록 자동 슬라이스 (원본은 길이
#      불일치 시 ``compute_displacement_error`` 에서 crash).
#
# Histogram / metametric 본체 (``compute_wosac_metametric_from_features_torch``)
# 는 dim-dynamic 이라 그대로 사용 가능 — ``wosac_metametric_pytorch_2s.py`` 는
# 자기 완결성을 위해 함께 복사만 하고 변경 없음.

from temp.ol_vs_cl_rmm.hard_rmm_2s.metric_features_torch_2s import (
    compute_metric_features as compute_metric_features_2s,
    compute_scenario_rollouts_features as compute_scenario_rollouts_features_2s,
    scenario_to_joint_scene as scenario_to_joint_scene_2s,
)
from temp.ol_vs_cl_rmm.hard_rmm_2s.wosac_metametric_pytorch_2s import (
    compute_wosac_metametric_from_features_torch as compute_wosac_metametric_from_features_torch_2s,
)
