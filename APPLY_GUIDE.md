# Apply guide

이 ZIP은 `seulbinHwang/catk` main 브랜치 기준 변경 파일을 같은 경로에 그대로 덮어쓰는 형태입니다.

## 적용

```bash
cd /path/to/catk
unzip /path/to/catk_main_wosac_distribution_metrics_patch.zip -d /tmp/catk_wosac_patch
cp -R /tmp/catk_wosac_patch/catk_wosac_distribution_main_patch/* .
```

## 반영 파일

- `src/smart/metrics/wosac_distribution_metrics.py`
- `src/smart/metrics/__init__.py`
- `src/smart/model/smart.py`
- `configs/model/smart.yaml`
- `README.md`
- `tests/test_wosac_distribution_metrics.py`

## 사용

closed-loop validation/test에서 rollout이 만들어진 뒤 자동 계산됩니다.

- validation: `val_closed/WOSAC-CPD/value`, `val_closed/WOSAC-CES/value`
- test submission export: `test/WOSAC-CPD/value`
- pretrain 기준 CPD가 있으면 `model.model_config.wosac_cpd_reference=<value>`로 DPR도 기록됩니다.

예시:

```bash
python -m src.run \
  experiment=local_val \
  ckpt_path=/path/to/checkpoint.ckpt \
  model.model_config.wosac_cpd_reference=<SMART_PRETRAIN_CPD>
```
