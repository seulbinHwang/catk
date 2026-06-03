"""wandb run 의 단일-scene RMM 추세를 평가해 RISING/PROMISING/FLAT 판정.

사용: python tools/eval_rmm_trend.py <wandb_run_id> [<entity/project>]
출력 한 줄: "<VERDICT> d=<Δ> n=<점수> first=<..> last=<..> slope=<..> cpd_d=<..>"
  RISING   : mean(last 1/3) - mean(first 1/3) > 0.008  (노이즈 위)
  PROMISING: 0.004 < d <= 0.008
  FLAT     : d <= 0.004
"""
from __future__ import annotations
import sys

RMM_KEY = "val_closed/sim_agents_2025/realism_meta_metric"
CPD_KEY = "val_closed/WOSAC-CPD/value"


def main() -> None:
    rid = sys.argv[1]
    proj = sys.argv[2] if len(sys.argv) > 2 else "se99an/clsft-catk"
    import wandb

    r = wandb.Api().run(f"{proj}/{rid}")
    rows = r.history(keys=[RMM_KEY, CPD_KEY, "_step"], samples=2000, pandas=False)
    pts = [(x.get("_step"), x.get(RMM_KEY), x.get(CPD_KEY)) for x in rows if x.get(RMM_KEY) is not None]
    pts.sort(key=lambda t: (t[0] if t[0] is not None else 0))
    rmms = [m for _, m, _ in pts]
    cpds = [c for _, _, c in pts if c is not None]
    n = len(rmms)
    if n < 6:
        print(f"TOOFEW d=0 n={n} first=NA last=NA slope=0 cpd_d=0")
        return
    k = max(2, n // 3)
    first = sum(rmms[:k]) / k
    last = sum(rmms[-k:]) / k
    d = last - first
    # 선형 회귀 기울기 (step 정규화)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(rmms) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, rmms))
    den = sum((x - mx) ** 2 for x in xs) or 1.0
    slope = num / den  # RMM per val-point
    cpd_d = (sum(cpds[-k:]) / k - sum(cpds[:k]) / k) if len(cpds) >= 2 * k else 0.0
    verdict = "RISING" if d > 0.008 else ("PROMISING" if d > 0.004 else "FLAT")
    print(
        f"{verdict} d={d:+.4f} n={n} first={first:.4f} last={last:.4f} "
        f"slope={slope:+.5f} cpd_d={cpd_d:+.4f}"
    )


if __name__ == "__main__":
    main()
