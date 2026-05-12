# Delayed-Window Self-Forcing review fix

## Review result

`6184acb4` already implements the main delayed-window schedule:

| epoch | rollout | trained window |
|---:|---:|---:|
| 0~3 | 2s | 0~2s |
| 4~7 | 4s | 2~4s |
| 8~11 | 6s | 4~6s |
| 12~15 | 8s | 6~8s |

It also keeps the trained 2-second window connected while cutting the connection between the skipped part and the trained part.

## Required correction

The skipped part creates a delayed current state, but two context details must also move to that delayed current state:

1. The context motion token id must be chosen from the self-generated 0.5s chunk, not copied from the original GT-derived future slot.
2. The traffic-light elapsed-time feature must be recomputed from the delayed current time, then passed into map-agent relation encoding.

This patch applies only those two corrections. It does not add a new loss, random window, mixed window, or RMM-based loss.

## Apply

From the repository root:

```bash
unzip catk_delayed_self_forcing_review_fix.zip -d .
python tools/apply_delayed_self_forcing_review_patch.py
```

Then run the focused test:

```bash
pytest tests/test_self_forced_delayed_window.py
```
