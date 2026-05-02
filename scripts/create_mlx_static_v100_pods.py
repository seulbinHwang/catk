#!/usr/bin/env python3
"""Create long-lived MLX V100 pods for static multi-node CAT-K training."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import textwrap
import time


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_PODS = ["testsv", "testsvv", "testsvvv", "testsvvvv"]
DEFAULT_ZONE = "private-v100-naverlabs-0"
DEFAULT_IMAGE = "labs-ad2flow.n3r.reg.navercorp.com/mlx_exp/pnc_traffic_model:20260121"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_BRANCH = "semi_continuous_track_loss"


def shq(value: object) -> str:
    return shlex.quote(str(value))


def run(cmd: list[str], *, input_text: str | None = None, capture: bool = False, check: bool = True) -> str:
    result = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=check,
    )
    return result.stdout.strip() if capture and result.stdout else ""


def kubectl(args: list[str], *, input_text: str | None = None, capture: bool = False, check: bool = True) -> str:
    return run(["kubectl", *args], input_text=input_text, capture=capture, check=check)


def log(message: str) -> None:
    print(f"[create-pods] {message}", flush=True)


def exists(namespace: str, pod: str) -> bool:
    result = subprocess.run(
        ["kubectl", "get", "pod", pod, "-n", namespace],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def pod_json(namespace: str, pod: str) -> dict:
    text = kubectl(["get", "pod", pod, "-n", namespace, "-o", "json"], capture=True)
    return json.loads(text)


def pod_ready(namespace: str, pod: str) -> bool:
    try:
        data = pod_json(namespace, pod)
    except subprocess.CalledProcessError:
        return False
    for condition in data.get("status", {}).get("conditions", []):
        if condition.get("type") == "Ready" and condition.get("status") == "True":
            return True
    return False


def pod_node(namespace: str, pod: str) -> str:
    try:
        return pod_json(namespace, pod).get("spec", {}).get("nodeName", "") or ""
    except subprocess.CalledProcessError:
        return ""


def wait_deleted(namespace: str, pod: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not exists(namespace, pod):
            return
        time.sleep(3)
    raise TimeoutError(f"pod did not delete within {timeout}s: {pod}")


def delete_pod(namespace: str, pod: str, *, wait: bool = True) -> None:
    if not exists(namespace, pod):
        return
    kubectl(["delete", "pod", pod, "-n", namespace, "--ignore-not-found=true"], check=False)
    if wait:
        wait_deleted(namespace, pod, timeout=300)


def recent_events(namespace: str, pod: str) -> str:
    return kubectl(
        [
            "get",
            "events",
            "-n",
            namespace,
            "--field-selector",
            f"involvedObject.name={pod}",
            "--sort-by=.lastTimestamp",
        ],
        capture=True,
        check=False,
    )


def parse_profiles(text: str) -> list[tuple[str, str, str, str]]:
    profiles: list[tuple[str, str, str, str]] = []
    for item in text.split(","):
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 4:
            raise argparse.ArgumentTypeError(
                "profiles must be comma-separated cpu_request:cpu_limit:memory_request:memory_limit"
            )
        profiles.append((parts[0], parts[1], parts[2], parts[3]))
    if not profiles:
        raise argparse.ArgumentTypeError("at least one resource profile is required")
    return profiles


def render_pod_yaml(
    *,
    args: argparse.Namespace,
    pod: str,
    cpu_request: str,
    cpu_limit: str,
    memory_request: str,
    memory_limit: str,
) -> str:
    startup = f"""\
set -Eeo pipefail
umask 002
trap 'echo "Exit (code $?) - keeping pod alive for inspection"; sleep infinity' EXIT

export LANG=ko_KR.utf8
export LANGUAGE=ko_KR:ko
export LC_ALL=ko_KR.utf8
export LC_CTYPE=ko_KR.utf8
export TERM=xterm-256color

export PATH=/mnt/nuplan/miniforge/bin:$PATH
if [ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]; then
  source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
  conda activate catk 2>/dev/null || conda activate base 2>/dev/null || true
  echo "[startup] conda activated: $(conda info --envs | grep '*')"
else
  echo "[startup] /mnt/nuplan/miniforge not found - using image default python"
fi

if command -v pip &>/dev/null; then
  pip install --no-cache-dir wandb || echo "[startup] pip install wandb failed - continuing"
  if [ -n "${{WANDB_API_KEY:-}}" ]; then
    wandb login --relogin "$WANDB_API_KEY" || echo "[startup] wandb login failed - continuing"
  fi
else
  echo "[startup] pip not found - skipping wandb setup"
fi

mkdir -p "$HOME"

cat >> "$HOME/.bashrc" <<'RCEOF'
if [ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]; then
  source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
  conda activate catk 2>/dev/null || conda activate base 2>/dev/null || true
fi
RCEOF

if infocmp tmux-256color >/dev/null 2>&1; then
  TMUX_TERM="tmux-256color"
else
  TMUX_TERM="screen-256color"
fi

cat > "$HOME/.tmux.conf" <<EOF
set -g mouse on
set -g history-limit 100000
setw -g mode-keys vi
set -g default-terminal "$TMUX_TERM"
set -g default-command "/bin/bash"
EOF

if [ ! -d "$PROJECT_ROOT/.git" ]; then
  echo "[startup] PROJECT_ROOT not found - cloning repository into emptyDir"
  mkdir -p "$(dirname "$PROJECT_ROOT")"
  git clone https://github.com/seulbinHwang/catk.git "$PROJECT_ROOT" || true
fi

if [ ! -d "$PROJECT_ROOT/.git" ]; then
  echo "[startup] PROJECT_ROOT is invalid: $PROJECT_ROOT"
  exit 1
fi

git config --global --add safe.directory "$PROJECT_ROOT"
cd "$PROJECT_ROOT"
if git fetch origin {shq(args.branch)}; then
  git checkout {shq(args.branch)} 2>/dev/null || git checkout -b {shq(args.branch)} origin/{args.branch}
  git pull --ff-only origin {shq(args.branch)} || true
else
  echo "[startup] git fetch failed - keeping current checkout"
fi

SESSION_NAME="npc-training-v100"
tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_ROOT" || true
tmux split-window -h -t "$SESSION_NAME:0" -c "$PROJECT_ROOT" || true
tmux select-layout -t "$SESSION_NAME" even-horizontal || true

echo "Started tmux session '$SESSION_NAME'."
echo "Claude path: $(command -v claude || echo unavailable)"
claude --version || true
echo "Headless login help: /mnt/nuplan/tools/claude-login-help"
/mnt/nuplan/tools/claude-login-help || true

sleep infinity
"""
    startup_indented = textwrap.indent(startup.rstrip(), "          ")
    return f"""apiVersion: v1
kind: Pod
metadata:
  name: {pod}
  namespace: {args.namespace}
  annotations:
    sidecar.istio.io/inject: "false"
spec:
  terminationGracePeriodSeconds: 120
  restartPolicy: OnFailure
  nodeSelector:
    mlx.navercorp.com/zone: {args.zone}
  imagePullSecrets:
    - name: {args.image_pull_secret}
  securityContext:
    fsGroup: 1000
  volumes:
    - name: nuplan-storage
      emptyDir: {{}}
    - name: dshm2
      emptyDir:
        medium: Memory
        sizeLimit: {args.shm_size}
  initContainers:
    - name: install-nubescli
      image: {args.image}
      securityContext:
        allowPrivilegeEscalation: false
      command:
        - /bin/bash
        - -c
        - |
          set -Eeuo pipefail
          umask 002
          mkdir -p /mnt/nuplan/tools /mnt/nuplan/logs
          if [ ! -s /mnt/nuplan/tools/nubescli ]; then
            echo "[init] nubescli not found - downloading..."
            if curl -fsSL -o /mnt/nuplan/tools/nubescli \\
              http://owfsrepo.navercorp.com/nubes/dist/nubescli_latest/linux/nubescli; then
              chmod +x /mnt/nuplan/tools/nubescli
            else
              echo "[init] failed to download nubescli - continuing with a stub wrapper"
              cat > /mnt/nuplan/tools/nubescli <<'EOF'
          #!/usr/bin/env bash
          echo "[nubescli wrapper] nubescli is unavailable in this pod." >&2
          exit 1
          EOF
              chmod +x /mnt/nuplan/tools/nubescli
            fi
          fi
      volumeMounts:
        - name: nuplan-storage
          mountPath: /mnt/nuplan
    - name: setup-miniforge
      image: {args.image}
      securityContext:
        allowPrivilegeEscalation: false
      resources:
        requests:
          cpu: "4"
          memory: "16Gi"
        limits:
          cpu: "4"
          memory: "16Gi"
      command:
        - /bin/bash
        - -c
        - |
          set -Eeuo pipefail
          umask 002
          MINIFORGE_DIR=/mnt/nuplan/miniforge
          if [ -x "$MINIFORGE_DIR/bin/conda" ]; then
            echo "[init] miniforge already installed - skip"
          else
            echo "[init] Installing Miniforge..."
            curl -fsSL -o /tmp/miniforge.sh \\
              https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
            bash /tmp/miniforge.sh -b -p "$MINIFORGE_DIR"
            rm -f /tmp/miniforge.sh
          fi
          export PATH="$MINIFORGE_DIR/bin:$PATH"
          source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
          conda activate base
          if conda env list | awk '{{print $1}}' | grep -qx catk; then
            echo "[init] catk env already exists - skip creation"
          else
            conda create -y -n catk python=3.11.9
          fi
          conda activate catk
          pip install --no-cache-dir wandb || echo "[init] wandb install failed - continuing"
          conda info --envs
      volumeMounts:
        - name: nuplan-storage
          mountPath: /mnt/nuplan
    - name: setup-claude-code
      image: {args.image}
      securityContext:
        allowPrivilegeEscalation: false
      resources:
        requests:
          cpu: "2"
          memory: "8Gi"
        limits:
          cpu: "2"
          memory: "8Gi"
      command:
        - /bin/bash
        - -c
        - |
          set -Eeuo pipefail
          umask 002
          export HOME=/mnt/nuplan/home
          export PATH="$HOME/.local/bin:/mnt/nuplan/tools:$PATH"
          mkdir -p /mnt/nuplan/tools "$HOME/.local/bin" "$HOME/.claude"
          chmod 700 "$HOME/.claude"
          INSTALL_LOG=$(mktemp)
          if ! curl -fsSL https://claude.ai/install.sh | bash 2>&1 | tee "$INSTALL_LOG"; then
            echo "[init] Claude Code installation failed" >&2
            exit 1
          fi
          for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
            touch "$rc"
            if ! grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' "$rc"; then
              printf '\\nexport PATH="$HOME/.local/bin:$PATH"\\n' >> "$rc"
            fi
          done
          if [ ! -x "$HOME/.local/bin/claude" ]; then
            echo "[init] expected Claude Code launcher not found at $HOME/.local/bin/claude" >&2
            find "$HOME" -maxdepth 5 -type f -name claude -perm -111 || true
            exit 1
          fi
          ln -sf "$HOME/.local/bin/claude" /mnt/nuplan/tools/claude
          cat > /mnt/nuplan/tools/claude-login-help <<'EOF'
          #!/usr/bin/env bash
          cat <<'EOT'
          Claude Code headless first login:
          1. Run: claude
          2. Select: Claude account with subscription
          3. Open the printed login URL on a separate browser-enabled device
          4. Finish SSO there and copy the returned code
          5. Paste that code back into this terminal
          6. Verify with: claude
             Then run /status inside Claude Code if needed
          EOT
          EOF
          chmod +x /mnt/nuplan/tools/claude-login-help
          /mnt/nuplan/tools/claude --version
      volumeMounts:
        - name: nuplan-storage
          mountPath: /mnt/nuplan
  containers:
    - name: {args.container}
      image: {args.image}
      securityContext:
        allowPrivilegeEscalation: false
      env:
        - name: LANG
          value: "ko_KR.utf8"
        - name: LANGUAGE
          value: "ko_KR:ko"
        - name: LC_ALL
          value: "ko_KR.utf8"
        - name: LC_CTYPE
          value: "ko_KR.utf8"
        - name: TERM
          value: "xterm-256color"
        - name: PROJECT_ROOT
          value: "{args.project_root}"
        - name: HOME
          value: "/mnt/nuplan/home"
        - name: PATH
          value: "/mnt/nuplan/miniforge/envs/catk/bin:/mnt/nuplan/miniforge/bin:/mnt/nuplan/tools:/mnt/nuplan/home/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        - name: NUBES_GATEWAY_ADDRESS
          value: "c.nubes.sto.navercorp.com:8000"
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: wandb-secret
              key: api-key
        - name: WANDB_ENTITY
          value: "jksg01019-naver-labs"
        - name: WANDB_PROJECT
          value: "SMART-FLOW"
        - name: WANDB_MODE
          value: "online"
      command: ["/bin/bash", "-c"]
      args:
        - |
{startup_indented}
      resources:
        requests:
          cpu: "{cpu_request}"
          memory: "{memory_request}"
          nvidia.com/gpu: "{args.gpu_count}"
        limits:
          cpu: "{cpu_limit}"
          memory: "{memory_limit}"
          nvidia.com/gpu: "{args.gpu_count}"
      volumeMounts:
        - name: nuplan-storage
          mountPath: /mnt/nuplan
        - name: dshm2
          mountPath: /dev/shm
"""


def apply_profile(args: argparse.Namespace, pod: str, profile: tuple[str, str, str, str]) -> bool:
    cpu_request, cpu_limit, memory_request, memory_limit = profile
    log(
        f"creating {pod}: gpu={args.gpu_count}, cpu={cpu_request}/{cpu_limit}, "
        f"memory={memory_request}/{memory_limit}"
    )
    yaml_text = render_pod_yaml(
        args=args,
        pod=pod,
        cpu_request=cpu_request,
        cpu_limit=cpu_limit,
        memory_request=memory_request,
        memory_limit=memory_limit,
    )
    result = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=yaml_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        log(result.stdout.strip())
        return False
    log(result.stdout.strip())

    deadline = time.monotonic() + args.schedule_timeout
    while time.monotonic() < deadline:
        node = pod_node(args.namespace, pod)
        if node:
            log(f"{pod} scheduled on {node}")
            return True
        time.sleep(args.poll_interval)
    log(f"{pod} was not scheduled within {args.schedule_timeout}s")
    events = recent_events(args.namespace, pod)
    if events:
        log(f"recent events for {pod}:\n{events}")
    return False


def wait_ready(args: argparse.Namespace, pod: str) -> None:
    deadline = time.monotonic() + args.ready_timeout
    while time.monotonic() < deadline:
        if pod_ready(args.namespace, pod):
            log(f"{pod} is Ready")
            return
        phase = ""
        node = ""
        try:
            data = pod_json(args.namespace, pod)
            phase = data.get("status", {}).get("phase", "")
            node = data.get("spec", {}).get("nodeName", "")
        except subprocess.CalledProcessError:
            pass
        log(f"waiting for {pod} Ready: phase={phase or '?'} node={node or '?'}")
        time.sleep(args.ready_poll_interval)
    events = recent_events(args.namespace, pod)
    raise TimeoutError(f"{pod} did not become Ready within {args.ready_timeout}s\n{events}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--pods", nargs="+", default=DEFAULT_PODS)
    parser.add_argument("--container", default="main")
    parser.add_argument("--zone", default=DEFAULT_ZONE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--image-pull-secret", default="pnc-secret")
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument(
        "--resource-profiles",
        type=parse_profiles,
        default=parse_profiles("32:32:128Gi:480Gi,24:24:96Gi:384Gi,16:16:64Gi:256Gi"),
        help="comma-separated cpu_request:cpu_limit:memory_request:memory_limit profiles",
    )
    parser.add_argument("--shm-size", default="256Gi")
    parser.add_argument("--schedule-timeout", type=int, default=240)
    parser.add_argument("--ready-timeout", type=int, default=2400)
    parser.add_argument("--poll-interval", type=int, default=5)
    parser.add_argument("--ready-poll-interval", type=int, default=30)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for pod in args.pods:
        if exists(args.namespace, pod):
            if args.replace:
                log(f"{pod} already exists; deleting because --replace was set")
                delete_pod(args.namespace, pod)
            else:
                log(f"{pod} already exists; skipping create")
                continue

        created = False
        for profile in args.resource_profiles:
            if args.dry_run:
                print(
                    render_pod_yaml(
                        args=args,
                        pod=pod,
                        cpu_request=profile[0],
                        cpu_limit=profile[1],
                        memory_request=profile[2],
                        memory_limit=profile[3],
                    )
                )
                created = True
                break
            if apply_profile(args, pod, profile):
                created = True
                break
            delete_pod(args.namespace, pod)
        if not created:
            raise SystemExit(f"failed to create scheduled pod after all profiles: {pod}")

    if args.dry_run:
        return
    for pod in args.pods:
        wait_ready(args, pod)


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, TimeoutError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
