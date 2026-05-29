#!/bin/sh
set -eu

if [ "$#" -lt 1 ]; then
  echo "usage: $0 LOG_PATH [WINDOW_NAME] [SESSION]" >&2
  exit 2
fi

LOG_PATH="$1"
WINDOW_NAME="${2:-logs}"
SESSION="${3:-}"
WORKDIR="${TMUX_TAIL_WORKDIR:-$(pwd)}"
TAIL_LINES="${TMUX_TAIL_LINES:-100}"

if ! command -v tmux >/dev/null 2>&1; then
  exit 0
fi

if [ -z "${SESSION}" ]; then
  SESSION="$(tmux display-message -p '#S' 2>/dev/null || true)"
fi
if [ -z "${SESSION}" ]; then
  exit 0
fi

mkdir -p "$(dirname "${LOG_PATH}")"
touch "${LOG_PATH}"

quote_sh() {
  printf "'%s'" "$(printf "%s" "$1" | sed "s/'/'\\\\''/g")"
}

CMD="LOG_PATH=$(quote_sh "${LOG_PATH}") TAIL_LINES=$(quote_sh "${TAIL_LINES}") sh -lc 'printf \"[tail] %s\\n\\n\" \"\$LOG_PATH\"; tail -n \"\$TAIL_LINES\" -F \"\$LOG_PATH\"; exec \"\${SHELL:-/bin/sh}\"'"

tmux new-window -d -t "${SESSION}:" -n "${WINDOW_NAME}" -c "${WORKDIR}" "${CMD}"
