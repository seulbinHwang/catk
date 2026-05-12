#!/usr/bin/env bash
# Code repo → Overleaf 단방향 figure sync.
# 절대 양방향 X. 절대 .tex 를 code repo 로 끌어오지 않음.
#
# 사용법:
#   bash scripts/sync_figs_to_overleaf.sh           # dry-run (변경 미리보기)
#   bash scripts/sync_figs_to_overleaf.sh --apply   # 실제 복사
#
# 전제:
#   - Code figures 는 project_3/paper_assets/figures/ 에 모아둠 (gitignored)
#   - Overleaf repo 는 /home2/pnc2/repos_overleaf/69fbf3e1697f670db6f6110c/ 에 clone 되어 있음
#   - Overleaf 측 figure 폴더는 figures/ (없으면 자동 생성)
#
# 운영 규칙:
#   - 이 스크립트는 항상 code → paper 방향만 작동
#   - Overleaf 측에서 figure 를 직접 수정하지 말 것 (이 스크립트가 덮어씀)
#   - sync 후 Overleaf repo 에서 git add/commit/push 는 사용자가 직접 (Overleaf 측 인증 분리)

set -euo pipefail

CODE_REPO="/home2/pnc2/repos_python/project_3"
PAPER_REPO="/home2/pnc2/repos_overleaf/69fbf3e1697f670db6f6110c"
SRC_DIR="${CODE_REPO}/paper_assets/figures"
DST_DIR="${PAPER_REPO}/figures"

if [[ ! -d "${PAPER_REPO}" ]]; then
    echo "[ERR] Overleaf repo not found at ${PAPER_REPO}" >&2
    echo "      먼저 clone: cd /home2/pnc2/repos_overleaf && git clone https://git@git.overleaf.com/69fbf3e1697f670db6f6110c 69fbf3e1697f670db6f6110c" >&2
    exit 1
fi

if [[ ! -d "${SRC_DIR}" ]]; then
    echo "[ERR] Figure source dir not found: ${SRC_DIR}" >&2
    echo "      mkdir -p ${SRC_DIR} 후 사용" >&2
    exit 1
fi

# Safety: Overleaf repo 의 remote 가 overleaf 만 가리키는지 확인 (GitHub 섞임 방지)
if ! (cd "${PAPER_REPO}" && git remote -v | grep -q "overleaf.com"); then
    echo "[ERR] ${PAPER_REPO} 의 remote 에 overleaf.com 이 없음. clone 위치 확인 필요." >&2
    exit 1
fi
if (cd "${PAPER_REPO}" && git remote -v | grep -qi "github.com"); then
    echo "[ERR] ${PAPER_REPO} remote 에 github.com 이 섞여있음. 즉시 정리 필요." >&2
    exit 1
fi

mkdir -p "${DST_DIR}"

RSYNC_FLAGS=(-av --delete --include='*/' --include='*.pdf' --include='*.png' --include='*.jpg' --include='*.svg' --exclude='*')

if [[ "${1:-}" == "--apply" ]]; then
    echo "[apply] rsync ${SRC_DIR}/ → ${DST_DIR}/"
    rsync "${RSYNC_FLAGS[@]}" "${SRC_DIR}/" "${DST_DIR}/"
    echo "[done] Overleaf 측에서 commit + push 는 직접: "
    echo "       cd ${PAPER_REPO} && git status"
else
    echo "[dry-run] 실제 적용하려면 --apply"
    rsync "${RSYNC_FLAGS[@]}" --dry-run "${SRC_DIR}/" "${DST_DIR}/"
fi
