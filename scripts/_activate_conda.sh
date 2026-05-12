#!/usr/bin/env sh

_catk_conda_env="${CATK_CONDA_ENV:-catk}"

if [ "${CONDA_DEFAULT_ENV:-}" = "${_catk_conda_env}" ]; then
  unset _catk_conda_env
  return 0
fi

_conda_base=""
if command -v conda >/dev/null 2>&1; then
  _conda_base="$(conda info --base 2>/dev/null || true)"
elif [ -n "${CONDA_EXE:-}" ]; then
  _conda_base="$(dirname "$(dirname "${CONDA_EXE}")")"
fi

if [ -n "${_conda_base}" ] && [ -f "${_conda_base}/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  . "${_conda_base}/etc/profile.d/conda.sh"
  conda activate "${_catk_conda_env}"
elif [ -d "/media/user/E/miniforge/envs/${_catk_conda_env}" ]; then
  export CONDA_PREFIX="/media/user/E/miniforge/envs/${_catk_conda_env}"
  export CONDA_DEFAULT_ENV="${_catk_conda_env}"
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
else
  echo "Failed to activate conda env: ${_catk_conda_env}" >&2
  echo "Set CATK_CONDA_ENV or initialize conda before running this script." >&2
  unset _catk_conda_env _conda_base
  return 1
fi

if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/bin" ]; then
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
fi

unset _catk_conda_env _conda_base
