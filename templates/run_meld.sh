#!/usr/bin/env bash
# Launch only on an allocated HPC GPU, with the MELD environment activated.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -f Data/data_store.dat || ! -f setup_report.json ]]; then
    printf '%s\n' 'Run python system_setup.py to initialize this folder first.' >&2
    exit 1
fi
if [[ -z "${OPENMM_CUDA_COMPILER:-}" ]]; then
    if command -v nvcc >/dev/null 2>&1; then
        export OPENMM_CUDA_COMPILER="$(command -v nvcc)"
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        export OPENMM_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
    else
        printf '%s\n' \
            'CUDA compiler nvcc was not found. MELD requires the compiler as well as CUDA runtime libraries.' \
            'Load a CUDA toolkit module or install compiler packages matching your CUDA runtime.' \
            'Then export OPENMM_CUDA_COMPILER=/absolute/path/to/nvcc. See README.md.' >&2
        exit 1
    fi
fi
if [[ ! -x "$OPENMM_CUDA_COMPILER" ]]; then
    printf 'OPENMM_CUDA_COMPILER is not executable: %s\n' "$OPENMM_CUDA_COMPILER" >&2
    exit 1
fi
printf 'Using CUDA compiler: %s\n' "$OPENMM_CUDA_COMPILER"
python system_setup.py --check
exec launch_remd_multiplex --platform CUDA --debug "$@"
