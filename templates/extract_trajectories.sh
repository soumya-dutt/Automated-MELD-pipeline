#!/usr/bin/env bash
# Extract numeric replica DCDs from one simulation; does not run dynamics.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash extract_trajectories.sh [OPTIONS]
  --run-dir PATH       Simulation folder (default: directory of this script)
  --first-replica N    First index, inclusive (default: 0)
  --last-replica N     Last index, inclusive (default: saved/configured count - 1)
  --overwrite          Replace existing nonempty DCDs after successful extraction
  --dry-run            Print commands only; Data need not exist
  -h, --help           Show this help

Activate meld_new first. Outputs: 0.dcd, 1.dcd, ... inside the simulation folder.
Existing nonempty outputs are skipped by default. Log: extract_trajectories.log.
EOF
}

EXTRACT_RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXTRACT_FIRST=0
EXTRACT_LAST=""
EXTRACT_OVERWRITE=0
EXTRACT_DRY_RUN=0
while (( $# )); do
    case "$1" in
        --run-dir)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            EXTRACT_RUN_DIR="$2"; shift 2 ;;
        --first-replica|--last-replica)
            if (( $# < 2 )) || [[ ! "$2" =~ ^[0-9]+$ ]]; then
                printf '%s requires a nonnegative integer.\n' "$1" >&2; exit 2
            fi
            if [[ "$1" == --first-replica ]]; then
                EXTRACT_FIRST=$((10#$2))
            else
                EXTRACT_LAST=$((10#$2))
            fi
            shift 2 ;;
        --overwrite) EXTRACT_OVERWRITE=1; shift ;;
        --dry-run) EXTRACT_DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
EXTRACT_RUN_DIR="$(cd -- "$EXTRACT_RUN_DIR" && pwd)"
cd -- "$EXTRACT_RUN_DIR"
EXTRACT_COUNT="$(python - "$EXTRACT_RUN_DIR" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
config = json.loads((root / 'input.json').read_text())
count = config['n_replicas']
report = root / 'setup_report.json'
if report.exists():
    saved = json.loads(report.read_text())['effective_config']['n_replicas']
    if saved != count:
        sys.exit('Replica count differs between input.json and the saved setup. Restore the matching configuration.')
    count = saved
if isinstance(count, bool) or not isinstance(count, int) or count < 1:
    sys.exit('n_replicas must be a positive integer.')
print(count)
PY
)"
if [[ -z "$EXTRACT_LAST" ]]; then EXTRACT_LAST=$((EXTRACT_COUNT - 1)); fi
if (( EXTRACT_FIRST > EXTRACT_LAST || EXTRACT_LAST >= EXTRACT_COUNT )); then
    printf 'Invalid replica range %s-%s: this run has %s replicas, indexed 0-%s.\n' \
        "$EXTRACT_FIRST" "$EXTRACT_LAST" "$EXTRACT_COUNT" "$((EXTRACT_COUNT - 1))" >&2
    exit 2
fi
if (( ! EXTRACT_DRY_RUN )); then
    if [[ ! -f Data/data_store.dat ]]; then
        printf 'Missing Data/data_store.dat in %s. Run the simulation before extracting.\n' "$EXTRACT_RUN_DIR" >&2
        exit 1
    fi
    if ! command -v extract_trajectory >/dev/null 2>&1; then
        printf '%s\n' 'extract_trajectory is unavailable. Activate meld_new first.' >&2
        exit 1
    fi
fi

EXTRACT_DONE=0
EXTRACT_SKIPPED=0
EXTRACT_FAILED=0
printf 'Simulation: %s; replicas %s-%s\n' "$EXTRACT_RUN_DIR" "$EXTRACT_FIRST" "$EXTRACT_LAST"
for ((index=EXTRACT_FIRST; index<=EXTRACT_LAST; index++)); do
    filename="$index.dcd"
    if (( ! EXTRACT_OVERWRITE )) && [[ -s "$filename" ]]; then
        printf 'Skipping existing: %s\n' "$filename"
        EXTRACT_SKIPPED=$((EXTRACT_SKIPPED + 1))
        continue
    fi
    echo "Running: extract_trajectory extract_traj_dcd --replica $index $filename"
    if (( EXTRACT_DRY_RUN )); then continue; fi
    # Retain old outputs until the new extraction has succeeded.
    if (
        temporary=".${index}.extracting.$$.dcd"
        trap 'rm -f -- "$temporary"' EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM
        printf '\n[%s] Replica %s -> %s\n' "$(date -Is)" "$index" "$filename" >> extract_trajectories.log
        extract_trajectory extract_traj_dcd --replica "$index" "$temporary" >> extract_trajectories.log 2>&1 &&
            test -s "$temporary" && mv -f -- "$temporary" "$filename"
    ); then
        EXTRACT_DONE=$((EXTRACT_DONE + 1))
    else
        printf 'Failed: replica %s; see %s/extract_trajectories.log\n' "$index" "$EXTRACT_RUN_DIR" >&2
        EXTRACT_FAILED=$((EXTRACT_FAILED + 1))
    fi
done
if (( EXTRACT_DRY_RUN )); then
    printf '%s\n' 'Dry run complete; no files written.'
else
    printf 'Extracted: %s; skipped: %s; failed: %s.\n' "$EXTRACT_DONE" "$EXTRACT_SKIPPED" "$EXTRACT_FAILED"
fi
if (( EXTRACT_FAILED )); then exit 1; fi
