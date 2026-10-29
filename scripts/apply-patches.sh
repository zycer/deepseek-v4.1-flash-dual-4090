#!/usr/bin/env bash
# Extract the patch targets from the image, apply the diffs, and leave an overlay tree
# that launch.sh bind-mounts. The image itself is never modified.
#
#   IMAGE=... ./apply-patches.sh
#
# Writes to $PATCH_DIR (default <repo>/patched), mirroring site-packages layout:
#   patched/vllm/models/deepseek_v4_1/common/engram.py
#   ...
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
IMAGE="${IMAGE:?set IMAGE to the vLLM image}"
PATCH_DIR="${PATCH_DIR:-$ROOT/patched}"
SP=/opt/venv/lib/python3.12/site-packages

command -v patch >/dev/null || { echo "error: 'patch' not installed" >&2; exit 1; }
command -v docker >/dev/null || { echo "error: 'docker' not installed" >&2; exit 1; }

shopt -s nullglob
patches=("$ROOT"/patches/*.patch)
shopt -u nullglob
[ "${#patches[@]}" -gt 0 ] || { echo "error: no patches found in $ROOT/patches" >&2; exit 1; }

echo "image : $IMAGE"
echo "output: $PATCH_DIR"
echo

# A created-but-not-started container is enough for docker cp and costs no GPU.
cid=$(docker create "$IMAGE")
cleanup() { docker rm -f "$cid" >/dev/null 2>&1 || true; }
trap cleanup EXIT

mkdir -p "$PATCH_DIR"
ok=0; fail=0

for p in "${patches[@]}"; do
  name="$(basename "$p")"
  # The +++ b/<path> header is the authoritative target; no hardcoded mapping to drift.
  rel=$(sed -n 's|^+++ b/||p' "$p" | head -n1)
  if [ -z "$rel" ]; then
    printf '  %-46s SKIP (no +++ header)\n' "$name"; fail=$((fail+1)); continue
  fi

  dest="$PATCH_DIR/$rel"
  mkdir -p "$(dirname "$dest")"

  if ! docker cp "$cid:$SP/$rel" "$dest" 2>/dev/null; then
    printf '  %-46s FAIL (not found in image: %s)\n' "$name" "$rel"
    fail=$((fail+1)); continue
  fi

  # --no-backup-if-mismatch: leave no .orig files behind; fuzz 0: no silent drift
  if ( cd "$PATCH_DIR" && patch -p1 --no-backup-if-mismatch --fuzz=0 -s -i "$p" ); then
    printf '  %-46s OK\n' "$name"
    ok=$((ok+1))
  else
    printf '  %-46s FAIL (context does not match this image)\n' "$name"
    fail=$((fail+1))
  fi
done

echo
echo "applied: ${ok}   failed: ${fail}"

if [ "${fail}" -gt 0 ]; then
  cat >&2 <<'EOF'

At least one patch did not apply. Most likely causes:
  * the image tag differs from the one the patches were cut against
    (vllm 0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130)
  * upstream moved the code, so line numbers shifted

The hunks carry generous context, so `--fuzz=2` usually lands them anyway:
  ( cd $PATCH_DIR && patch -p1 --fuzz=2 -i <patch> )
Re-verify the result against docs/02-patches.md before booting.
EOF
  exit 1
fi

echo
echo "overlay tree:"
find "$PATCH_DIR" -name '*.py' -printf '  %p  (%s bytes)\n' | sort
echo
echo "next: MASTER_ADDR=... MODEL_PATH=... IMAGE=... ./scripts/launch.sh 0"
