#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:?Usage: prepare_local_text_corpus.sh OUT_DIR}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_FILE="$OUT_DIR/train.txt"
VALIDATION_FILE="$OUT_DIR/validation.txt"

mkdir -p "$OUT_DIR"

if [[ -s "$TRAIN_FILE" && -s "$VALIDATION_FILE" ]]; then
  exit 0
fi

tmp_train="$TRAIN_FILE.tmp.$$"
tmp_validation="$VALIDATION_FILE.tmp.$$"
: > "$tmp_train"
: > "$tmp_validation"

append_repeated() {
  local target="$1"
  local repeats="$2"
  shift 2

  local source
  local repeat
  for source in "$@"; do
    if [[ -f "$REPO_ROOT/$source" ]]; then
      for ((repeat = 1; repeat <= repeats; repeat++)); do
        printf '\n\n[source=%s repeat=%s]\n' "$source" "$repeat" >> "$target"
        sed 's/[`#*_<>]/ /g' "$REPO_ROOT/$source" >> "$target"
      done
    fi
  done
}

append_repeated "$tmp_train" 24 \
  "README.md" \
  "docs/source/en/model_doc/qwen3.md" \
  "docs/source/en/model_doc/qwen3_5.md" \
  "checkpoints/Qwen3-1.7B/README.md"

append_repeated "$tmp_validation" 12 \
  "tests/fixtures/sample_text_no_unicode.txt" \
  "tests/fixtures/sample_text.txt" \
  "docs/source/en/model_doc/qwen3_moe.md"

if [[ ! -s "$tmp_train" || ! -s "$tmp_validation" ]]; then
  echo "Unable to build local Qwen3 text corpus under $OUT_DIR" >&2
  rm -f "$tmp_train" "$tmp_validation"
  exit 1
fi

mv "$tmp_train" "$TRAIN_FILE"
mv "$tmp_validation" "$VALIDATION_FILE"
