#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALPASIM_ROOT="${ALPASIM_ROOT:-$ROOT/alpasim}"
ALPASIM_UPSTREAM_URL="${ALPASIM_UPSTREAM_URL:-https://github.com/NVlabs/alpasim.git}"
ALPASIM_UPSTREAM_REF="${ALPASIM_UPSTREAM_REF:-v2026.4}"

if [[ -d "$ALPASIM_ROOT/src/driver" && -d "$ALPASIM_ROOT/src/wizard" ]]; then
  echo "Using existing AlpaSim checkout at $ALPASIM_ROOT"
else
  if [[ -e "$ALPASIM_ROOT" && -n "$(find "$ALPASIM_ROOT" -mindepth 1 -maxdepth 1 2>/dev/null)" ]]; then
    echo "ALPASIM_ROOT exists but is not a usable checkout: $ALPASIM_ROOT" >&2
    echo "Expected src/driver and src/wizard, or an empty directory." >&2
    exit 1
  fi
  rm -rf "$ALPASIM_ROOT"
  echo "Cloning $ALPASIM_UPSTREAM_URL @ $ALPASIM_UPSTREAM_REF into $ALPASIM_ROOT"
  git clone --branch "$ALPASIM_UPSTREAM_REF" --depth 1 "$ALPASIM_UPSTREAM_URL" "$ALPASIM_ROOT"
fi

export ALPASIM_ROOT
"$ROOT/scripts/bootstrap_alpasim_env.sh"
