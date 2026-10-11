#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 LOG_PATH COMMAND [ARG ...]" >&2
  exit 2
fi

log_path=$1
shift
mkdir -p "$(dirname "$log_path")"
if [[ -e "$log_path" ]]; then
  echo "refusing to overwrite existing nohup log: $log_path" >&2
  exit 1
fi

set -o noclobber
nohup "$@" >"$log_path" 2>&1 &
pid=$!
echo "started pid=$pid log=$log_path"
