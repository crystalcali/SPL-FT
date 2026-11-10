#!/usr/bin/env bash
set -euo pipefail
if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "arm64" ]]; then exit 0; fi
if [[ ! -e /proc/sys/fs/binfmt_misc/qemu-x86_64 ]]; then
  printf 'Registering QEMU AMD64 emulation for Terminal-Bench task containers.\n'
  docker run --privileged --rm tonistiigi/binfmt:latest --install amd64
fi
docker run --rm --platform linux/amd64 alpine:3.21 uname -m
