#!/usr/bin/env sh

set -eu

for volume_name in \
    docling-cache \
    huggingface-cache \
    ollama-cache
do
    docker volume create "${volume_name}" >/dev/null
    printf "Ensured Docker volume exists: %s\n" "${volume_name}"
done
