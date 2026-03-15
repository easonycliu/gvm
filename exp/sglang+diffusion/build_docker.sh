#!/bin/bash

script_dir=$(dirname ${BASH_SOURCE[0]})
project_dir=$(realpath $script_dir/../..)

target=${1:-all}

build_sglang() {
    local image_name=${1:-sglang-tgs:latest}
    echo "=== Building sglang image: $image_name ==="
    docker build -t "$image_name" \
        -f "$script_dir/Dockerfile.sglang" \
        "$script_dir"
}

build_diffusion() {
    local image_name=${1:-diffusion-tgs:latest}
    echo "=== Building diffusion image: $image_name ==="
    docker build -t "$image_name" \
        -f "$script_dir/Dockerfile.diffusion" \
        "$script_dir"
}

case $target in
    sglang)
        build_sglang "$2"
        ;;
    diffusion)
        build_diffusion "$2"
        ;;
    all)
        build_sglang
        build_diffusion
        ;;
    *)
        echo "Usage: $0 [sglang|diffusion|all] [image_name]"
        exit 1
        ;;
esac
