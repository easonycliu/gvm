#!/bin/bash

script_dir=$(dirname ${BASH_SOURCE[0]})
project_dir=$(realpath $script_dir/..)

$project_dir/3rdparty/xsched/output/bin/xserver HPF 50000
