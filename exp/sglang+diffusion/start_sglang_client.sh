#!/bin/bash

script_dir=$(dirname ${BASH_SOURCE[0]})
project_dir=$(realpath $script_dir/../..)

pidfile=
model=
prompts=
dataset=

for flag in "$@"; do
	case $flag in
		--pidfile=*)
			pidfile=$(echo $flag | awk -F = '{print $2}')
			;;
		--model=*)
			model=$(echo $flag | awk -F = '{print $2}')
			;;
		--prompts=*)
			prompts=$(echo $flag | awk -F = '{print $2}')
			;;
		--dataset=*)
			dataset=$(echo $flag | awk -F = '{print $2}')
			;;
		*)
			echo "Unknown command-line flag" $flag
			exit
	esac
done

if [ -z $model ]; then
	echo "Missing operand: --model"
	exit
fi
if [ -z $prompts ]; then
	echo "Missing operand: --prompts"
	exit
fi
if [ -z $dataset ]; then
	echo "Missing operand: --dataset"
	exit
fi

echo "Running burstgpt"
source $project_dir/playground/infer/venv/vllm/bin/activate
set -x
python3 $project_dir/playground/infer/vllm/benchmarks/benchmark_serving.py --model $model --backend sglang --dataset-name burstgpt --dataset-path $dataset --num-prompts $prompts --request-rate 4 --burstiness 1 --tokenizer $model --trust-remote-code --save-result --save-detailed --result-dir $project_dir/playground/infer/vllm/benchmark_log --port 30000 &

# python3 $project_dir/playground/infer/vllm/benchmarks/benchmark_serving.py --model $model --backend sglang --dataset-name burstgpt --dataset-path $dataset --num-prompts $prompts --tokenizer $model --trust-remote-code --save-result --save-detailed --result-dir $project_dir/playground/infer/vllm/benchmark_log --port 30000 &

rootpid=$!
if [ -n $pidfile ]; then
	echo $rootpid | tee $pidfile
fi

wait $rootpid

set +x
