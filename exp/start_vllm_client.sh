#!/bin/bash

script_dir=$(dirname ${BASH_SOURCE[0]})
project_dir=$(realpath $script_dir/..)

pidfile=
mode=
model=
prompts=
dataset=
result_dir=
result_filename=

for flag in "$@"; do
	case $flag in
		--pidfile=*)
			pidfile=$(echo $flag | awk -F = '{print $2}')
			;;
		--mode=*)
			mode=$(echo $flag | awk -F = '{print $2}')
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
		--result-dir=*)
			result_dir=$(echo $flag | awk -F = '{print $2}')
			;;
		--result-filename=*)
			result_filename=$(echo $flag | awk -F = '{print $2}')
			;;
		*)
			echo "Unknown command-line flag" $flag
			exit
	esac
done

if [ -z $mode ]; then
	mode="text"
fi

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
if [ -z "$result_dir" ] || [ -z "$result_filename" ]; then
	echo "Missing result destination"
	exit
fi

echo "Running burstgpt"
set -x
source $project_dir/venv/BenchmarkVenv/bin/activate
benchmark_args=(
	--model "$model"
	--dataset-path "$dataset"
	--num-prompts "$prompts"
	--tokenizer "$model"
	--trust-remote-code
	--result-dir "$result_dir"
	--result-filename "$result_filename"
)
if [ "$mode" == "text" ]; then
	python3 $project_dir/apps/benchmark/benchmark_serving.py \
		"${benchmark_args[@]}" &
elif [ "$mode" == "video" ]; then
	python3 $project_dir/apps/benchmark/benchmark_serving.py \
		"${benchmark_args[@]}" \
		--mode video \
		--video-dir "$project_dir/exp/data/mmvu_cache" \
		--expected-output-len 64 &
else
	echo "Unsupported mode $mode"
	exit
fi

rootpid=$!
if [ -n $pidfile ]; then
	echo $rootpid | tee $pidfile
fi

forward_signal() {
	kill -"$1" "$rootpid" 2>/dev/null || true
}
trap 'forward_signal INT' INT
trap 'forward_signal TERM' TERM

wait "$rootpid"
status=$?
while kill -0 "$rootpid" 2>/dev/null; do
	wait "$rootpid"
	status=$?
done

set +x
exit $status
