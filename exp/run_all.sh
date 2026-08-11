#!/bin/bash

exp_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
project_dir=$(realpath "$exp_dir/..")
experiment_dir=$(realpath "$PWD")
experiment_name=$(basename "$experiment_dir")
be_launcher=$experiment_dir/start_be.sh

if [ ! -f "$be_launcher" ]; then
	echo "Run this script from an experiment directory containing start_be.sh"
	exit 1
fi

method=
duration=
dataset=
lcpriority=2
bepriority=10
lcmemlimit=40000000000
bememlimit=6000000000
lcdevice=
bedevice=
mode=
eval_type=
result_label=
for flag in "$@"; do
	case $flag in
		--method=*)
			method=$(echo $flag | awk -F = '{print $2}')
			;;
		--duration=*)
			duration=$(echo $flag | awk -F = '{print $2}')
			;;
		--dataset=*)
			dataset=$(echo $flag | awk -F = '{print $2}')
			;;
		--lcpriority=*)
			lcpriority=$(echo $flag | awk -F = '{print $2}')
			;;
		--bepriority=*)
			bepriority=$(echo $flag | awk -F = '{print $2}')
			;;
		--lcmemlimit=*)
			lcmemlimit=$(echo $flag | awk -F = '{print $2}')
			;;
		--bememlimit=*)
			bememlimit=$(echo $flag | awk -F = '{print $2}')
			;;
		--lcdevice=*)
			lcdevice=$(echo $flag | awk -F = '{print $2}')
			;;
		--bedevice=*)
			bedevice=$(echo $flag | awk -F = '{print $2}')
			;;
		--mode=*)
			mode=$(echo $flag | awk -F = '{print $2}')
			;;
		--eval=*)
			eval_type=$(echo $flag | awk -F = '{print $2}')
			;;
		--label=*)
			result_label=$(echo $flag | awk -F = '{print $2}')
			;;
		*)
			echo "Unknown command-line flag" $flag
			exit
	esac
done

if [ -z $method ]; then
	echo "Missing operand: --method"
	exit
fi
if [ -z $duration ]; then
	echo "Missing operand: --duration"
	exit
fi
if [ -z $dataset ]; then
	echo "Missing operand: --dataset"
	exit
fi
case $eval_type in
	nomem|overall|pareto|breakdown)
		;;
	"")
		echo "Missing operand: --eval=<nomem|overall|pareto|breakdown>"
		exit
		;;
	*)
		echo "Unsupported evaluation type: $eval_type"
		exit
		;;
esac
if [ "$eval_type" == "breakdown" ] && [ -z "$result_label" ]; then
	echo "Missing operand: --label for breakdown evaluation"
	exit
fi
if [ -n "$result_label" ] && [[ ! "$result_label" =~ ^[A-Za-z0-9._-]+$ ]]; then
	echo "Invalid --label: use letters, numbers, dot, underscore, or hyphen"
	exit
fi
if [ "$method" == "MIG" ] && [ -z $lcdevice ]; then
	echo "Missing operand: --lcdevice"
	exit
fi
if [ "$method" == "MIG" ] && [ -z $bedevice ]; then
	echo "Missing operand: --bedevice"
	exit
fi

if [ -z $mode ]; then
	mode="text"
fi

if [ "$mode" == "video" ]; then
	result_experiment_name=$experiment_name.vl
else
	result_experiment_name=$experiment_name
fi
output_dir=$project_dir/data/$eval_type/$result_experiment_name
mkdir -p "$output_dir"

case $method in
	GVM)
		method_label=GVM-$lcpriority-$bepriority
		;;
	GVMFT)
		method_label=GVMDYN-$lcpriority-$bepriority
		;;
	GVMAC)
		method_label=GVMCOOP
		;;
	*)
		method_label=$method
		;;
esac
if [ -n "$result_label" ]; then
	method_label=$method_label-$result_label
fi
result_timestamp=$(date +%Y%m%d-%H%M%S)
vllm_result_filename=vllm-$method_label-$result_timestamp.json
be_output_arg=
llamafactory_log=
llamafactory_result=
if [ "$experiment_name" == "vllm+diffusion" ]; then
	be_output_arg=--log-file=$output_dir/diffusion-$method_label-$result_timestamp.txt
elif [ "$experiment_name" == "vllm+llama-factory" ]; then
	if [ "$mode" == "video" ]; then
		llamafactory_log=$experiment_dir/saves/Qwen2.5-VL-3B-Instruct/trainer_log.jsonl
	else
		llamafactory_log=$experiment_dir/saves/llama3.2-3b/trainer_log.jsonl
	fi
	llamafactory_result=$output_dir/llama_factory-$method_label-$result_timestamp.txt
else
	echo "Unsupported experiment directory $experiment_name"
	exit 1
fi

scheduler_pid=
if [ "$method" == "xsched" ]; then
	$exp_dir/launch_xserver.sh &
	scheduler_pid=$!
fi

model=
if [ "$mode" == "text" ]; then
	model="meta-llama/Llama-3.2-3B"
elif [ "$mode" == "video" ]; then
	model="Qwen/Qwen2.5-VL-3B-Instruct"
else
	echo "Unsupported mode $mode"
	exit
fi
server_pid=
server_pid_file=$(mktemp)
server_script_pid=
client_pid=
client_pid_file=$(mktemp)
client_script_pid=
preempt_pid=
preempt_pid_file=$(mktemp)
preempt_script_pid=
if [[ "$method" == "GVM" || "$method" == "GVMFT" || "$method" == "GVMAC" ]]; then
	$exp_dir/start_vllm_server.sh --pidfile=$server_pid_file --method=$method --mode=$mode --model=$model --memlimit=$lcmemlimit --priority=$lcpriority &
	server_script_pid=$!
	sleep 60
	$be_launcher --pidfile=$preempt_pid_file --method=$method --memlimit=$bememlimit --priority=$bepriority --mode=$mode $be_output_arg &
	preempt_script_pid=$!
elif [ "$method" == "MIG" ]; then
	$exp_dir/start_vllm_server.sh --pidfile=$server_pid_file --method=$method --mode=$mode --model=$model --device=$lcdevice &
	server_script_pid=$!
	sleep 60
	$be_launcher --pidfile=$preempt_pid_file --method=$method --device=$bedevice --mode=$mode $be_output_arg &
	preempt_script_pid=$!
else
	$exp_dir/start_vllm_server.sh --pidfile=$server_pid_file --method=$method --mode=$mode --model=$model &
	server_script_pid=$!
	sleep 60
	$be_launcher --pidfile=$preempt_pid_file --method=$method --mode=$mode $be_output_arg &
	preempt_script_pid=$!
fi

echo "Waiting for system startup"
sleep 90

$exp_dir/start_vllm_client.sh --pidfile=$client_pid_file --mode=$mode --model=$model --prompts=16384 --dataset=$dataset --result-dir=$output_dir --result-filename=$vllm_result_filename &
client_script_pid=$!

while [ ! -s "$server_pid_file" ]; do sleep 0.5; done
server_pid=$(cat $server_pid_file)
rm -f $server_pid_file
while [ ! -s "$client_pid_file" ]; do sleep 0.5; done
client_pid=$(cat $client_pid_file)
rm -f $client_pid_file
while [ ! -s "$preempt_pid_file" ]; do sleep 0.5; done
preempt_pid=$(cat $preempt_pid_file)
rm -f $preempt_pid_file

if [[ "$method" == "GVMFT" || "$method" == "GVMAC" ]]; then
	sudo $exp_dir/launch_scheduler.py --lcpid $server_pid --bepid $preempt_pid --lcmemlimit $lcmemlimit --bememlimit $bememlimit &
	scheduler_pid=$!
fi

interrupt_experiment() {
	kill -2 $client_pid $server_script_pid 2>/dev/null
	if [ -n "$llamafactory_result" ]; then
		kill -9 $preempt_pid 2>/dev/null
	else
		kill -2 $preempt_pid 2>/dev/null
	fi
	if [ -n "$scheduler_pid" ]; then
		kill -9 $scheduler_pid 2>/dev/null
	fi
}
trap interrupt_experiment SIGINT

sleep $duration

if [ -n "$llamafactory_result" ]; then
	if [ ! -f "$llamafactory_log" ] && [ -f "${llamafactory_log%l}" ]; then
		llamafactory_log=${llamafactory_log%l}
	fi
	$project_dir/venv/BenchmarkVenv/bin/python3 \
		$project_dir/apps/benchmark/llamafactory_step_time.py \
		--log "$llamafactory_log" \
		--duration "$duration" \
		--output "$llamafactory_result"
fi

kill -2 $client_pid
wait $client_pid

if [ -n "$llamafactory_result" ]; then
	kill -9 $preempt_pid
else
	kill -2 $preempt_pid
fi
wait $preempt_pid

kill -2 -$$
