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
server_control_pid=
server_control_pid_file=$(mktemp)
server_script_pid=
client_pid=
client_pid_file=$(mktemp)
client_script_pid=
preempt_pid=
preempt_pid_file=$(mktemp)
preempt_script_pid=
if [[ "$method" == "GVM" || "$method" == "GVMFT" || "$method" == "GVMAC" ]]; then
	$exp_dir/start_vllm_server.sh --pidfile=$server_pid_file --control-pidfile=$server_control_pid_file --method=$method --mode=$mode --model=$model --memlimit=$lcmemlimit --priority=$lcpriority &
	server_script_pid=$!
	sleep 60
	$be_launcher --pidfile=$preempt_pid_file --method=$method --memlimit=$bememlimit --priority=$bepriority --mode=$mode $be_output_arg &
	preempt_script_pid=$!
elif [ "$method" == "MIG" ]; then
	$exp_dir/start_vllm_server.sh --pidfile=$server_pid_file --control-pidfile=$server_control_pid_file --method=$method --mode=$mode --model=$model --device=$lcdevice &
	server_script_pid=$!
	sleep 60
	$be_launcher --pidfile=$preempt_pid_file --method=$method --device=$bedevice --mode=$mode $be_output_arg &
	preempt_script_pid=$!
else
	$exp_dir/start_vllm_server.sh --pidfile=$server_pid_file --control-pidfile=$server_control_pid_file --method=$method --mode=$mode --model=$model &
	server_script_pid=$!
	sleep 60
	$be_launcher --pidfile=$preempt_pid_file --method=$method --mode=$mode $be_output_arg &
	preempt_script_pid=$!
fi

echo "Waiting for system startup"
sleep 90

while [ ! -s "$server_pid_file" ]; do sleep 0.5; done
server_pid=$(cat $server_pid_file)
rm -f $server_pid_file
while [ ! -s "$server_control_pid_file" ]; do sleep 0.5; done
server_control_pid=$(cat $server_control_pid_file)
rm -f $server_control_pid_file
while [ ! -s "$preempt_pid_file" ]; do sleep 0.5; done
preempt_pid=$(cat $preempt_pid_file)
rm -f $preempt_pid_file

if [[ "$method" == "GVMFT" || "$method" == "GVMAC" ]]; then
	mkdir -p $exp_dir/scheduler_logs
	sudo $exp_dir/launch_scheduler.py --lcpid $server_pid --bepid $preempt_pid --lcmemlimit $lcmemlimit --bememlimit $bememlimit --trace-file=$exp_dir/scheduler_logs/scheduler-$method_label-$result_timestamp.csv &
	scheduler_pid=$!
fi

$exp_dir/start_vllm_client.sh --pidfile=$client_pid_file --mode=$mode --model=$model --prompts=16384 --dataset=$dataset --result-dir=$output_dir --result-filename=$vllm_result_filename &
client_script_pid=$!
while [ ! -s "$client_pid_file" ]; do sleep 0.5; done
client_pid=$(cat $client_pid_file)
rm -f $client_pid_file

wait_for_child() {
	local child_pid=$1
	local timeout_seconds=$2
	local elapsed=0
	while kill -0 "$child_pid" 2>/dev/null && [ "$elapsed" -lt "$timeout_seconds" ]; do
		sleep 1
		elapsed=$((elapsed + 1))
	done
	if ! kill -0 "$child_pid" 2>/dev/null; then
		wait "$child_pid" 2>/dev/null || true
		return 0
	fi
	return 1
}

stop_launcher() {
	local launcher_pid=$1
	local workload_pid=$2
	local graceful_signal=$3
	local timeout_seconds=$4
	local name=$5

	[ -z "$launcher_pid" ] && return
	# The launcher is an asynchronous Bash job, for which Bash ignores SIGINT.
	# Deliver the application-level interrupt directly to the workload, but wait
	# on the launcher because it is the child process owned by this shell.
	kill -CONT "$workload_pid" 2>/dev/null || true
	kill -"$graceful_signal" "$workload_pid" 2>/dev/null || true
	if wait_for_child "$launcher_pid" "$timeout_seconds"; then
		return
	fi
	echo "$name did not exit after ${timeout_seconds}s; killing it"
	kill -KILL "$workload_pid" 2>/dev/null || true
	kill -KILL "$launcher_pid" 2>/dev/null || true
	wait "$launcher_pid" 2>/dev/null || true
}

shutdown_started=0
shutdown_experiment() {
	if [ "$shutdown_started" -eq 1 ]; then
		return
	fi
	shutdown_started=1
	trap - SIGINT SIGTERM

	# Stop scheduling before resuming BE; otherwise the scheduler can stop it
	# again between SIGCONT and its application-level shutdown handler.
	if [ -n "$scheduler_pid" ]; then
		kill -TERM "$scheduler_pid" 2>/dev/null || true
		if ! wait_for_child "$scheduler_pid" 5; then
			kill -KILL "$scheduler_pid" 2>/dev/null || true
			wait "$scheduler_pid" 2>/dev/null || true
		fi
	fi

	# The standalone benchmark catches INT, cancels outstanding requests, and
	# serializes all completed-request metrics before its launcher exits.
	stop_launcher "$client_script_pid" "$client_pid" INT 30 "vLLM benchmark"

	if [ -n "$llamafactory_result" ]; then
		# LLaMA Factory cannot leave a reliable signal handler in a blocked CUDA
		# call. Its JSONL log is durable, so terminate it and extract afterward.
		kill -CONT "$preempt_pid" 2>/dev/null || true
		kill -KILL "$preempt_pid" 2>/dev/null || true
		wait_for_child "$preempt_script_pid" 10 || {
			kill -KILL "$preempt_script_pid" 2>/dev/null || true
			wait "$preempt_script_pid" 2>/dev/null || true
		}
	else
		# Diffusion writes its collected latencies from its INT handler. Allow
		# the current GPU batch to return before escalating.
		stop_launcher "$preempt_script_pid" "$preempt_pid" INT 120 "diffusion"
	fi

	stop_launcher "$server_script_pid" "$server_control_pid" INT 30 "vLLM server"
}
trap shutdown_experiment SIGINT SIGTERM

sleep $duration

shutdown_experiment

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
