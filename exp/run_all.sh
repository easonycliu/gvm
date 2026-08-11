#!/bin/bash

exp_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
experiment_dir=$(realpath "$PWD")
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
	$be_launcher --pidfile=$preempt_pid_file --method=$method --memlimit=$bememlimit --priority=$bepriority --mode=$mode &
	preempt_script_pid=$!
elif [ "$method" == "MIG" ]; then
	$exp_dir/start_vllm_server.sh --pidfile=$server_pid_file --method=$method --mode=$mode --model=$model --device=$lcdevice &
	server_script_pid=$!
	sleep 60
	$be_launcher --pidfile=$preempt_pid_file --method=$method --device=$bedevice --mode=$mode &
	preempt_script_pid=$!
else
	$exp_dir/start_vllm_server.sh --pidfile=$server_pid_file --method=$method --mode=$mode --model=$model &
	server_script_pid=$!
	sleep 60
	$be_launcher --pidfile=$preempt_pid_file --method=$method --mode=$mode &
	preempt_script_pid=$!
fi

echo "Waiting for system startup"
sleep 90

$exp_dir/start_vllm_client.sh --pidfile=$client_pid_file --mode=$mode --model=$model --prompts=16384 --dataset=$dataset &
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

trap 'kill -2 $client_pid $preempt_pid $server_script_pid; kill -9 $scheduler_pid' SIGINT

sleep $duration

kill -2 $client_pid
wait $client_pid

kill -2 $preempt_pid
wait $preempt_pid

kill -2 -$$
