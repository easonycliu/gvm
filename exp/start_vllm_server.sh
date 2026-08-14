#!/bin/bash

script_dir=$(dirname ${BASH_SOURCE[0]})
project_dir=$(realpath $script_dir/..)

pidfile=
control_pidfile=
method=
priority=
memlimit=
mode=
model=
device=
param="--gpu-memory-utilization 0.8 --disable-log-requests --enforce-eager"

for flag in "$@"; do
	case $flag in
		--pidfile=*)
			pidfile=$(echo $flag | awk -F = '{print $2}')
			;;
		--control-pidfile=*)
			control_pidfile=$(echo $flag | awk -F = '{print $2}')
			;;
		--method=*)
			method=$(echo $flag | awk -F = '{print $2}')
			;;
		--mode=*)
			mode=$(echo $flag | awk -F = '{print $2}')
			;;
		--model=*)
			model=$(echo $flag | awk -F = '{print $2}')
			;;
		--priority=*)
			priority=$(echo $flag | awk -F = '{print $2}')
			;;
		--memlimit=*)
			memlimit=$(echo $flag | awk -F = '{print $2}')
			;;
		--device=*)
			device=$(echo $flag | awk -F = '{print $2}')
			;;
		*)
			echo "Unknown command-line flag" $flag
			exit
	esac
done

if [ -z $mode ]; then
	mode="text"
fi

if [ -z $method ]; then
	echo "Missing operand: --method"
	exit
fi
if [ -z $model ]; then
	echo "Missing operand: --model"
	exit
fi
if [[ "$method" == "GVM" || "$method" == "GVMFT" || "$method" == "GVMAC" ]] && [ -z $priority ]; then
	echo "Missing operand: --priority"
	exit
fi
if [[ "$method" == "GVM" || "$method" == "GVMFT" || "$method" == "GVMAC" ]] && [ -z $memlimit ]; then
	echo "Missing operand: --memlimit"
	exit
fi
if [ "$method" == "MIG" ] && [ -z $device ]; then
	echo "Missing operand: --device"
	exit
fi

if [ "$mode" == "video" ]; then
	param=$(echo "--max-model-len 32768" "$param")
fi

if [ "$method" == "GVMAC" ]; then
	param="--gpu-memory-utilization 0.65 --disable-log-requests --enforce-eager"
	if [ "$mode" == "video" ]; then
		param="--max-model-len 32768 $param"
	fi
fi

if [[ "$method" == "GVM" || "$method" == "GVMFT" ]]; then
	source $project_dir/venv/VllmVenv/bin/activate
	LD_LIBRARY_PATH=$project_dir/gvm-cuda-driver/install:$LD_LIBRARY_PATH vllm serve $model $param &
elif [ "$method" == "GVMAC" ]; then
	source $project_dir/venv/VllmACVenv/bin/activate
	LD_LIBRARY_PATH=$project_dir/gvm-cuda-driver/install:$LD_LIBRARY_PATH vllm serve $model $param &
elif [ "$method" == "exclusive-lc" ]; then
	source $project_dir/venv/VllmVenv/bin/activate
	LD_LIBRARY_PATH=$project_dir/gvm-cuda-driver/install:$LD_LIBRARY_PATH vllm serve $model $param &
elif [ "$method" == "GPreempt" ]; then
	source $project_dir/venv/VllmVenv/bin/activate
	LD_LIBRARY_PATH=$project_dir/gvm-cuda-driver/install:$LD_LIBRARY_PATH vllm serve $model $param &
elif [ "$method" == "TGS" ]; then
	docker run --rm --name job_2 --gpus "device=0" --ipc host --network host --cap-add sys_nice -u root --cpuset-cpus 0-5 -v $project_dir/3rdparty/TGS:/cluster -v $project_dir/apps/vllm:/vllm -v $project_dir/../BurstGPTDataset/burstgpt:/burstgpt -v $PWD:/exp -v ~/.cache/huggingface:/root/.cache/huggingface -v $project_dir/3rdparty/TGS/hijack/high-priority-lib/libcontroller.so:/libcontroller.so:ro -v $project_dir/3rdparty/TGS/hijack/high-priority-lib/libcuda.so:/libcuda.so:ro -v $project_dir/3rdparty/TGS/hijack/high-priority-lib/libcuda.so.1:/libcuda.so.1:ro -v $project_dir/3rdparty/TGS/hijack/high-priority-lib/libnvidia-ml.so:/libnvidia-ml.so:ro -v $project_dir/3rdparty/TGS/hijack/high-priority-lib/libnvidia-ml.so.1:/libnvidia-ml.so.1:ro -v $project_dir/3rdparty/TGS/hijack/high-priority-lib/ld.so.preload:/etc/ld.so.preload:ro -v $project_dir/3rdparty/TGS/gsharing:/etc/gsharing -e TGS_WORKER_IP=10.128.0.122 -e TGS_WORKER_PORT=6889 -e TGS_TRAINER_PORT=59967 -e TGS_JOB_ID=2 -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -e GPU_CONFIG_FILE=/gpu_config.json -e GPU_STATUS_FILE=/gpu_status.json easonliu12138/gvm_cuda_12_9 vllm serve $model $param &
elif [ "$method" == "xsched" ]; then
	if [ -z "$(ps -a | grep -e "xserver$")" ]; then
		echo "Please launch xsched server before start application with xsched"
		exit;
	fi

	export XSCHED_POLICY=GBL
	export XSCHED_ENABLE_MANAGED=ON
	export XSCHED_AUTO_XQUEUE=ON
	export XSCHED_AUTO_XQUEUE_PRIORITY=1
	export XSCHED_AUTO_XQUEUE_LEVEL=1
	export XSCHED_AUTO_XQUEUE_THRESHOLD=16
	export XSCHED_AUTO_XQUEUE_BATCH_SIZE=8
	export LD_LIBRARY_PATH=$project_dir/3rdparty/xsched/output/lib:$LD_LIBRARY_PATH

	# XSched requires the vLLM integration hooks in the adapted source tree;
	# interception alone is not fully transparent for vLLM.
	source $project_dir/venv/VllmACVenv/bin/activate
	vllm serve $model $param &
elif [ "$method" == "UVM" ]; then
	source $project_dir/venv/VllmVenv/bin/activate
	LD_LIBRARY_PATH=$project_dir/gvm-cuda-driver/install:$LD_LIBRARY_PATH vllm serve $model $param &
elif [ "$method" == "MIG" ]; then
	source $project_dir/venv/VllmVenv/bin/activate
	export CUDA_VISIBLE_DEVICES=$device
	LD_LIBRARY_PATH=$project_dir/gvm-cuda-driver/install:$LD_LIBRARY_PATH vllm serve $model $param &
else
	echo "Unknown method: $method"
	exit
fi

rootpid=$!
if [ -n "$control_pidfile" ]; then
	echo "$rootpid" | tee "$control_pidfile"
fi
if [ -n $pidfile ]; then
	while true; do
		if nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | grep "VLLM"; then
			vllm_active_process=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | grep "VLLM" | awk -F "," '{print $1}')
			echo "VLLM process detected at $vllm_active_process!"
			echo $vllm_active_process | tee $pidfile
			break
		fi
		sleep 1
	done
fi

if [[ "$method" == "GVM" || "$method" == "GVMFT" || "$method" == "GVMAC" ]]; then
	$script_dir/setup_cgroup.sh --priority=$priority --memlimit=$memlimit --rootpid=$rootpid
elif [ "$method" == "GPreempt" ]; then
	$script_dir/setup_cgroup.sh --priority=0 --memlimit=400000000000 --rootpid=$rootpid
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
exit $status
