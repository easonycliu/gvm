#!/bin/bash

script_dir=$(dirname ${BASH_SOURCE[0]})
project_dir=$(realpath $script_dir/../..)

pidfile=
method=
priority=
memlimit=
device=
for flag in "$@"; do
	case $flag in
		--pidfile=*)
			pidfile=$(echo $flag | awk -F = '{print $2}')
			;;
		--method=*)
			method=$(echo $flag | awk -F = '{print $2}')
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
	esac
done

if [ -z $method ]; then
	echo "Missing operand: --method"
	exit
fi
if [ "$method" == "GVM" ] && [ -z $priority ]; then
	echo "Missing operand: --priority"
	exit
fi
if [ "$method" == "GVM" ] && [ -z $memlimit ]; then
	echo "Missing operand: --memlimit"
	exit
fi
if [ "$method" == "MIG" ] && [ -z $device ]; then
	echo "Missing operand: --device"
	exit
fi

if [ "$method" == "GVM" ]; then
	source $project_dir/playground/infer/venv/diffusion/bin/activate
	LD_PRELOAD="$project_dir/cuda_custom/libcustom_cuda.so" python3 diffusion.py --dataset_path vidprom.txt --log_file stats-$(date +%Y%m%d-%H%M%S).txt &
elif [ "$method" == "GPreempt" ]; then
	source $project_dir/playground/infer/venv/diffusion/bin/activate
	LD_PRELOAD="$project_dir/cuda_custom/libcustom_cuda.so" python3 diffusion.py --dataset_path vidprom.txt --log_file stats-$(date +%Y%m%d-%H%M%S).txt &
elif [ "$method" == "TGS" ]; then
	docker run --rm --name job_1 --gpus "device=0" --ipc host --network host --cap-add sys_nice -u root --cpuset-cpus 0-5 -v $project_dir/playground/infer/diffusion:/diffusion -v $project_dir/playground/train/LLaMA-Factory:/LLaMA-Factory -v $script_dir:/exp -v ~/.cache/huggingface:/root/.cache/huggingface -v $project_dir/3rdparty/TGS:/cluster -v $project_dir/3rdparty/TGS/hijack/low-priority-lib/libcontroller.so:/libcontroller.so:ro -v $project_dir/3rdparty/TGS/hijack/low-priority-lib/libcuda.so:/libcuda.so:ro -v $project_dir/3rdparty/TGS/hijack/low-priority-lib/libcuda.so.1:/libcuda.so.1:ro -v $project_dir/3rdparty/TGS/hijack/low-priority-lib/libnvidia-ml.so:/libnvidia-ml.so:ro -v $project_dir/3rdparty/TGS/hijack/low-priority-lib/libnvidia-ml.so.1:/libnvidia-ml.so.1:ro -v $project_dir/3rdparty/TGS/hijack/high-priority-lib/ld.so.preload:/etc/ld.so.preload:ro -v $project_dir/3rdparty/TGS/gsharing:/etc/gsharing -e TGS_WORKER_IP=10.128.0.122 -e TGS_WORKER_PORT=6889 -e TGS_TRAINER_PORT=47123 -e TGS_JOB_ID=1 -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -e GPU_CONFIG_FILE=/gpu_config.json -e GPU_STATUS_FILE=/gpu_status.json easonliu12138/gvm_cuda_12_9 python3 /exp/diffusion.py --dataset_path /exp/vidprom.txt --log_file /exp/diffusion_outputs/stats-$(date +%Y%m%d-%H%M%S).txt &
elif [ "$method" == "xsched" ]; then
	if [ -z "$(ps -a | grep -e "xserver$")" ]; then
		echo "Please launch xsched server before start application with xsched"
		exit;
	fi

	export XSCHED_POLICY=GBL
	export XSCHED_ENABLE_MANAGED=ON
	export XSCHED_AUTO_XQUEUE=ON
	export XSCHED_AUTO_XQUEUE_PRIORITY=0
	export XSCHED_AUTO_XQUEUE_LEVEL=1
	export XSCHED_AUTO_XQUEUE_THRESHOLD=16
	export XSCHED_AUTO_XQUEUE_BATCH_SIZE=8
	export LD_LIBRARY_PATH=$project_dir/3rdparty/xsched/output/lib:$LD_LIBRARY_PATH

	source $project_dir/playground/infer/venv/diffusion/bin/activate
	python3 diffusion.py --dataset_path vidprom.txt --log_file stats-$(date +%Y%m%d-%H%M%S).txt
elif [ "$method" == "UVM" ]; then
	source $project_dir/playground/infer/venv/diffusion/bin/activate
	LD_PRELOAD="$project_dir/cuda_custom/libcustom_cuda.so" python3 diffusion.py --dataset_path vidprom.txt --log_file stats-$(date +%Y%m%d-%H%M%S).txt &
elif [ "$method" == "MIG" ]; then
	source $project_dir/playground/infer/venv/diffusion/bin/activate
	export CUDA_VISIBLE_DEVICES=$device
	LD_PRELOAD="$project_dir/cuda_custom/libcustom_cuda.so" python3 diffusion.py --dataset_path vidprom.txt --log_file stats-$(date +%Y%m%d-%H%M%S).txt &
else
	echo "Unknown method: $method"
	exit
fi

rootpid=$!
if [ -n $pidfile ]; then
	echo $rootpid | tee $pidfile
fi

if [ "$method" == "GVM" ]; then
	./setup_cgroup.sh --priority=$priority --memlimit=$memlimit --rootpid=$rootpid
elif [ "$method" == "GPreempt" ]; then
	./setup_cgroup.sh --priority=15 --memlimit=400000000000 --rootpid=$rootpid
fi

trap 'wait $rootpid; exit' INT
wait $rootpid
