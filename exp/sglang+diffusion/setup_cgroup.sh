#!/bin/bash

priority=
memlimit=
rootpid=
for flag in "$@"; do
	case $flag in
		--priority=*)
			priority=$(echo $flag | awk -F = '{print $2}')
			;;
		--memlimit=*)
			memlimit=$(echo $flag | awk -F = '{print $2}')
			;;
		--rootpid=*)
			rootpid=$(echo $flag | awk -F = '{print $2}')
			;;
		*)
			echo "Unknown command-line flag" $flag
	esac
done

if [ -z $priority ]; then
	echo "Missing operand: --priority"
	exit
fi
if [ -z $memlimit ]; then
	echo "Missing operand: --memlimit"
	exit
fi
if [ -z $rootpid ]; then
	echo "Missing operand: --rootpid"
	exit
fi

while true; do
	child_pids=$(pstree -p $rootpid 2>/dev/null | grep -oP '\(\d+\)' | tr -d '()')
	if [ -z "$child_pids" ]; then
		echo "Break because no pid is found"
		break
	fi

	gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)

	gcgroup_set=false
	for pid in $child_pids; do
		if echo "$gpu_pids" | grep -qx "$pid"; then
			echo $memlimit | sudo tee /sys/kernel/debug/nvidia-uvm/processes/$pid/0/memory.limit
			echo $priority | sudo tee /sys/kernel/debug/nvidia-uvm/processes/$pid/0/compute.priority
			gcgroup_set=true
			echo "Setup cgroup for pid $pid"
			break
		fi
	done

	if [ "$gcgroup_set" == true ]; then
		break
	fi

	sleep 1
done
