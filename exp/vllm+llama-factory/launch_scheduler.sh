#!/bin/bash

function preempt() {
	if [ "${preempted}" == false ]; then
		pid=$1
		kill -SIGSTOP ${pid}
		# echo 1 | sudo tee /sys/kernel/debug/nvidia-uvm/processes/${pid}/0/compute.freeze

		echo Preempt $pid
	fi
}

function reschedule() {
	if [ "${preempted}" == true ]; then
		pid=$1
		# echo 0 | sudo tee /sys/kernel/debug/nvidia-uvm/processes/${pid}/0/compute.freeze
		kill -SIGCONT ${pid}

		echo Reschedule $pid
	fi
}

listening_port=
preempt_pid=
for flag in "$@"; do
	case $flag in
		--listening_port=*)
			listening_port=$(echo $flag | awk -F = '{print $2}')
			;;
		--preempt_pid=*)
			preempt_pid=$(echo $flag | awk -F = '{print $2}')
			;;
		*)
			echo "Unknown command-line flag" $flag
	esac
done

if [ "${listening_port}" == "" -o "${preempt_pid}" == "" ]; then
	echo "Usage: ./launch_scheduler.sh --listening_port=<port> --preempt_pid=<pid>"
	exit
fi

logfile=$(mktemp)
sudo tcpdump -l -i lo port $listening_port 2>&1 | grep --line-buffered "localhost.$listening_port:.*sackOK,TS" >$logfile &
tcpdump_pid=$!
trap 'echo "Cleaning up"; kill "$tcpdump_pid"; rm "$logfile"; exit' SIGINT
request_num=0
preempted=false
while sleep 1; do
	new_request_num=$(wc -l < "$logfile")
	increased_request_num=$((new_request_num - request_num))
	echo "Incoming requests: $increased_request_num"
	if [ "${increased_request_num}" -gt 15 ]; then
		preempt $preempt_pid $preempted
		preempted=true
	elif [ "${increased_request_num}" -lt 10 ]; then
		reschedule $preempt_pid $preempted
		preempted=false
	fi
	request_num=$new_request_num
done
