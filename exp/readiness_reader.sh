#!/bin/bash

signal_type=$1
ready_file=$2
signaled=false

while IFS= read -r line || [ -n "$line" ]; do
	printf '%s\n' "$line"
	if [ "$signaled" == true ]; then
		continue
	fi
	case $signal_type in
		vllm)
			[[ "$line" == *"Application startup complete."* ]] && matched=true
			;;
		diffusion)
			[[ "$line" == *"Completed batch ['req1'] in"* ]] && matched=true
			;;
		llamafactory)
			if [[ "$line" == *"'loss':"* && "$line" == *"'grad_norm':"* &&
			      "$line" == *"'learning_rate':"* && "$line" == *"'epoch':"* ]]; then
				matched=true
			fi
			;;
		*)
			echo "Unknown readiness signal: $signal_type" >&2
			exit 2
			;;
	esac
	if [ "$matched" == true ]; then
		: > "$ready_file"
		signaled=true
	fi
done
