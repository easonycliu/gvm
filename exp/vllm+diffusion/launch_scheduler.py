#!/usr/bin/python3

import argparse
import os
import signal
import time
import ctypes

CGROUP_BASE_DIR = "/sys/kernel/debug/nvidia-uvm/processes"
CHECKING_INTERVAL_MS = 100

def parse_args():
	parser = argparse.ArgumentParser(
		description="Scheduler for GPU sharing between BE task and LC task"
	)
	parser.add_argument(
		"--lcpid",
		required=True,
		type=int,
		help="PID of latency critical task."
	)
	parser.add_argument(
		"--bepid",
		required=True,
		type=int,
		help="PID of best effort task."
	)
	parser.add_argument(
		"--lcmemlimit",
		required=True,
		type=int,
		help="Initial memlimit of latency critical task, -1 means unlimited."
	)
	parser.add_argument(
		"--bememlimit",
		required=True,
		type=int,
		help="Initial memlimit of best effort task, -1 means unlimited."
	)
	return parser.parse_args()

def set_mem_limit(pid, limit):
	print("Set {}'s memory limit to {}".format(pid, limit))
	with open(os.path.join(CGROUP_BASE_DIR, str(pid), "0", "memory.limit.high"), "w") as f:
		f.write("{}\n".format(limit))

def get_gcgroup_stat(pid):
	nr_submitted_kernels=0
	nr_ended_kernels=0
	nr_pending_kernels=0

	with open(os.path.join(CGROUP_BASE_DIR, str(pid), "0", "gcgroup.stat"), "r") as f:
		for line in f:
			if line.startswith("nr_submitted_kernels"):
				nr_submitted_kernels = int(line.split(" ")[1])
			if line.startswith("nr_ended_kernels"):
				nr_ended_kernels = int(line.split(" ")[1])
			if line.startswith("nr_pending_kernels"):
				nr_pending_kernels = int(line.split(" ")[1])

	return nr_submitted_kernels, nr_ended_kernels, nr_pending_kernels

if __name__ == "__main__":
	args = parse_args()
	if args.bememlimit == -1:
		args.bememlimit = ctypes.c_ulong(-1).value
	if args.lcmemlimit == -1:
		args.lcmemlimit = ctypes.c_ulong(-1).value

	# TODO: implement scheduling policy (e.g., monitor gcgroup.stat
	# and adjust memory limits / preempt based on pending kernels).
	while True:
		time.sleep(CHECKING_INTERVAL_MS / 1000)
