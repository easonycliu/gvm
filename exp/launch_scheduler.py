#!/usr/bin/python3

import argparse
import os
import signal
import time
import ctypes

from enum import Enum

CGROUP_BASE_DIR = "/sys/kernel/debug/nvidia-uvm/processes"
CHECKING_INTERVAL_MS = 100
OPERATE_INTERVAL_MIN_MS = 3000
SLIDE_WINDOW_SIZE = 30
PENDING_KERNEL_UPPER_THRESHOLD = 128
PENDING_KERNEL_LOWER_THRESHOLD = 4

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

def preempt(pid):
	print("Preempt {}".format(pid))
	os.kill(pid, signal.SIGSTOP)

def reschedule(pid):
	print("Reschedule {}".format(pid))
	os.kill(pid, signal.SIGCONT)

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

class BE_STATUS(Enum):
	UNLIMITED=0
	LIMITED=1
	PREEMPTED=2

if __name__ == "__main__":
	args = parse_args()
	if args.bememlimit == -1:
		args.bememlimit = ctypes.c_ulong(-1).value
	if args.lcmemlimit == -1:
		args.lcmemlimit = ctypes.c_ulong(-1).value

	nr_pending_kernels_list = [0 for _ in range(SLIDE_WINDOW_SIZE)]
	nr_submitted_kernels_list = [0 for _ in range(SLIDE_WINDOW_SIZE)]
	operate_time = 0.0
	be_status = BE_STATUS.UNLIMITED
	while True:
		time.sleep(CHECKING_INTERVAL_MS / 1000)
		nr_submitted_kernels, nr_ended_kernels, nr_pending_kernels = get_gcgroup_stat(args.lcpid)
		nr_pending_kernels_list.append(nr_pending_kernels)
		nr_submitted_kernels_list.append(nr_submitted_kernels)
		if (time.time() > operate_time + OPERATE_INTERVAL_MIN_MS / 1000):
			if len(set(nr_submitted_kernels_list[-4:])) == 1:
				print(nr_submitted_kernels_list[-4:])
				if be_status == BE_STATUS.UNLIMITED:
					pass
				elif be_status == BE_STATUS.LIMITED:
					set_mem_limit(args.bepid, ctypes.c_ulong(-1).value)
					be_status = BE_STATUS.UNLIMITED
					operate_time = time.time()
				elif be_status == BE_STATUS.PREEMPTED:
					set_mem_limit(args.bepid, ctypes.c_ulong(-1).value)
					reschedule(args.bepid)
					be_status = BE_STATUS.UNLIMITED
					operate_time = time.time()
				else:
					raise AssertionError("Invalid status")
			else:
				slide_window_avg_pending_kernels = sum(nr_pending_kernels_list[-SLIDE_WINDOW_SIZE:]) / SLIDE_WINDOW_SIZE
				if slide_window_avg_pending_kernels > PENDING_KERNEL_UPPER_THRESHOLD:
					print(nr_pending_kernels_list[-SLIDE_WINDOW_SIZE:])
					if be_status == BE_STATUS.UNLIMITED:
						preempt(args.bepid)
						set_mem_limit(args.bepid, args.bememlimit)
						be_status = BE_STATUS.PREEMPTED
						operate_time = time.time()
					elif be_status == BE_STATUS.LIMITED:
						preempt(args.bepid)
						be_status = BE_STATUS.PREEMPTED
						operate_time = time.time()
					elif be_status == BE_STATUS.PREEMPTED:
						pass
					else:
						raise AssertionError("Invalid status")
				elif PENDING_KERNEL_LOWER_THRESHOLD <= slide_window_avg_pending_kernels <= PENDING_KERNEL_UPPER_THRESHOLD:
					if be_status == BE_STATUS.UNLIMITED:
						set_mem_limit(args.bepid, args.bememlimit)
						be_status = BE_STATUS.LIMITED
						operate_time = time.time()
					elif be_status == BE_STATUS.LIMITED:
						pass
					elif be_status == BE_STATUS.PREEMPTED:
						pass
					else:
						raise AssertionError("Invalid status")
				elif slide_window_avg_pending_kernels < PENDING_KERNEL_LOWER_THRESHOLD:
					if be_status == BE_STATUS.UNLIMITED:
						pass
					elif be_status == BE_STATUS.LIMITED:
						pass
					elif be_status == BE_STATUS.PREEMPTED:
						reschedule(args.bepid)
						be_status = BE_STATUS.LIMITED
						operate_time = time.time()
					else:
						raise AssertionError("Invalid status")
