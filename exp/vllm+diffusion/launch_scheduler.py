#!/usr/bin/python3

import argparse
import collections
import ctypes
import os
import signal
import subprocess
import sys
import time

from datetime import datetime
from enum import Enum

CGROUP_BASE_DIR = "/sys/kernel/debug/nvidia-uvm/processes"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts():
	return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def log(msg):
	print("[{}] {}".format(ts(), msg), flush=True)

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
	parser.add_argument(
		"--policy",
		default="burst_freeze",
		choices=["burst_freeze", "dynamic_priority"],
		help="Scheduling policy (default: burst_freeze)."
	)
	parser.add_argument(
		"--gpu",
		default=0,
		type=int,
		help="GPU index (default: 0)."
	)
	parser.add_argument(
		"--interval",
		default=100,
		type=int,
		help="Checking interval in ms (default: 100)."
	)
	parser.add_argument(
		"--cooldown",
		default=3000,
		type=int,
		help="Min ms between state transitions (default: 3000)."
	)
	parser.add_argument(
		"--window",
		default=15,
		type=int,
		help="Sliding window size for averaging (default: 15)."
	)
	parser.add_argument(
		"--upper",
		default=160,
		type=int,
		help="Upper pending-kernel threshold for preempt (default: 160)."
	)
	parser.add_argument(
		"--lower",
		default=32,
		type=int,
		help="Lower pending-kernel threshold for reschedule (default: 32)."
	)
	# dynamic_priority thresholds
	parser.add_argument(
		"--dyn-light",
		default=2,
		type=int,
		help="dynamic_priority: light load threshold (default: 2)."
	)
	parser.add_argument(
		"--dyn-heavy",
		default=8,
		type=int,
		help="dynamic_priority: heavy load threshold (default: 8)."
	)
	parser.add_argument(
		"--dyn-extreme",
		default=20,
		type=int,
		help="dynamic_priority: extreme/freeze threshold (default: 20)."
	)
	parser.add_argument(
		"--dyn-idle-priority",
		default=0,
		type=int,
		help="dynamic_priority: BE priority at idle (default: 0)."
	)
	parser.add_argument(
		"--dyn-light-priority",
		default=4,
		type=int,
		help="dynamic_priority: BE priority at light load (default: 4)."
	)
	parser.add_argument(
		"--dyn-heavy-priority",
		default=12,
		type=int,
		help="dynamic_priority: BE priority at heavy load (default: 12)."
	)
	return parser.parse_args()

# ---------------------------------------------------------------------------
# Sysfs operations
# ---------------------------------------------------------------------------

def sysfs_path(pid, gpu, filename):
	return os.path.join(CGROUP_BASE_DIR, str(pid), str(gpu), filename)

def preempt(pid, gpu):
	log("PREEMPT BE pid={}".format(pid))
	os.kill(pid, signal.SIGSTOP)
	try:
		with open(sysfs_path(pid, gpu, "compute.freeze"), "w") as f:
			f.write("1\n")
	except (IOError, OSError) as e:
		log("  Warning: compute.freeze write failed: {}".format(e))

def reschedule(pid, gpu):
	log("RESCHEDULE BE pid={}".format(pid))
	try:
		with open(sysfs_path(pid, gpu, "compute.freeze"), "w") as f:
			f.write("0\n")
	except (IOError, OSError) as e:
		log("  Warning: compute.freeze write failed: {}".format(e))
	os.kill(pid, signal.SIGCONT)

def set_mem_limit(pid, gpu, limit):
	log("SET memory.limit pid={} limit={}".format(pid, limit))
	try:
		with open(sysfs_path(pid, gpu, "memory.limit"), "w") as f:
			f.write("{}\n".format(limit))
	except (IOError, OSError) as e:
		log("  Warning: memory.limit write failed: {}".format(e))

def set_priority(pid, gpu, priority):
	log("SET compute.priority pid={} priority={}".format(pid, priority))
	try:
		with open(sysfs_path(pid, gpu, "compute.priority"), "w") as f:
			f.write("{}\n".format(priority))
	except (IOError, OSError) as e:
		log("  Warning: compute.priority write failed: {}".format(e))

def get_gcgroup_stat(pid, gpu):
	nr_submitted_kernels = 0
	nr_ended_kernels = 0
	nr_pending_kernels = 0

	try:
		with open(sysfs_path(pid, gpu, "gcgroup.stat"), "r") as f:
			for line in f:
				parts = line.strip().split()
				if len(parts) != 2:
					continue
				key = parts[0].rstrip(":")
				try:
					val = int(parts[1])
				except ValueError:
					continue
				if key == "nr_submitted_kernels":
					nr_submitted_kernels = val
				elif key == "nr_ended_kernels":
					nr_ended_kernels = val
				elif key == "nr_pending_kernels":
					nr_pending_kernels = val
	except (IOError, OSError):
		pass  # process may have exited

	return nr_submitted_kernels, nr_ended_kernels, nr_pending_kernels

# ---------------------------------------------------------------------------
# nvidia-smi GPU utilization fallback
# ---------------------------------------------------------------------------

def get_gpu_utilization():
	"""Returns GPU utilization 0-100, or -1 on error."""
	try:
		result = subprocess.run(
			["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
			capture_output=True, text=True, timeout=5
		)
		if result.returncode != 0:
			return -1
		lines = result.stdout.strip().split("\n")
		return int(lines[0].strip()) if lines else -1
	except Exception:
		return -1

def get_lc_load(lcpid, gpu, use_gpu_util, pending_window):
	"""
	Returns the LC load signal. Tries gcgroup.stat first;
	if counters are always zero, falls back to nvidia-smi utilization
	scaled to a pending-kernel-equivalent (util/4, so 0-25).
	"""
	if not use_gpu_util[0]:
		_, _, pending = get_gcgroup_stat(lcpid, gpu)
		pending_window.append(pending)
		# Check if we should switch: GPU active but counters always zero
		if len(pending_window) >= 10 and sum(pending_window) == 0:
			util = get_gpu_utilization()
			if util > 10:
				use_gpu_util[0] = True
				log("gcgroup.stat counters are zero despite GPU activity ({}%). "
					"Switching to nvidia-smi utilization mode.".format(util))
				return util // 4
		return pending

	# nvidia-smi utilization mode
	util = get_gpu_utilization()
	if util < 0:
		return 0
	return util // 4

# ---------------------------------------------------------------------------
# Policy: burst_freeze (original 3-state machine, improved)
# ---------------------------------------------------------------------------

class BE_STATUS(Enum):
	UNLIMITED = 0
	LIMITED = 1
	PREEMPTED = 2

def run_burst_freeze(args):
	gpu = args.gpu
	window_size = args.window
	upper_threshold = args.upper
	lower_threshold = args.lower
	cooldown_s = args.cooldown / 1000.0

	log("burst_freeze config: upper={} lower={} window={} cooldown={}ms".format(
		upper_threshold, lower_threshold, window_size, args.cooldown))

	pending_window = collections.deque(maxlen=window_size)
	submitted_window = collections.deque(maxlen=4)
	use_gpu_util = [False]  # mutable for pass-by-ref
	raw_pending_window = collections.deque(maxlen=window_size)
	operate_time = 0.0
	be_status = BE_STATUS.UNLIMITED

	while True:
		time.sleep(args.interval / 1000.0)

		load = get_lc_load(args.lcpid, gpu, use_gpu_util, raw_pending_window)
		pending_window.append(load)

		if not use_gpu_util[0]:
			nr_submitted, _, _ = get_gcgroup_stat(args.lcpid, gpu)
			submitted_window.append(nr_submitted)

		now = time.time()
		if now < operate_time + cooldown_s:
			continue

		# LC idle detection: submitted count unchanged (only for gcgroup mode)
		if not use_gpu_util[0] and len(submitted_window) >= 4 and len(set(submitted_window)) == 1:
			if be_status == BE_STATUS.LIMITED:
				set_mem_limit(args.bepid, gpu, ctypes.c_ulong(-1).value)
				be_status = BE_STATUS.UNLIMITED
				operate_time = now
				log("burst_freeze: LIMITED -> UNLIMITED (LC idle)")
			elif be_status == BE_STATUS.PREEMPTED:
				set_mem_limit(args.bepid, gpu, ctypes.c_ulong(-1).value)
				reschedule(args.bepid, gpu)
				be_status = BE_STATUS.UNLIMITED
				operate_time = now
				log("burst_freeze: PREEMPTED -> UNLIMITED (LC idle)")
			continue

		avg_pending = sum(pending_window) / len(pending_window) if pending_window else 0

		if avg_pending > upper_threshold:
			if be_status == BE_STATUS.UNLIMITED:
				preempt(args.bepid, gpu)
				set_mem_limit(args.bepid, gpu, args.bememlimit)
				be_status = BE_STATUS.PREEMPTED
				operate_time = now
				log("burst_freeze: UNLIMITED -> PREEMPTED (avg_pending={:.1f} > {})".format(
					avg_pending, upper_threshold))
			elif be_status == BE_STATUS.LIMITED:
				preempt(args.bepid, gpu)
				be_status = BE_STATUS.PREEMPTED
				operate_time = now
				log("burst_freeze: LIMITED -> PREEMPTED (avg_pending={:.1f} > {})".format(
					avg_pending, upper_threshold))
		elif lower_threshold <= avg_pending <= upper_threshold:
			if be_status == BE_STATUS.UNLIMITED:
				set_mem_limit(args.bepid, gpu, args.bememlimit)
				be_status = BE_STATUS.LIMITED
				operate_time = now
				log("burst_freeze: UNLIMITED -> LIMITED (avg_pending={:.1f})".format(avg_pending))
		elif avg_pending < lower_threshold:
			if be_status == BE_STATUS.PREEMPTED:
				reschedule(args.bepid, gpu)
				be_status = BE_STATUS.LIMITED
				operate_time = now
				log("burst_freeze: PREEMPTED -> LIMITED (avg_pending={:.1f} < {})".format(
					avg_pending, lower_threshold))

# ---------------------------------------------------------------------------
# Policy: dynamic_priority (graduated priority + freeze)
# ---------------------------------------------------------------------------

def run_dynamic_priority(args):
	gpu = args.gpu
	light_threshold = args.dyn_light
	heavy_threshold = args.dyn_heavy
	extreme_threshold = args.dyn_extreme
	idle_priority = args.dyn_idle_priority
	light_priority = args.dyn_light_priority
	heavy_priority = args.dyn_heavy_priority

	log("dynamic_priority config: thresholds=[light={} heavy={} extreme={}] "
		"priorities=[idle={} light={} heavy={}]".format(
		light_threshold, heavy_threshold, extreme_threshold,
		idle_priority, light_priority, heavy_priority))

	use_gpu_util = [False]
	raw_pending_window = collections.deque(maxlen=30)
	last_priority = None
	is_frozen = False

	while True:
		time.sleep(args.interval / 1000.0)

		load = get_lc_load(args.lcpid, gpu, use_gpu_util, raw_pending_window)

		if load >= extreme_threshold:
			if not is_frozen:
				preempt(args.bepid, gpu)
				is_frozen = True
				last_priority = None
				log("dynamic_priority: FREEZE BE (load={} >= extreme={})".format(
					load, extreme_threshold))
		else:
			if is_frozen:
				reschedule(args.bepid, gpu)
				is_frozen = False
				log("dynamic_priority: UNFREEZE BE (load={})".format(load))

			# Determine priority level
			if load < light_threshold:
				target_priority = idle_priority
				level = "idle"
			elif load < heavy_threshold:
				target_priority = light_priority
				level = "light"
			else:
				target_priority = heavy_priority
				level = "heavy"

			if target_priority != last_priority:
				set_priority(args.bepid, gpu, target_priority)
				log("dynamic_priority: BE priority {} -> {} (load={}, level={})".format(
					last_priority, target_priority, load, level))
				last_priority = target_priority

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
	args = parse_args()
	if args.bememlimit == -1:
		args.bememlimit = ctypes.c_ulong(-1).value
	if args.lcmemlimit == -1:
		args.lcmemlimit = ctypes.c_ulong(-1).value

	log("Scheduler starting: policy={} lcpid={} bepid={} gpu={}".format(
		args.policy, args.lcpid, args.bepid, args.gpu))

	# Graceful shutdown
	def shutdown(signum, frame):
		log("Shutting down (signal={}).".format(signum))
		sys.exit(0)
	signal.signal(signal.SIGINT, shutdown)
	signal.signal(signal.SIGTERM, shutdown)

	if args.policy == "burst_freeze":
		run_burst_freeze(args)
	elif args.policy == "dynamic_priority":
		run_dynamic_priority(args)
