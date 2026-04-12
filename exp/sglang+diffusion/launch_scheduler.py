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
		choices=["burst_freeze", "dynamic_priority", "memory_relaxation", "swap_throttling", "memory_aware"],
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
	# Two-level memory limits
	parser.add_argument(
		"--lc-memlimit-high",
		default=-1,
		type=int,
		help="LC memory.limit.high in bytes (-1 = unlimited)."
	)
	parser.add_argument(
		"--lc-memlimit-low",
		default=0,
		type=int,
		help="LC memory.limit.low in bytes (0 = no reservation)."
	)
	parser.add_argument(
		"--be-memlimit-high",
		default=-1,
		type=int,
		help="BE memory.limit.high in bytes (-1 = unlimited)."
	)
	parser.add_argument(
		"--be-memlimit-low",
		default=0,
		type=int,
		help="BE memory.limit.low in bytes (0 = no reservation)."
	)
	# Memory relaxation parameters
	parser.add_argument(
		"--mem-borrow-fraction",
		default=0.7,
		type=float,
		help="Memory relaxation: fraction of HP slack to lend to BE (default: 0.7)."
	)
	parser.add_argument(
		"--mem-safety-mb",
		default=50,
		type=int,
		help="Memory relaxation: safety margin in MB (default: 50)."
	)
	parser.add_argument(
		"--mem-ramp-down-mb",
		default=100,
		type=int,
		help="Memory relaxation: ramp-down step in MB (default: 100)."
	)
	parser.add_argument(
		"--mem-growth-mb",
		default=10,
		type=int,
		help="Memory relaxation: growth threshold in MB (default: 10)."
	)
	parser.add_argument(
		"--mem-fast-growth-mb",
		default=50,
		type=int,
		help="Memory relaxation: fast growth threshold in MB (default: 50)."
	)
	parser.add_argument(
		"--mem-leadtime-ms",
		default=200,
		type=int,
		help="Memory relaxation: predictive reclaim leadtime in ms (default: 200)."
	)
	# Swap throttling parameters
	parser.add_argument(
		"--swap-freeze-ratio",
		default=0.3,
		type=float,
		help="Swap throttling: freeze ratio (default: 0.3)."
	)
	parser.add_argument(
		"--swap-throttle-ratio",
		default=0.1,
		type=float,
		help="Swap throttling: throttle ratio (default: 0.1)."
	)
	parser.add_argument(
		"--swap-clear-ratio",
		default=0.02,
		type=float,
		help="Swap throttling: clear ratio (default: 0.02)."
	)
	parser.add_argument(
		"--swap-cooldown-ticks",
		default=4,
		type=int,
		help="Swap throttling: cooldown ticks (default: 4)."
	)
	# Combined memory_aware policy
	parser.add_argument(
		"--compute-policy",
		default="dynamic_priority",
		choices=["burst_freeze", "dynamic_priority"],
		help="For memory_aware: underlying compute policy (default: dynamic_priority)."
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
	"""Legacy: sets memory.limit.high."""
	set_mem_limit_high(pid, gpu, limit)

def set_mem_limit_high(pid, gpu, limit):
	log("SET memory.limit.high pid={} limit={}".format(pid, limit))
	try:
		with open(sysfs_path(pid, gpu, "memory.limit.high"), "w") as f:
			f.write("{}\n".format(limit))
	except (IOError, OSError) as e:
		log("  Warning: memory.limit.high write failed: {}".format(e))

def set_mem_limit_low(pid, gpu, limit):
	log("SET memory.limit.low pid={} limit={}".format(pid, limit))
	try:
		with open(sysfs_path(pid, gpu, "memory.limit.low"), "w") as f:
			f.write("{}\n".format(limit))
	except (IOError, OSError) as e:
		log("  Warning: memory.limit.low write failed: {}".format(e))

def get_mem_current(pid, gpu):
	"""Read memory.current (bytes on GPU device)."""
	try:
		with open(sysfs_path(pid, gpu, "memory.current"), "r") as f:
			return int(f.read().strip())
	except (IOError, OSError, ValueError):
		return 0

def get_mem_limit_high(pid, gpu):
	"""Read memory.limit.high."""
	try:
		with open(sysfs_path(pid, gpu, "memory.limit.high"), "r") as f:
			return int(f.read().strip())
	except (IOError, OSError, ValueError):
		return -1

def get_mem_swap_current(pid, gpu):
	"""Read memory.swap.current."""
	try:
		with open(sysfs_path(pid, gpu, "memory.swap.current"), "r") as f:
			return int(f.read().strip())
	except (IOError, OSError, ValueError):
		return 0

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
# Policy: memory_relaxation (two-level dynamic limit.high adjustment)
# ---------------------------------------------------------------------------

def fmt_mb(b):
	return "{:.1f}MB".format(b / (1024 * 1024))

def run_memory_relaxation(args):
	gpu = args.gpu
	borrow_fraction = args.mem_borrow_fraction
	safety_margin = args.mem_safety_mb * 1024 * 1024
	ramp_down_step = args.mem_ramp_down_mb * 1024 * 1024
	growth_threshold = args.mem_growth_mb * 1024 * 1024
	fast_growth_threshold = args.mem_fast_growth_mb * 1024 * 1024
	reclaim_leadtime_ms = args.mem_leadtime_ms
	poll_interval_ms = args.interval

	log("memory_relaxation config: borrow={:.2f} safety={}MB ramp_down={}MB growth={}MB fast_growth={}MB leadtime={}ms".format(
		borrow_fraction, args.mem_safety_mb, args.mem_ramp_down_mb,
		args.mem_growth_mb, args.mem_fast_growth_mb, reclaim_leadtime_ms))

	# State
	base_limit_high = None  # BE's initial limit.high (recorded on first read)
	base_limit_low = None   # BE's initial limit.low
	current_high = None     # last-written limit.high
	hp_mem_prev = None      # LC memory.current from previous tick
	hp_mem_initialized = False

	use_gpu_util = [False]
	raw_pending_window = collections.deque(maxlen=30)

	while True:
		time.sleep(poll_interval_ms / 1000.0)

		# Read LC (HP) stats
		lc_mem_current = get_mem_current(args.lcpid, gpu)
		lc_mem_limit_high = get_mem_limit_high(args.lcpid, gpu)
		lc_load = get_lc_load(args.lcpid, gpu, use_gpu_util, raw_pending_window)

		# Read BE (LP) stats
		be_mem_limit_high = get_mem_limit_high(args.bepid, gpu)

		# Record base limits on first sight
		if base_limit_high is None:
			base_limit_high = be_mem_limit_high
			base_limit_low = args.be_memlimit_low if args.be_memlimit_low > 0 else 0
			current_high = be_mem_limit_high
			log("memory_relaxation: BE base_high={} base_low={}".format(
				fmt_mb(base_limit_high), fmt_mb(base_limit_low)))

		# HP memory growth
		hp_mem_delta = 0
		if hp_mem_initialized:
			hp_mem_delta = lc_mem_current - hp_mem_prev
		else:
			hp_mem_initialized = True
		hp_mem_prev = lc_mem_current

		hp_slack = lc_mem_limit_high - lc_mem_current if lc_mem_limit_high > 0 else 0

		# Floor: never drop below limit.low or base_high
		floor = base_limit_low if base_limit_low > 0 else base_limit_high

		# Decision logic
		if lc_load == 0 and hp_slack > safety_margin and hp_mem_delta <= 0:
			borrowable = int((hp_slack - safety_margin) * borrow_fraction)
			new_high = base_limit_high + borrowable
		elif lc_load > 0 or hp_mem_delta > growth_threshold:
			new_high = current_high - ramp_down_step
			if new_high < floor:
				new_high = floor
		else:
			new_high = current_high

		# Predictive reclaim
		if hp_mem_delta > fast_growth_threshold:
			leadtime_ticks = max(1, reclaim_leadtime_ms // poll_interval_ms)
			projected_hp = lc_mem_current + hp_mem_delta * leadtime_ticks
			# Try to read GPU total memory from nvidia-smi
			try:
				result = subprocess.run(
					["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
					capture_output=True, text=True, timeout=5)
				gpu_total = int(result.stdout.strip().split("\n")[0]) * 1024 * 1024  # MiB → bytes
			except Exception:
				gpu_total = 0
			if gpu_total > 0:
				lp_ceiling = gpu_total - projected_hp - safety_margin
				if lp_ceiling < floor:
					lp_ceiling = floor
				if new_high > lp_ceiling:
					new_high = lp_ceiling

		if new_high < floor:
			new_high = floor

		# Only write if meaningfully changed (> 1MB)
		if abs(new_high - current_high) > 1024 * 1024:
			set_mem_limit_high(args.bepid, gpu, new_high)
			if new_high > current_high:
				log("memory_relaxation: BE limit.high {} -> {} (lending, hp_slack={})".format(
					fmt_mb(current_high), fmt_mb(new_high), fmt_mb(hp_slack)))
			else:
				log("memory_relaxation: BE limit.high {} -> {} (reclaiming, hp_delta={})".format(
					fmt_mb(current_high), fmt_mb(new_high), fmt_mb(hp_mem_delta)))
			current_high = new_high

# ---------------------------------------------------------------------------
# Policy: swap_throttling
# ---------------------------------------------------------------------------

def run_swap_throttling(args):
	gpu = args.gpu
	freeze_ratio = args.swap_freeze_ratio
	throttle_ratio = args.swap_throttle_ratio
	clear_ratio = args.swap_clear_ratio
	cooldown_ticks = args.swap_cooldown_ticks

	log("swap_throttling config: freeze={:.2f} throttle={:.2f} clear={:.2f} cooldown={}".format(
		freeze_ratio, throttle_ratio, clear_ratio, cooldown_ticks))

	cooldown_remaining = 0
	is_frozen = False
	throttle_priority = None

	while True:
		time.sleep(args.interval / 1000.0)

		be_swap = get_mem_swap_current(args.bepid, gpu)
		be_limit_high = get_mem_limit_high(args.bepid, gpu)

		if be_limit_high <= 0:
			continue

		swap_ratio = be_swap / be_limit_high

		# Cooldown
		if cooldown_remaining > 0:
			cooldown_remaining -= 1
			if cooldown_remaining == 0:
				reschedule(args.bepid, gpu)
				is_frozen = False
				log("swap_throttling: cooldown expired, unfreeze BE")
			continue

		if swap_ratio >= freeze_ratio:
			if not is_frozen:
				preempt(args.bepid, gpu)
				is_frozen = True
				cooldown_remaining = cooldown_ticks
				log("swap_throttling: FREEZE BE (swap_ratio={:.3f} >= {:.2f}, swap={}, cooldown={})".format(
					swap_ratio, freeze_ratio, fmt_mb(be_swap), cooldown_ticks))
		elif swap_ratio >= throttle_ratio:
			# Interpolate priority 4-12 within throttle zone
			zone_width = freeze_ratio - throttle_ratio
			if zone_width > 0:
				t = (swap_ratio - throttle_ratio) / zone_width
			else:
				t = 0.5
			new_priority = int(4 + t * 8)
			if new_priority != throttle_priority:
				set_priority(args.bepid, gpu, new_priority)
				log("swap_throttling: BE priority -> {} (swap_ratio={:.3f})".format(new_priority, swap_ratio))
				throttle_priority = new_priority
		elif swap_ratio < clear_ratio:
			if throttle_priority is not None:
				set_priority(args.bepid, gpu, 0)
				log("swap_throttling: cleared, BE priority -> 0")
				throttle_priority = None

# ---------------------------------------------------------------------------
# Policy: memory_aware (combined: memory_relaxation + swap_throttling + compute)
# ---------------------------------------------------------------------------

def run_memory_aware(args):
	gpu = args.gpu
	compute_policy = args.compute_policy

	# Memory relaxation state
	borrow_fraction = args.mem_borrow_fraction
	safety_margin = args.mem_safety_mb * 1024 * 1024
	ramp_down_step = args.mem_ramp_down_mb * 1024 * 1024
	growth_threshold = args.mem_growth_mb * 1024 * 1024
	fast_growth_threshold = args.mem_fast_growth_mb * 1024 * 1024
	reclaim_leadtime_ms = args.mem_leadtime_ms
	poll_interval_ms = args.interval

	# Swap throttling state
	swap_freeze_ratio = args.swap_freeze_ratio
	swap_throttle_ratio = args.swap_throttle_ratio
	swap_clear_ratio = args.swap_clear_ratio
	swap_cooldown_ticks = args.swap_cooldown_ticks

	# Dynamic priority / burst freeze state
	light_threshold = args.dyn_light
	heavy_threshold = args.dyn_heavy
	extreme_threshold = args.dyn_extreme
	idle_priority = args.dyn_idle_priority
	light_priority = args.dyn_light_priority
	heavy_priority = args.dyn_heavy_priority
	upper_threshold = args.upper
	lower_threshold = args.lower

	log("memory_aware config: compute_policy={}".format(compute_policy))
	log("  memory_relaxation: borrow={:.2f} safety={}MB ramp_down={}MB".format(
		borrow_fraction, args.mem_safety_mb, args.mem_ramp_down_mb))
	log("  swap_throttling: freeze={:.2f} throttle={:.2f} clear={:.2f} cooldown={}".format(
		swap_freeze_ratio, swap_throttle_ratio, swap_clear_ratio, swap_cooldown_ticks))

	# Memory relaxation state
	base_limit_high = None
	base_limit_low = None
	current_high = None
	hp_mem_prev = None
	hp_mem_initialized = False

	# Swap state
	swap_cooldown_remaining = 0
	swap_is_frozen = False
	swap_throttle_priority = None

	# Compute state
	compute_last_priority = None
	compute_is_frozen = False

	use_gpu_util = [False]
	raw_pending_window = collections.deque(maxlen=30)

	while True:
		time.sleep(poll_interval_ms / 1000.0)

		# ---- Read stats ----
		lc_mem_current = get_mem_current(args.lcpid, gpu)
		lc_mem_limit_high = get_mem_limit_high(args.lcpid, gpu)
		lc_load = get_lc_load(args.lcpid, gpu, use_gpu_util, raw_pending_window)
		be_mem_limit_high = get_mem_limit_high(args.bepid, gpu)
		be_swap = get_mem_swap_current(args.bepid, gpu)

		# ---- 1. Memory Relaxation ----
		if base_limit_high is None:
			base_limit_high = be_mem_limit_high
			base_limit_low = args.be_memlimit_low if args.be_memlimit_low > 0 else 0
			current_high = be_mem_limit_high
			log("memory_relaxation: BE base_high={} base_low={}".format(
				fmt_mb(base_limit_high), fmt_mb(base_limit_low)))

		hp_mem_delta = 0
		if hp_mem_initialized:
			hp_mem_delta = lc_mem_current - hp_mem_prev
		else:
			hp_mem_initialized = True
		hp_mem_prev = lc_mem_current

		hp_slack = lc_mem_limit_high - lc_mem_current if lc_mem_limit_high > 0 else 0
		floor = base_limit_low if base_limit_low > 0 else base_limit_high

		if lc_load == 0 and hp_slack > safety_margin and hp_mem_delta <= 0:
			borrowable = int((hp_slack - safety_margin) * borrow_fraction)
			new_high = base_limit_high + borrowable
		elif lc_load > 0 or hp_mem_delta > growth_threshold:
			new_high = current_high - ramp_down_step
			if new_high < floor:
				new_high = floor
		else:
			new_high = current_high

		if new_high < floor:
			new_high = floor

		if abs(new_high - current_high) > 1024 * 1024:
			set_mem_limit_high(args.bepid, gpu, new_high)
			if new_high > current_high:
				log("memory_relaxation: BE limit.high {} -> {} (lending)".format(
					fmt_mb(current_high), fmt_mb(new_high)))
			else:
				log("memory_relaxation: BE limit.high {} -> {} (reclaiming)".format(
					fmt_mb(current_high), fmt_mb(new_high)))
			current_high = new_high

		# ---- 2. Swap Throttling ----
		swap_override_active = False
		if be_mem_limit_high > 0:
			swap_ratio = be_swap / be_mem_limit_high

			if swap_cooldown_remaining > 0:
				swap_cooldown_remaining -= 1
				swap_override_active = True
				if swap_cooldown_remaining == 0:
					reschedule(args.bepid, gpu)
					swap_is_frozen = False
					log("swap_throttling: cooldown expired, unfreeze BE")
			elif swap_ratio >= swap_freeze_ratio:
				if not swap_is_frozen:
					preempt(args.bepid, gpu)
					swap_is_frozen = True
					swap_cooldown_remaining = swap_cooldown_ticks
					swap_override_active = True
					log("swap_throttling: FREEZE BE (swap_ratio={:.3f})".format(swap_ratio))
			elif swap_ratio >= swap_throttle_ratio:
				zone_width = swap_freeze_ratio - swap_throttle_ratio
				t = (swap_ratio - swap_throttle_ratio) / zone_width if zone_width > 0 else 0.5
				new_priority = int(4 + t * 8)
				if new_priority != swap_throttle_priority:
					set_priority(args.bepid, gpu, new_priority)
					log("swap_throttling: BE priority -> {} (swap_ratio={:.3f})".format(new_priority, swap_ratio))
					swap_throttle_priority = new_priority
				swap_override_active = True
			elif swap_ratio < swap_clear_ratio:
				if swap_throttle_priority is not None:
					swap_throttle_priority = None

		# ---- 3. Compute Policy (only if swap is not overriding) ----
		if not swap_override_active:
			if compute_policy == "dynamic_priority":
				if lc_load >= extreme_threshold:
					if not compute_is_frozen:
						preempt(args.bepid, gpu)
						compute_is_frozen = True
						compute_last_priority = None
						log("dynamic_priority: FREEZE BE (load={})".format(lc_load))
				else:
					if compute_is_frozen:
						reschedule(args.bepid, gpu)
						compute_is_frozen = False
						log("dynamic_priority: UNFREEZE BE (load={})".format(lc_load))
					if lc_load < light_threshold:
						target = idle_priority
					elif lc_load < heavy_threshold:
						target = light_priority
					else:
						target = heavy_priority
					if target != compute_last_priority:
						set_priority(args.bepid, gpu, target)
						log("dynamic_priority: BE priority -> {} (load={})".format(target, lc_load))
						compute_last_priority = target
			elif compute_policy == "burst_freeze":
				if lc_load > upper_threshold:
					if not compute_is_frozen:
						preempt(args.bepid, gpu)
						compute_is_frozen = True
						log("burst_freeze: FREEZE BE (load={})".format(lc_load))
				elif lc_load < lower_threshold:
					if compute_is_frozen:
						reschedule(args.bepid, gpu)
						compute_is_frozen = False
						log("burst_freeze: UNFREEZE BE (load={})".format(lc_load))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
	args = parse_args()
	if args.bememlimit == -1:
		args.bememlimit = ctypes.c_ulong(-1).value
	if args.lcmemlimit == -1:
		args.lcmemlimit = ctypes.c_ulong(-1).value

	# Apply initial two-level memory limits if specified
	if args.lc_memlimit_high > 0:
		set_mem_limit_high(args.lcpid, args.gpu, args.lc_memlimit_high)
	if args.lc_memlimit_low > 0:
		set_mem_limit_low(args.lcpid, args.gpu, args.lc_memlimit_low)
	if args.be_memlimit_high > 0:
		set_mem_limit_high(args.bepid, args.gpu, args.be_memlimit_high)
	elif args.bememlimit > 0:
		# Backwards compat: --bememlimit sets limit.high
		set_mem_limit_high(args.bepid, args.gpu, args.bememlimit)
	if args.be_memlimit_low > 0:
		set_mem_limit_low(args.bepid, args.gpu, args.be_memlimit_low)

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
	elif args.policy == "memory_relaxation":
		run_memory_relaxation(args)
	elif args.policy == "swap_throttling":
		run_swap_throttling(args)
	elif args.policy == "memory_aware":
		run_memory_aware(args)
