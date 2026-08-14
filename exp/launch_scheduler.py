#!/usr/bin/python3
import argparse
import csv
import ctypes
import os
import signal
import statistics
import time
import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

BASE = "/sys/kernel/debug/nvidia-uvm/processes"
UNLIMITED = ctypes.c_ulong(-1).value
AC_LC_HEAVY_LIMIT = 38000000000


class State(Enum):
    UNLIMITED = 0
    LIMITED = 1
    PREEMPTED = 2


class Demand(Enum):
    IDLE = 0
    UNKNOWN = 1
    LIGHT = 2
    HEAVY = 3


@dataclass
class Sample:
    t: float
    submitted: int
    ended: int
    pending: int


@dataclass
class Observation:
    t: float
    duration: float
    submitted: int
    ended: int
    finish_rate: float
    active_ratio: float
    pending_p90: float
    pending_max: int
    backlog_growth: int
    kernel_time_ms: float


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lcpid", required=True, type=int)
    parser.add_argument("--bepid", required=True, type=int)
    parser.add_argument("--lcmemlimit", required=True, type=int)
    parser.add_argument("--bememlimit", required=True, type=int)
    parser.add_argument("--policy", choices=("ft", "ac"), default="ft")

    # These are the only policy controls normally worth configuring.
    parser.add_argument("--sample-ms", type=int, default=100)
    parser.add_argument("--idle-ms", type=int, default=400)
    parser.add_argument("--lc-slowdown-budget", type=float, default=0.10)
    parser.add_argument("--learning-window", type=int, default=120,
                        help="number of active observations retained per state")
    parser.add_argument("--trace-file")

    # Advanced controls. The controller learns all other thresholds.
    parser.add_argument("--probe-ms", type=int, default=2000)
    parser.add_argument("--probe-backoff-max-ms", type=int, default=30000)
    parser.add_argument("--emergency-margin", type=float, default=2.0)
    return parser.parse_args()


def read_stats(pid):
    names = ("nr_submitted_kernels", "nr_ended_kernels", "nr_pending_kernels")
    values = {}
    path = os.path.join(BASE, str(pid), "0", "gcgroup.stat")
    with open(path) as stat_file:
        for line in stat_file:
            fields = line.split()
            if len(fields) >= 2:
                name = fields[0].rstrip(":")
                if name in names:
                    values[name] = int(fields[1])
    missing = [name for name in names if name not in values]
    if missing:
        raise RuntimeError("Missing gcgroup statistics: " + ", ".join(missing))
    return tuple(values[name] for name in names)


def percentile(values, quantile):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def measure(samples, now, seconds):
    selected = [sample for sample in samples if sample.t >= now - seconds]
    if len(selected) < 2:
        return None
    duration = selected[-1].t - selected[0].t
    if duration <= 0:
        return None
    submitted = selected[-1].submitted - selected[0].submitted
    ended = selected[-1].ended - selected[0].ended
    active_intervals = sum(
        right.submitted != left.submitted
        or right.ended != left.ended
        or right.pending > 0
        for left, right in zip(selected, selected[1:])
    )
    pending = [sample.pending for sample in selected]
    finish_rate = ended / duration
    average_pending = sum(pending) / len(pending)
    return Observation(
        t=now,
        duration=duration,
        submitted=submitted,
        ended=ended,
        finish_rate=finish_rate,
        active_ratio=active_intervals / (len(selected) - 1),
        pending_p90=percentile(pending, 0.90),
        pending_max=max(pending),
        backlog_growth=submitted - ended,
        kernel_time_ms=(average_pending / finish_rate * 1000
                        if finish_rate > 0 else float("inf")),
    )


def set_limit(pid, limit):
    print(f"Set {pid}'s memory limit to {limit}", flush=True)
    path = os.path.join(BASE, str(pid), "0", "memory.limit.high")
    with open(path, "w") as limit_file:
        limit_file.write(f"{limit}\n")


def set_low(pid, limit):
    print(f"Set {pid}'s memory protection to {limit}", flush=True)
    path = os.path.join(BASE, str(pid), "0", "memory.limit.low")
    with open(path, "w") as limit_file:
        limit_file.write(f"{limit}\n")


class Learner:
    """Online, robust state profiles and timing estimates."""

    def __init__(self, capacity, sample_seconds):
        self.profiles = {
            state: deque(maxlen=capacity) for state in State
        }
        self.wave_intervals = deque(maxlen=capacity)
        self.last_progress = None
        self.sample_seconds = sample_seconds

    def observe_progress(self, sample):
        if sample.ended <= 0:
            return
        if self.last_progress is not None:
            interval = sample.t - self.last_progress
            # Ignore long request gaps; they are not engine iteration times.
            if self.sample_seconds * 0.5 <= interval <= 2.0:
                self.wave_intervals.append(interval)
        self.last_progress = sample.t

    def add(self, state, observation):
        if observation and observation.ended > 0 and observation.active_ratio >= 0.5:
            self.profiles[state].append(observation)

    def baseline_rate(self):
        clean = self.profiles[State.PREEMPTED]
        if clean:
            return percentile([item.finish_rate for item in clean], 0.75)
        # During bootstrap LIMITED is safer than UNLIMITED and provides a
        # provisional reference until the first clean calibration finishes.
        limited = self.profiles[State.LIMITED]
        return percentile([item.finish_rate for item in limited], 0.90)

    def clean_guard(self, field, margin, floor):
        clean = self.profiles[State.PREEMPTED]
        if not clean:
            clean = self.profiles[State.LIMITED]
        learned = percentile([getattr(item, field) for item in clean], 0.99)
        return max(floor, learned * margin)

    def iteration_seconds(self):
        if not self.wave_intervals:
            return max(0.2, self.sample_seconds * 2)
        return max(self.sample_seconds, statistics.median(self.wave_intervals))

    def settle_seconds(self):
        # State residence time contains workload pressure and must not be
        # learned as hardware stabilization time. Derive it from engine waves
        # and cap it so overload protection cannot become arbitrarily slow.
        return min(2.0, max(0.5, self.iteration_seconds() * 3))


class DemandLearner:
    """Slow classifier based on independent, stage-mixed rate windows."""

    def __init__(self, capacity):
        self.clean_rates = deque(maxlen=max(12, capacity // 3))
        self.heavy_rate = 0.0
        self.light_rate = 0.0
        self.heavy_votes = 0
        self.light_votes = 0
        self.windows = 0

    def modes(self):
        if not self.heavy_rate or not self.light_rate:
            return None
        return (self.heavy_rate, self.light_rate,
                math.sqrt(self.heavy_rate * self.light_rate))

    @staticmethod
    def bounded_update(center, value, alpha=0.15, limit=0.05):
        target = center + alpha * (value - center)
        return min(center * (1 + limit), max(center * (1 - limit), target))

    def initialize(self):
        positive_rates = [rate for rate in self.clean_rates if rate > 0]
        if len(positive_rates) < 4:
            return
        logs = sorted(math.log(rate) for rate in positive_rates)
        low = percentile(logs, 0.25)
        high = percentile(logs, 0.75)
        if math.exp(high - low) < 1.20:
            return
        self.heavy_rate = math.exp(low)
        self.light_rate = math.exp(high)

    def observe(self, observation, state, demand):
        if observation is None or observation.active_ratio < 0.7:
            return demand
        rate = observation.finish_rate
        clean = state == State.PREEMPTED
        self.windows += 1
        if clean:
            self.clean_rates.append(rate)
            if not self.modes():
                self.initialize()

        modes = self.modes()
        if not modes:
            return Demand.UNKNOWN if demand == Demand.IDLE else demand
        heavy, light, boundary = modes

        # Only clean windows may teach the low-rate center. A high rate observed
        # with BE active is safe evidence for LIGHT because BE cannot speed LC.
        if clean:
            if rate <= boundary:
                self.heavy_rate = self.bounded_update(heavy, rate)
            else:
                self.light_rate = self.bounded_update(light, rate)
        elif rate >= light:
            self.light_rate = self.bounded_update(light, rate)

        _, _, boundary = self.modes()
        # A high completion rate while LIMITED is sufficient to establish that
        # an UNKNOWN workload is LIGHT: restricting BE cannot artificially
        # improve LC throughput.  Once demand is HEAVY, however, recovery must
        # still be demonstrated with BE fully PREEMPTED.  UNLIMITED windows
        # never vote LIGHT because they are probe/interference observations.
        light_evidence = (
            rate > boundary * 1.10
            and (clean or (state == State.LIMITED and demand != Demand.HEAVY))
        )
        if rate < boundary * 0.90:
            self.heavy_votes += 1
            self.light_votes = 0
        elif light_evidence:
            self.light_votes += 1
            self.heavy_votes = 0
        else:
            self.heavy_votes = 0
            self.light_votes = 0

        if self.heavy_votes >= 3:
            return Demand.HEAVY
        if self.light_votes >= 2:
            return Demand.LIGHT
        return Demand.UNKNOWN if demand == Demand.IDLE else demand


class Controller:
    def __init__(self, args):
        self.args = args
        self.state = State.LIMITED
        self.changed = time.monotonic()
        self.learner = Learner(args.learning_window, args.sample_ms / 1000)
        self.demand_learner = DemandLearner(args.learning_window)
        self.evidence = {"slow": 0.0, "healthy": 0.0}
        self.calibrating = False
        self.probe_backoff = args.probe_ms / 1000
        self.next_probe = self.changed + self.probe_backoff
        self.last_observation_ended = None
        self.demand = Demand.IDLE
        self.drain_bad_since = None
        self.uncalibrated_active_since = None
        self.lc_heavy_memory = False

    def enter_heavy_memory(self, now):
        """Give a confirmed HEAVY LC exclusive, protected GPU memory."""
        self.transition(State.PREEMPTED,
                        "heavy LC requires exclusive GPU", now)
        if self.args.policy != "ac" or self.lc_heavy_memory:
            return
        # Protect the existing/default LC residency before allowing vLLM to
        # grow into the larger hard limit. BE is already stopped above.
        set_low(self.args.lcpid, AC_LC_HEAVY_LIMIT)
        set_limit(self.args.lcpid, AC_LC_HEAVY_LIMIT)
        self.lc_heavy_memory = True
        self.changed = now
        self.evidence = {"slow": 0.0, "healthy": 0.0}
        self.last_observation_ended = None

    def leave_heavy_memory(self, now, reason):
        """Return LC to its normal footprint while BE remains preempted."""
        if self.args.policy != "ac" or not self.lc_heavy_memory:
            return False
        set_low(self.args.lcpid, 0)
        set_limit(self.args.lcpid, self.args.lcmemlimit)
        print(f"LC heavy memory -> normal: {reason}", flush=True)
        self.lc_heavy_memory = False
        self.changed = now
        self.evidence = {"slow": 0.0, "healthy": 0.0}
        self.last_observation_ended = None
        self.drain_bad_since = None
        return True

    def transition(self, target, reason, now):
        if target == self.state:
            return
        old = self.state
        if target == State.UNLIMITED:
            set_limit(self.args.bepid, UNLIMITED)
            if old == State.PREEMPTED:
                os.kill(self.args.bepid, signal.SIGCONT)
                print(f"Reschedule {self.args.bepid}", flush=True)
        elif target == State.LIMITED:
            set_limit(self.args.bepid, self.args.bememlimit)
            if old == State.PREEMPTED:
                os.kill(self.args.bepid, signal.SIGCONT)
                print(f"Reschedule {self.args.bepid}", flush=True)
        else:
            print(f"Preempt {self.args.bepid}", flush=True)
            os.kill(self.args.bepid, signal.SIGSTOP)
            set_limit(self.args.bepid, self.args.bememlimit)
        print(f"State {old.name} -> {target.name}: {reason}", flush=True)
        self.state = target
        self.changed = now
        self.evidence = {"slow": 0.0, "healthy": 0.0}
        self.last_observation_ended = None
        self.drain_bad_since = None

    def is_idle(self, idle_samples):
        if len(idle_samples) < 2:
            return False
        required = self.args.idle_ms / 1000
        if idle_samples[-1].t - idle_samples[0].t < required * 0.75:
            return False
        first = idle_samples[0]
        return all(
            item.submitted == first.submitted
            and item.ended == first.ended
            and item.pending == 0
            and item.submitted == item.ended
            for item in idle_samples
        )

    def evaluate(self, now, idle_samples, observation, demand_observation,
                 current_pending):
        if self.is_idle(idle_samples):
            already_idle = self.demand == Demand.IDLE and self.state == State.UNLIMITED
            self.calibrating = False
            self.demand = Demand.IDLE
            self.drain_bad_since = None
            self.uncalibrated_active_since = None
            self.leave_heavy_memory(now, "LC idle")
            self.transition(State.UNLIMITED, "LC idle", now)
            return ("idle-hold" if already_idle else "idle"), 1.0, 0.0, 0.0, 0.0

        if observation is None or observation.ended <= 0:
            return "warming", 1.0, 0.0, 0.0, 0.0

        self.learner.add(self.state, observation)
        if (demand_observation is not None and
                demand_observation.t - demand_observation.duration >= self.changed):
            self.demand = self.demand_learner.observe(
                demand_observation, self.state, self.demand)
        if self.demand == Demand.IDLE:
            self.demand = Demand.UNKNOWN
        modes = self.demand_learner.modes()
        if modes:
            heavy_rate, light_rate, _ = modes
            baseline = heavy_rate if self.demand == Demand.HEAVY else light_rate
        else:
            baseline = self.learner.baseline_rate()
        relative = observation.finish_rate / baseline if baseline else 1.0
        pending_guard = self.learner.clean_guard(
            "pending_p90", self.args.emergency_margin, 16)
        backlog_guard = self.learner.clean_guard(
            "backlog_growth", self.args.emergency_margin, 32)
        kernel_guard = self.learner.clean_guard(
            "kernel_time_ms", self.args.emergency_margin, 20)

        # Count evidence in estimated engine iterations, not milliseconds.
        delta_ended = observation.ended
        if self.last_observation_ended is not None:
            delta_ended = max(0, observation.ended - self.last_observation_ended)
        self.last_observation_ended = observation.ended
        # Each overlapping observation contributes at most one vote. This keeps
        # one large kernel burst from satisfying the confidence test by itself.
        evidence_step = 1.0 if delta_ended else 0.0
        slow = baseline > 0 and relative < 1.0 - self.args.lc_slowdown_budget
        healthy = baseline > 0 and relative >= 1.0 - self.args.lc_slowdown_budget / 2
        self.evidence["slow"] = self.evidence["slow"] + evidence_step if slow else 0.0
        self.evidence["healthy"] = (self.evidence["healthy"] + evidence_step
                                    if healthy else 0.0)

        # Historical P90 pressure must not keep a completed wave alive. A wave
        # is draining badly only while work is currently still pending.
        drain_bad = current_pending > 0 and (
            current_pending > pending_guard
            or observation.backlog_growth > backlog_guard
            or observation.kernel_time_ms > kernel_guard)
        if drain_bad:
            if self.drain_bad_since is None:
                self.drain_bad_since = now
        else:
            self.drain_bad_since = None
        # Submission-wave backlog is normal. It becomes an emergency only when
        # it survives several learned engine iterations without draining.
        drain_deadline = max(0.5, self.learner.iteration_seconds() * 3)
        persistent_drain = (self.drain_bad_since is not None and
                            now - self.drain_bad_since >= drain_deadline)
        settled = now - self.changed >= self.learner.settle_seconds()

        # The first active request after startup gets a clean calibration. Once
        # learned, an isolated request does not immediately restrict BE.
        clean_count = len(self.learner.profiles[State.PREEMPTED])
        if clean_count == 0 and observation.active_ratio >= 0.8:
            if self.uncalibrated_active_since is None:
                self.uncalibrated_active_since = now
        else:
            self.uncalibrated_active_since = None
        calibration_delay = max(1.0, self.learner.iteration_seconds() * 8)
        needs_initial_calibration = (
            clean_count == 0
            and self.uncalibrated_active_since is not None
            and now - self.uncalibrated_active_since >= calibration_delay
        )
        if needs_initial_calibration and self.state != State.PREEMPTED:
            self.calibrating = True
            self.uncalibrated_active_since = None
            self.transition(State.PREEMPTED, "initial LC calibration", now)
            return "calibrate", relative, pending_guard, backlog_guard, kernel_guard

        if persistent_drain and self.state != State.PREEMPTED:
            self.calibrating = False
            self.probe_backoff = min(
                self.args.probe_backoff_max_ms / 1000,
                max(self.args.probe_ms / 1000, self.probe_backoff * 2),
            )
            self.transition(State.PREEMPTED, "LC wave failed to drain", now)
            return "drain-emergency", relative, pending_guard, backlog_guard, kernel_guard

        confirm = 3.0
        if self.state == State.PREEMPTED:
            enough_clean_data = clean_count >= max(4, min(12, self.args.learning_window // 10))
            enough_demand_data = bool(self.demand_learner.clean_rates)
            relieved = healthy and not persistent_drain
            if not relieved:
                self.evidence["healthy"] = 0.0
            # HEAVY is an exclusivity policy, not just a rate baseline. Keep BE
            # stopped even when heavy LC matches its clean completion rate.
            if self.demand == Demand.HEAVY:
                self.enter_heavy_memory(now)
                self.evidence["healthy"] = 0.0
                decision = "exclusive"
            elif self.args.policy == "ac" and self.lc_heavy_memory:
                # The 38 GB allocation is intentionally sticky. A temporary
                # LIGHT/UNKNOWN classification must not resize vLLM again;
                # only the idle path above returns LC to its normal limit.
                self.evidence["healthy"] = 0.0
                decision = "heavy-memory-hold"
            elif (settled and enough_clean_data and enough_demand_data and
                  self.evidence["healthy"] >= confirm):
                self.calibrating = False
                self.next_probe = now + self.probe_backoff
                target = (State.UNLIMITED if self.demand == Demand.LIGHT
                          else State.LIMITED)
                self.transition(target, "clean LC demand mode learned", now)
                decision = "recover"
            else:
                decision = "calibrate" if self.calibrating else "protect"
        elif self.state == State.LIMITED:
            if self.demand == Demand.HEAVY:
                self.enter_heavy_memory(now)
                decision = "exclusive"
            elif settled and self.evidence["slow"] >= confirm:
                self.transition(State.PREEMPTED, "LC exceeds slowdown budget", now)
                decision = "slow"
            elif (self.demand == Demand.LIGHT and settled and
                  now >= self.next_probe and self.evidence["healthy"] >= confirm):
                self.transition(State.UNLIMITED, "probe BE without limit", now)
                decision = "probe"
            else:
                decision = "healthy" if healthy else "observe"
        else:
            if self.demand == Demand.HEAVY:
                self.next_probe = now + self.probe_backoff
                self.enter_heavy_memory(now)
                decision = "exclusive"
            elif settled and self.evidence["slow"] >= confirm:
                self.probe_backoff = min(
                    self.args.probe_backoff_max_ms / 1000,
                    max(self.args.probe_ms / 1000, self.probe_backoff * 2),
                )
                self.next_probe = now + self.probe_backoff
                self.transition(State.LIMITED, "UNLIMITED exceeds slowdown budget", now)
                decision = "slow"
            else:
                if settled and self.evidence["healthy"] >= confirm:
                    self.probe_backoff = max(self.args.probe_ms / 1000,
                                             self.probe_backoff / 2)
                decision = "healthy" if healthy else "observe"

        return decision, relative, pending_guard, backlog_guard, kernel_guard


def main():
    args = arguments()
    if args.sample_ms <= 0 or args.idle_ms <= 0 or args.learning_window < 4:
        raise ValueError("sample-ms and idle-ms must be positive; learning-window must be >= 4")
    if not 0 < args.lc_slowdown_budget < 1:
        raise ValueError("lc-slowdown-budget must be between 0 and 1")
    if args.emergency_margin <= 1:
        raise ValueError("emergency-margin must be greater than 1")
    if args.policy == "ac" and args.lcmemlimit >= AC_LC_HEAVY_LIMIT:
        raise ValueError("AC normal LC limit must be below its 38 GB HEAVY limit")
    args.bememlimit = UNLIMITED if args.bememlimit == -1 else args.bememlimit

    sample_seconds = args.sample_ms / 1000
    idle_seconds = args.idle_ms / 1000
    samples = deque()
    demand_samples = deque()
    controller = Controller(args)
    started = time.monotonic()
    demand_window_seconds = 3.0
    next_demand_observation = started + demand_window_seconds
    previous_ended = None
    trace = None
    writer = None

    if args.trace_file:
        path = Path(args.trace_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        trace = path.open("w", newline="")
        writer = csv.writer(trace)
        writer.writerow([
            "time_s", "policy", "state", "demand", "decision",
            "lc_limit_high", "lc_limit_low",
            "submitted", "ended", "pending",
            "finish_rate", "baseline_rate", "relative_rate", "active_ratio",
            "heavy_mode_rate", "light_mode_rate", "mode_boundary_rate",
            "pending_p90", "pending_guard", "backlog_growth", "backlog_guard",
            "kernel_time_ms", "kernel_time_guard_ms", "iteration_ms",
            "settle_ms", "clean_samples", "limited_samples",
            "unlimited_samples", "probe_backoff_ms", "drain_bad_ms",
            "demand_windows", "heavy_votes", "light_votes",
        ])

    try:
        while True:
            time.sleep(sample_seconds)
            now = time.monotonic()
            submitted, ended, pending = read_stats(args.lcpid)
            current = Sample(now, submitted, ended, pending)
            samples.append(current)
            demand_samples.append(current)
            delta_ended = 0 if previous_ended is None else max(0, ended - previous_ended)
            previous_ended = ended
            controller.learner.observe_progress(
                Sample(now, submitted, delta_ended, pending))

            # Eight learned completion waves normally span several prefill and
            # decode stages, making the comparison insensitive to one stage.
            observation_seconds = max(
                1.0, controller.learner.iteration_seconds() * 8)
            keep_seconds = max(idle_seconds, observation_seconds) + sample_seconds * 2
            while samples and samples[0].t < now - keep_seconds:
                samples.popleft()
            idle_samples = [item for item in samples if item.t >= now - idle_seconds]
            observation = measure(samples, now, observation_seconds)
            if (observation is not None and
                    observation.duration < observation_seconds * 0.8):
                observation = None
            demand_observation = None
            if now >= next_demand_observation:
                demand_observation = measure(
                    demand_samples, now, demand_window_seconds)
                if (demand_observation is not None and
                        demand_observation.duration < demand_window_seconds * 0.8):
                    demand_observation = None
                demand_samples.clear()
                demand_samples.append(current)
                next_demand_observation = now + demand_window_seconds
            previous_state = controller.state
            previous_lc_heavy = controller.lc_heavy_memory
            decision, relative, pending_guard, backlog_guard, kernel_guard = (
                controller.evaluate(now, idle_samples, observation,
                                    demand_observation, pending)
            )

            # Never compare or learn from an observation spanning two BE
            # states or two LC memory configurations.
            lc_memory_changed = controller.lc_heavy_memory != previous_lc_heavy
            if (decision == "idle" or controller.state != previous_state
                    or lc_memory_changed):
                samples.clear()
                samples.append(current)
            if lc_memory_changed:
                demand_samples.clear()
                demand_samples.append(current)
                next_demand_observation = now + demand_window_seconds

            if writer:
                current_observation = observation or Observation(
                    now, 0, 0, 0, 0, 0, 0, pending, 0, 0)
                profiles = controller.learner.profiles
                modes = controller.demand_learner.modes()
                heavy_rate, light_rate, mode_boundary = modes or (0, 0, 0)
                drain_bad_ms = (0 if controller.drain_bad_since is None else
                                (now - controller.drain_bad_since) * 1000)
                writer.writerow([
                    f"{now - started:.3f}", args.policy,
                    controller.state.name, controller.demand.name, decision,
                    (AC_LC_HEAVY_LIMIT if controller.lc_heavy_memory
                     else args.lcmemlimit),
                    (AC_LC_HEAVY_LIMIT if controller.lc_heavy_memory else 0),
                    submitted, ended, pending,
                    f"{current_observation.finish_rate:.3f}",
                    f"{(heavy_rate if controller.demand == Demand.HEAVY else light_rate) or controller.learner.baseline_rate():.3f}",
                    f"{relative:.3f}", f"{current_observation.active_ratio:.3f}",
                    f"{heavy_rate:.3f}", f"{light_rate:.3f}",
                    f"{mode_boundary:.3f}",
                    f"{current_observation.pending_p90:.3f}", f"{pending_guard:.3f}",
                    current_observation.backlog_growth, f"{backlog_guard:.3f}",
                    f"{current_observation.kernel_time_ms:.3f}", f"{kernel_guard:.3f}",
                    f"{controller.learner.iteration_seconds() * 1000:.3f}",
                    f"{controller.learner.settle_seconds() * 1000:.3f}",
                    len(profiles[State.PREEMPTED]), len(profiles[State.LIMITED]),
                    len(profiles[State.UNLIMITED]),
                    f"{controller.probe_backoff * 1000:.3f}",
                    f"{drain_bad_ms:.3f}",
                    controller.demand_learner.windows,
                    controller.demand_learner.heavy_votes,
                    controller.demand_learner.light_votes,
                ])
                trace.flush()
    finally:
        if trace:
            trace.close()


if __name__ == "__main__":
    main()
