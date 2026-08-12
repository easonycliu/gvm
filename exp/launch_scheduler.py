#!/usr/bin/python3
import argparse, csv, ctypes, os, signal, time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

BASE = "/sys/kernel/debug/nvidia-uvm/processes"
UNLIMITED = ctypes.c_ulong(-1).value

class State(Enum):
    UNLIMITED = 0
    LIMITED = 1
    PREEMPTED = 2

@dataclass
class Sample:
    t: float
    submitted: int
    ended: int
    pending: int

@dataclass
class Metrics:
    duration: float
    submit_rate: float
    finish_rate: float
    active_ratio: float
    pending_average: float
    pending_p90: float
    pending_max: int
    backlog_growth: int
    kernel_time: float

def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--lcpid", required=True, type=int)
    p.add_argument("--bepid", required=True, type=int)
    p.add_argument("--lcmemlimit", required=True, type=int)
    p.add_argument("--bememlimit", required=True, type=int)
    p.add_argument("--checking-interval-ms", type=int, default=100)
    p.add_argument("--idle-window-ms", type=int, default=400)
    p.add_argument("--critical-window-ms", type=int, default=500)
    p.add_argument("--pressure-window-ms", type=int, default=2000)
    p.add_argument("--settle-ms", type=int, default=500)
    p.add_argument("--moderate-confirm-ms", type=int, default=500)
    p.add_argument("--recovery-confirm-ms", type=int, default=800)
    p.add_argument("--probe-confirm-ms", type=int, default=2000)
    p.add_argument("--minimum-active-ratio", type=float, default=.7)
    p.add_argument("--moderate-relative-rate", type=float, default=.75)
    p.add_argument("--healthy-relative-rate", type=float, default=.9)
    p.add_argument("--critical-relative-rate", type=float, default=.4)
    p.add_argument("--moderate-pending", type=float, default=4)
    p.add_argument("--critical-pending", type=float, default=128)
    p.add_argument("--critical-backlog-growth", type=int, default=64)
    p.add_argument("--moderate-kernel-time-ms", type=float, default=20)
    p.add_argument("--critical-kernel-time-ms", type=float, default=100)
    p.add_argument("--trace-file")
    return p.parse_args()

def stats(pid):
    keys = ("nr_submitted_kernels", "nr_ended_kernels", "nr_pending_kernels")
    values = {}
    with open(os.path.join(BASE, str(pid), "0", "gcgroup.stat")) as f:
        for line in f:
            fields = line.split()
            if len(fields) >= 2:
                key = fields[0].rstrip(":")
                if key in keys:
                    values[key] = int(fields[1])
    missing = [key for key in keys if key not in values]
    if missing:
        raise RuntimeError("Missing gcgroup statistics: " + ", ".join(missing))
    return tuple(values[key] for key in keys)

def pct(values, q):
    values = sorted(values)
    if not values: return 0
    x = (len(values)-1)*q; lo = int(x); hi = min(lo+1, len(values)-1)
    return values[lo]*(hi-x) + values[hi]*(x-lo)

def window(samples, now, seconds):
    selected = [x for x in samples if x.t >= now-seconds]
    if len(selected) < 2: return None
    duration = selected[-1].t-selected[0].t
    if duration <= 0: return None
    ds = selected[-1].submitted-selected[0].submitted
    de = selected[-1].ended-selected[0].ended
    active = sum(b.submitted != a.submitted or b.ended != a.ended or b.pending > 0
                 or b.submitted > b.ended for a,b in zip(selected, selected[1:]))
    pending = [x.pending for x in selected]
    finish = de/duration; average = sum(pending)/len(pending)
    return Metrics(duration, ds/duration, finish, active/(len(selected)-1), average,
                   pct(pending,.9), max(pending), ds-de,
                   average/finish if finish > 0 else float("inf"))

def set_limit(pid, limit):
    print(f"Set {pid}'s memory limit to {limit}", flush=True)
    with open(os.path.join(BASE,str(pid),"0","memory.limit.high"),"w") as f:
        f.write(f"{limit}\n")

class Controller:
    def __init__(self, a):
        self.a=a; self.state=State.LIMITED; self.changed=time.monotonic()
        self.since={}; self.rates=deque(maxlen=120); self.reference=0
    def confirmed(self,name,value,now,ms):
        if not value: self.since.pop(name,None); return False
        return now-self.since.setdefault(name,now) >= ms/1000
    def transition(self,target,reason,now):
        if target == self.state: return
        old=self.state
        if target == State.UNLIMITED:
            set_limit(self.a.bepid,UNLIMITED)
            if old == State.PREEMPTED: os.kill(self.a.bepid,signal.SIGCONT); print(f"Reschedule {self.a.bepid}",flush=True)
        elif target == State.LIMITED:
            set_limit(self.a.bepid,self.a.bememlimit)
            if old == State.PREEMPTED: os.kill(self.a.bepid,signal.SIGCONT); print(f"Reschedule {self.a.bepid}",flush=True)
        else:
            print(f"Preempt {self.a.bepid}",flush=True); os.kill(self.a.bepid,signal.SIGSTOP)
            set_limit(self.a.bepid,self.a.bememlimit)
        print(f"State {old.name} -> {target.name}: {reason}",flush=True)
        self.state=target; self.changed=now; self.since.clear()
    def evaluate(self,now,idle,fast,pressure):
        ready_idle = idle and idle[-1].t-idle[0].t >= self.a.idle_window_ms/1000*.75
        is_idle = ready_idle and all(x.submitted==idle[0].submitted and x.ended==idle[0].ended
                                     and x.pending==0 and x.submitted==x.ended for x in idle)
        if is_idle:
            self.transition(State.UNLIMITED,"LC idle",now); return "idle",0
        pressure_ready = pressure and pressure.duration >= self.a.pressure_window_ms/1000*.8 and pressure.active_ratio >= self.a.minimum_active_ratio
        if pressure_ready and pressure.finish_rate>0 and pressure.backlog_growth<=0:
            self.rates.append(pressure.finish_rate); self.reference=max(self.reference,pct(self.rates,.9))
        relative = pressure.finish_rate/self.reference if pressure and self.reference else 1
        fast_ready = fast and fast.duration >= self.a.critical_window_ms/1000*.6 and fast.active_ratio >= self.a.minimum_active_ratio
        critical = bool((fast_ready and (fast.pending_p90>=self.a.critical_pending or fast.backlog_growth>=self.a.critical_backlog_growth)) or
                        (pressure_ready and self.reference and relative<self.a.critical_relative_rate and
                         (pressure.pending_average>=self.a.moderate_pending or pressure.kernel_time*1000>=self.a.critical_kernel_time_ms)))
        moderate = bool(pressure_ready and ((self.reference and relative<self.a.moderate_relative_rate) or
                        pressure.kernel_time*1000>=self.a.moderate_kernel_time_ms or
                        (pressure.pending_average>=self.a.moderate_pending and pressure.backlog_growth>0)))
        healthy = bool(pressure_ready and relative>=self.a.healthy_relative_rate and
                       pressure.pending_average<self.a.moderate_pending and pressure.backlog_growth<=0)
        settled=now-self.changed>=self.a.settle_ms/1000
        if critical:
            self.transition(State.PREEMPTED,"critical LC pressure",now); decision="critical"
        elif self.state == State.UNLIMITED:
            if settled and self.confirmed("moderate",moderate,now,self.a.moderate_confirm_ms): self.transition(State.LIMITED,"sustained LC pressure",now)
            decision="moderate" if moderate else "healthy"
        elif self.state == State.LIMITED:
            if settled and self.confirmed("probe",healthy,now,self.a.probe_confirm_ms): self.transition(State.UNLIMITED,"LC healthy; probe BE",now)
            decision="healthy" if healthy else "moderate"
        else:
            relieved=bool(pressure_ready and not moderate)
            if settled and self.confirmed("recovery",relieved,now,self.a.recovery_confirm_ms): self.transition(State.LIMITED,"LC pressure relieved",now)
            decision="recovering" if relieved else "critical"
        return decision,relative

def main():
    a=arguments(); a.bememlimit=UNLIMITED if a.bememlimit==-1 else a.bememlimit
    keep=max(a.idle_window_ms,a.critical_window_ms,a.pressure_window_ms)/1000
    samples=deque(); control=Controller(a); started=time.monotonic()
    trace=None; writer=None
    if a.trace_file:
        path=Path(a.trace_file); path.parent.mkdir(parents=True,exist_ok=True); trace=path.open("w",newline="")
        writer=csv.writer(trace); writer.writerow("time_s state decision submitted ended pending submit_rate finish_rate reference_rate relative_rate active_ratio pending_average pending_p90 pending_max backlog_growth estimated_kernel_time_ms".split())
    try:
        while True:
            time.sleep(a.checking_interval_ms/1000); now=time.monotonic(); s,e,p=stats(a.lcpid)
            current=Sample(now,s,e,p); samples.append(current)
            while samples and samples[0].t<now-keep-.2: samples.popleft()
            idle=[x for x in samples if x.t>=now-a.idle_window_ms/1000]
            fast=window(samples,now,a.critical_window_ms/1000); pressure=window(samples,now,a.pressure_window_ms/1000)
            decision,relative=control.evaluate(now,idle,fast,pressure)
            if decision=="idle": samples.clear(); samples.append(current); fast=pressure=None
            if writer:
                m=pressure or Metrics(0,0,0,0,0,0,p,0,0)
                writer.writerow([f"{now-started:.3f}",control.state.name,decision,s,e,p,f"{m.submit_rate:.3f}",f"{m.finish_rate:.3f}",f"{control.reference:.3f}",f"{relative:.3f}",f"{m.active_ratio:.3f}",f"{m.pending_average:.3f}",f"{m.pending_p90:.3f}",m.pending_max,m.backlog_growth,f"{m.kernel_time*1000:.3f}"]); trace.flush()
    finally:
        if trace: trace.close()

if __name__ == "__main__": main()
