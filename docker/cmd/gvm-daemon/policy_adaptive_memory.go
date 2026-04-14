package main

import (
	"fmt"
)

// AdaptiveMemoryPolicy implements bidirectional GPU memory management between
// HP (latency-critical, e.g. vLLM) and LP (best-effort, e.g. diffusion).
//
// Two key innovations over the simpler MemoryRelaxationPolicy:
//
//  1. HP cache sizing: apply memory.limit.high to the HP process (vLLM) to
//     control its KV cache size. When HP workload is low, shrink its cache to
//     free GPU memory for LP. When HP workload spikes, expand its cache.
//
//  2. Dynamic limit.low: LP's reservation (limit.low) is not fixed — it is
//     raised when memory frees up (HP cache shrunk) and lowered when HP needs
//     memory back. This lets LP *request* more guaranteed memory at runtime.
//
// The memory flow is:
//
//	HP idle  → HP.limit.high ↓ → LP.limit.low ↑, LP.limit.high ↑
//	HP busy  → LP.limit.low ↓ → LP.limit.high ↓ → HP.limit.high ↑
//
// An underlying compute policy (burst_freeze or dynamic_priority) handles
// compute scheduling orthogonally.
type AdaptiveMemoryPolicy struct {
	// Configuration
	hpIdleThreshold  int64   // HP load below this = idle (shrink HP cache)
	hpBusyThreshold  int64   // HP load above this = busy (expand HP cache)
	hpMinCacheFrac   float64 // minimum fraction of GPU for HP cache (e.g. 0.2)
	hpMaxCacheFrac   float64 // maximum fraction of GPU for HP cache (e.g. 0.8)
	lpMinReservation int64   // absolute minimum LP limit.low (bytes)
	rampStepBytes    int64   // adjustment step per tick (bytes)
	safetyMargin     int64   // buffer to prevent OOM (bytes)
	pollIntervalMS   int

	// Sub-policies
	swapThrottle  *SwapThrottlingPolicy
	computePolicy SchedulerPolicy

	// Per-HP state
	hpCurrentHigh map[lpKey]int64 // last-written HP limit.high
	hpInitialized map[lpKey]bool  // whether we've seen this HP before

	// Per-LP state
	lpCurrentHigh map[lpKey]int64 // last-written LP limit.high
	lpCurrentLow  map[lpKey]int64 // last-written LP limit.low
	lpBaseHigh    map[lpKey]int64 // LP's initial limit.high
	lpBaseLow     map[lpKey]int64 // LP's initial limit.low
	lpInitialized map[lpKey]bool

	// HP memory tracking
	hpMemPrev        int64
	hpMemInitialized bool
}

func (p *AdaptiveMemoryPolicy) Name() string {
	return "adaptive_memory"
}

func (p *AdaptiveMemoryPolicy) Init(config SchedulerConfig) error {
	p.hpIdleThreshold = config.AdaptiveHPIdleThreshold
	p.hpBusyThreshold = config.AdaptiveHPBusyThreshold
	p.hpMinCacheFrac = config.AdaptiveHPMinCacheFrac
	p.hpMaxCacheFrac = config.AdaptiveHPMaxCacheFrac
	p.lpMinReservation = config.AdaptiveLPMinReservationBytes
	p.rampStepBytes = config.AdaptiveRampStepBytes
	p.safetyMargin = config.AdaptiveSafetyMarginBytes
	p.pollIntervalMS = int(config.PollInterval.Milliseconds())
	if p.pollIntervalMS == 0 {
		p.pollIntervalMS = 50
	}

	p.hpCurrentHigh = make(map[lpKey]int64)
	p.hpInitialized = make(map[lpKey]bool)
	p.lpCurrentHigh = make(map[lpKey]int64)
	p.lpCurrentLow = make(map[lpKey]int64)
	p.lpBaseHigh = make(map[lpKey]int64)
	p.lpBaseLow = make(map[lpKey]int64)
	p.lpInitialized = make(map[lpKey]bool)

	// Initialize swap throttling sub-policy
	p.swapThrottle = &SwapThrottlingPolicy{}
	if err := p.swapThrottle.Init(config); err != nil {
		return fmt.Errorf("adaptive_memory: swap_throttling init: %w", err)
	}

	// Initialize underlying compute policy
	switch config.ComputePolicy {
	case "burst_freeze":
		p.computePolicy = &BurstFreezePolicy{}
	case "dynamic_priority":
		p.computePolicy = &DynamicPriorityPolicy{}
	default:
		return fmt.Errorf("adaptive_memory: unknown compute policy %q", config.ComputePolicy)
	}
	if err := p.computePolicy.Init(config); err != nil {
		return fmt.Errorf("adaptive_memory: compute policy init: %w", err)
	}

	fmt.Printf("[%s] adaptive_memory config: hp_idle=%d hp_busy=%d hp_cache=[%.0f%%,%.0f%%] lp_min_reservation=%s ramp_step=%s safety=%s compute=%s\n",
		ts(), p.hpIdleThreshold, p.hpBusyThreshold,
		p.hpMinCacheFrac*100, p.hpMaxCacheFrac*100,
		fmtMB(p.lpMinReservation), fmtMB(p.rampStepBytes), fmtMB(p.safetyMargin),
		p.computePolicy.Name())
	return nil
}

func (p *AdaptiveMemoryPolicy) Tick(ctx SchedulerContext) []SchedulingAction {
	var actions []SchedulingAction

	// --- Phase 1: HP cache sizing + LP reservation adjustment ---
	memActions := p.tickAdaptiveMemory(ctx)
	actions = append(actions, memActions...)

	// --- Phase 2: Swap-pressure throttling ---
	swapActions := p.swapThrottle.tickSwap(ctx)

	// --- Phase 3: Compute policy (unless swap override) ---
	if p.swapThrottle.HasSwapOverride(ctx) {
		actions = append(actions, swapActions...)
	} else {
		computeActions := p.computePolicy.Tick(ctx)
		actions = append(actions, computeActions...)
	}

	return actions
}

func (p *AdaptiveMemoryPolicy) tickAdaptiveMemory(ctx SchedulerContext) []SchedulingAction {
	var actions []SchedulingAction

	// Aggregate HP stats
	var hpMemTotal, hpLimitHighTotal int64
	for _, hp := range ctx.HPProcesses {
		hpMemTotal += hp.MemoryCurrent
		hpLimitHighTotal += hp.MemoryLimitHigh
	}

	// HP memory growth rate
	var hpMemDelta int64
	if p.hpMemInitialized {
		hpMemDelta = hpMemTotal - p.hpMemPrev
	} else {
		p.hpMemInitialized = true
	}
	p.hpMemPrev = hpMemTotal

	hpLoad := ctx.HPPending

	// === HP Cache Sizing ===
	// Adjust each HP process's limit.high based on workload
	for _, hp := range ctx.HPProcesses {
		hpKey := lpKey{PID: hp.PID, GPUIndex: hp.GPUIndex}
		gpuTotal := ctx.GPUTotalMem[hp.GPUIndex]
		if gpuTotal <= 0 {
			continue
		}

		hpMinCache := int64(float64(gpuTotal) * p.hpMinCacheFrac)
		hpMaxCache := int64(float64(gpuTotal) * p.hpMaxCacheFrac)

		// Record initial HP limit.high
		if !p.hpInitialized[hpKey] {
			p.hpCurrentHigh[hpKey] = hp.MemoryLimitHigh
			p.hpInitialized[hpKey] = true
			fmt.Printf("[%s] adaptive_memory: HP PID %d GPU %d initial limit.high=%s (cache range=[%s, %s])\n",
				ts(), hp.PID, hp.GPUIndex,
				fmtMB(hp.MemoryLimitHigh), fmtMB(hpMinCache), fmtMB(hpMaxCache))
		}

		currentHPHigh := p.hpCurrentHigh[hpKey]
		var newHPHigh int64

		if hpLoad <= p.hpIdleThreshold && hpMemDelta <= 0 {
			// HP is idle — shrink cache to free memory for LP
			// Target: just above current usage + safety margin
			target := hp.MemoryCurrent + p.safetyMargin
			if target < hpMinCache {
				target = hpMinCache
			}
			// Ramp down toward target
			newHPHigh = currentHPHigh - p.rampStepBytes
			if newHPHigh < target {
				newHPHigh = target
			}
		} else if hpLoad >= p.hpBusyThreshold || hpMemDelta > 0 {
			// HP is busy or growing — expand cache
			newHPHigh = currentHPHigh + p.rampStepBytes
			if newHPHigh > hpMaxCache {
				newHPHigh = hpMaxCache
			}
		} else {
			newHPHigh = currentHPHigh
		}

		// Clamp
		if newHPHigh < hpMinCache {
			newHPHigh = hpMinCache
		}
		if newHPHigh > hpMaxCache {
			newHPHigh = hpMaxCache
		}

		// Apply if changed significantly
		const epsilon = 1024 * 1024 // 1 MB
		if abs64(newHPHigh-currentHPHigh) > epsilon {
			actions = append(actions, SchedulingAction{
				PID:             hp.PID,
				GPUIndex:        hp.GPUIndex,
				Action:          "set_memory_limit_high",
				MemoryLimitHigh: newHPHigh,
			})
			if newHPHigh < currentHPHigh {
				fmt.Printf("[%s] adaptive_memory: HP PID %d GPU %d limit.high %s -> %s (shrinking cache, load=%d)\n",
					ts(), hp.PID, hp.GPUIndex, fmtMB(currentHPHigh), fmtMB(newHPHigh), hpLoad)
			} else {
				fmt.Printf("[%s] adaptive_memory: HP PID %d GPU %d limit.high %s -> %s (expanding cache, load=%d)\n",
					ts(), hp.PID, hp.GPUIndex, fmtMB(currentHPHigh), fmtMB(newHPHigh), hpLoad)
			}
			p.hpCurrentHigh[hpKey] = newHPHigh
		}
	}

	// === LP Dynamic Reservation ===
	// Compute available memory after HP usage + safety
	for _, lp := range ctx.LPProcesses {
		lk := lpKey{PID: lp.PID, GPUIndex: lp.GPUIndex}
		gpuTotal := ctx.GPUTotalMem[lp.GPUIndex]
		if gpuTotal <= 0 {
			continue
		}

		// Record initial LP limits
		if !p.lpInitialized[lk] {
			p.lpBaseHigh[lk] = lp.MemoryLimitHigh
			p.lpBaseLow[lk] = lp.MemoryLimitLow
			p.lpCurrentHigh[lk] = lp.MemoryLimitHigh
			p.lpCurrentLow[lk] = lp.MemoryLimitLow
			p.lpInitialized[lk] = true
			fmt.Printf("[%s] adaptive_memory: LP PID %d GPU %d initial limit.high=%s limit.low=%s\n",
				ts(), lp.PID, lp.GPUIndex,
				fmtMB(lp.MemoryLimitHigh), fmtMB(lp.MemoryLimitLow))
		}

		currentLPHigh := p.lpCurrentHigh[lk]
		currentLPLow := p.lpCurrentLow[lk]

		// How much memory is available for LP?
		// available = gpuTotal - (sum of HP current limit.high) - safety
		var totalHPReserved int64
		for _, hp := range ctx.HPProcesses {
			hk := lpKey{PID: hp.PID, GPUIndex: hp.GPUIndex}
			if reserved, ok := p.hpCurrentHigh[hk]; ok && reserved > 0 {
				totalHPReserved += reserved
			} else {
				totalHPReserved += hp.MemoryCurrent
			}
		}
		available := gpuTotal - totalHPReserved - p.safetyMargin

		var newLPHigh, newLPLow int64

		if hpLoad <= p.hpIdleThreshold && hpMemDelta <= 0 {
			// HP is idle — give LP more memory
			// Raise limit.low toward available (dynamic reservation increase)
			newLPLow = currentLPLow + p.rampStepBytes
			if newLPLow > available {
				newLPLow = available
			}
			// limit.high should be at least limit.low
			newLPHigh = available
			if newLPHigh < newLPLow {
				newLPHigh = newLPLow
			}
		} else if hpLoad >= p.hpBusyThreshold || hpMemDelta > 0 {
			// HP is busy — reclaim from LP
			// Step 1: lower limit.low (soft reclaim — reduce guaranteed memory)
			newLPLow = currentLPLow - p.rampStepBytes
			if newLPLow < p.lpMinReservation {
				newLPLow = p.lpMinReservation
			}
			// Step 2: lower limit.high (hard reclaim — may trigger eviction)
			newLPHigh = currentLPHigh - p.rampStepBytes
			if newLPHigh < newLPLow {
				newLPHigh = newLPLow
			}
		} else {
			// Moderate load — hold steady
			newLPLow = currentLPLow
			newLPHigh = currentLPHigh
		}

		// Floor: LP must always keep at least lpMinReservation
		if newLPLow < p.lpMinReservation {
			newLPLow = p.lpMinReservation
		}
		if newLPHigh < p.lpMinReservation {
			newLPHigh = p.lpMinReservation
		}
		// Invariant: limit.high >= limit.low
		if newLPHigh < newLPLow {
			newLPHigh = newLPLow
		}

		const epsilon = 1024 * 1024 // 1 MB

		// Apply limit.low change
		if abs64(newLPLow-currentLPLow) > epsilon {
			actions = append(actions, SchedulingAction{
				PID:            lp.PID,
				GPUIndex:       lp.GPUIndex,
				Action:         "set_memory_limit_low",
				MemoryLimitLow: newLPLow,
			})
			if newLPLow > currentLPLow {
				fmt.Printf("[%s] adaptive_memory: LP PID %d GPU %d limit.low %s -> %s (reservation increased, available=%s)\n",
					ts(), lp.PID, lp.GPUIndex, fmtMB(currentLPLow), fmtMB(newLPLow), fmtMB(available))
			} else {
				fmt.Printf("[%s] adaptive_memory: LP PID %d GPU %d limit.low %s -> %s (reservation reduced, hp_load=%d)\n",
					ts(), lp.PID, lp.GPUIndex, fmtMB(currentLPLow), fmtMB(newLPLow), hpLoad)
			}
			p.lpCurrentLow[lk] = newLPLow
		}

		// Apply limit.high change
		if abs64(newLPHigh-currentLPHigh) > epsilon {
			actions = append(actions, SchedulingAction{
				PID:             lp.PID,
				GPUIndex:        lp.GPUIndex,
				Action:          "set_memory_limit_high",
				MemoryLimitHigh: newLPHigh,
			})
			if newLPHigh > currentLPHigh {
				fmt.Printf("[%s] adaptive_memory: LP PID %d GPU %d limit.high %s -> %s (expanded)\n",
					ts(), lp.PID, lp.GPUIndex, fmtMB(currentLPHigh), fmtMB(newLPHigh))
			} else {
				fmt.Printf("[%s] adaptive_memory: LP PID %d GPU %d limit.high %s -> %s (reclaimed, hp_load=%d)\n",
					ts(), lp.PID, lp.GPUIndex, fmtMB(currentLPHigh), fmtMB(newLPHigh), hpLoad)
			}
			p.lpCurrentHigh[lk] = newLPHigh
		}
	}

	return actions
}
