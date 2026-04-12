package main

import (
	"fmt"
)

// MemoryRelaxationPolicy implements two-level dynamic memory limit adjustment.
//
// It uses the two-level memory API:
//   - memory.limit.low  = soft reservation (safe zone, guaranteed)
//   - memory.limit.high = hard ceiling (force-evict if exceeded)
//
// When HP is idle, the policy *lends* HP's unused memory to LP by raising
// LP's limit.high above its base limit.high. LP's limit.low stays at its
// configured reservation — the safe floor.
//
// When HP load returns, the policy *reclaims* by lowering LP's limit.high
// back toward limit.low. The kernel handles force-eviction when LP exceeds
// the new limit.high on the next page fault.
type MemoryRelaxationPolicy struct {
	borrowFraction      float64
	safetyMarginBytes   int64
	rampDownStepBytes   int64
	growthThreshold     int64
	fastGrowthThreshold int64
	reclaimLeadtimeMS   int
	pollIntervalMS      int

	// Per-LP state keyed by lpKey
	lpBaseLimitHigh map[lpKey]int64 // initial limit.high at startup (base ceiling)
	lpBaseLimitLow  map[lpKey]int64 // initial limit.low at startup (reservation)
	lpCurrentHigh   map[lpKey]int64 // last-written limit.high

	// HP memory tracking for growth rate
	hpMemPrevious    map[lpKey]int64
	hpMemInitialized bool
}

func (p *MemoryRelaxationPolicy) Name() string {
	return "memory_relaxation"
}

func (p *MemoryRelaxationPolicy) Init(config SchedulerConfig) error {
	p.borrowFraction = config.MemBorrowFraction
	p.safetyMarginBytes = config.MemSafetyMarginBytes
	p.rampDownStepBytes = config.MemRampDownStepBytes
	p.growthThreshold = config.MemGrowthThresholdBytes
	p.fastGrowthThreshold = config.MemFastGrowthThresholdBytes
	p.reclaimLeadtimeMS = config.MemReclaimLeadtimeMS
	p.pollIntervalMS = int(config.PollInterval.Milliseconds())
	if p.pollIntervalMS == 0 {
		p.pollIntervalMS = 50
	}

	p.lpBaseLimitHigh = make(map[lpKey]int64)
	p.lpBaseLimitLow = make(map[lpKey]int64)
	p.lpCurrentHigh = make(map[lpKey]int64)
	p.hpMemPrevious = make(map[lpKey]int64)

	fmt.Printf("[%s] memory_relaxation config: borrow_fraction=%.2f safety_margin=%s ramp_down_step=%s growth_threshold=%s fast_growth=%s leadtime=%dms\n",
		ts(), p.borrowFraction,
		fmtMB(p.safetyMarginBytes), fmtMB(p.rampDownStepBytes),
		fmtMB(p.growthThreshold), fmtMB(p.fastGrowthThreshold),
		p.reclaimLeadtimeMS)
	return nil
}

func (p *MemoryRelaxationPolicy) Tick(ctx SchedulerContext) []SchedulingAction {
	var actions []SchedulingAction

	// Aggregate HP memory stats
	var hpMemUsage, hpMemLimitHigh int64
	for _, hp := range ctx.HPProcesses {
		hpMemUsage += hp.MemoryCurrent
		hpMemLimitHigh += hp.MemoryLimitHigh
	}

	// Compute HP memory growth rate (aggregate)
	hpKey := lpKey{PID: -1, GPUIndex: -1} // sentinel key for aggregate HP
	var hpMemDelta int64
	if p.hpMemInitialized {
		hpMemDelta = hpMemUsage - p.hpMemPrevious[hpKey]
	} else {
		hpMemDelta = 0
		p.hpMemInitialized = true
	}
	p.hpMemPrevious[hpKey] = hpMemUsage

	hpMemSlack := hpMemLimitHigh - hpMemUsage

	for _, lp := range ctx.LPProcesses {
		key := lpKey{PID: lp.PID, GPUIndex: lp.GPUIndex}

		// Record base limits on first sight
		if _, exists := p.lpBaseLimitHigh[key]; !exists {
			p.lpBaseLimitHigh[key] = lp.MemoryLimitHigh
			p.lpBaseLimitLow[key] = lp.MemoryLimitLow
			p.lpCurrentHigh[key] = lp.MemoryLimitHigh
			fmt.Printf("[%s] memory_relaxation: LP PID %d GPU %d base_high=%s base_low=%s\n",
				ts(), lp.PID, lp.GPUIndex,
				fmtMB(lp.MemoryLimitHigh), fmtMB(lp.MemoryLimitLow))
		}

		baseHigh := p.lpBaseLimitHigh[key]
		baseLow := p.lpBaseLimitLow[key]
		currentHigh := p.lpCurrentHigh[key]
		var newHigh int64

		// Floor for limit.high: never drop below limit.low (safe zone)
		floor := baseLow
		if floor <= 0 {
			// If no reservation was set, use base limit.high as the floor
			floor = baseHigh
		}

		// Decision logic
		if ctx.HPPending == 0 && hpMemSlack > p.safetyMarginBytes && hpMemDelta <= 0 {
			// HP is idle — lend surplus memory to LP by raising limit.high
			borrowable := int64(float64(hpMemSlack-p.safetyMarginBytes) * p.borrowFraction)
			newHigh = baseHigh + borrowable
		} else if ctx.HPPending > 0 || hpMemDelta > p.growthThreshold {
			// HP is active or memory growing — reclaim by lowering limit.high
			newHigh = currentHigh - p.rampDownStepBytes
			if newHigh < floor {
				newHigh = floor
			}
		} else {
			newHigh = currentHigh // no change
		}

		// Predictive reclaim: if HP memory is growing fast, shrink LP preemptively
		if hpMemDelta > p.fastGrowthThreshold {
			leadtimeTicks := int64(p.reclaimLeadtimeMS) / int64(p.pollIntervalMS)
			if leadtimeTicks < 1 {
				leadtimeTicks = 1
			}
			projectedHP := hpMemUsage + hpMemDelta*leadtimeTicks

			gpuTotal := ctx.GPUTotalMem[lp.GPUIndex]
			if gpuTotal > 0 {
				lpCeiling := gpuTotal - projectedHP - p.safetyMarginBytes
				if lpCeiling < floor {
					lpCeiling = floor
				}
				if newHigh > lpCeiling {
					newHigh = lpCeiling
				}
			}
		}

		// Never go below floor
		if newHigh < floor {
			newHigh = floor
		}

		// Only write if meaningfully changed (epsilon = 1 MB)
		const epsilon = 1024 * 1024
		if abs64(newHigh-currentHigh) > epsilon {
			actions = append(actions, SchedulingAction{
				PID:             lp.PID,
				GPUIndex:        lp.GPUIndex,
				Action:          "set_memory_limit_high",
				MemoryLimitHigh: newHigh,
			})
			if newHigh > currentHigh {
				fmt.Printf("[%s] memory_relaxation: LP PID %d GPU %d limit.high %s -> %s (lending, hp_slack=%s)\n",
					ts(), lp.PID, lp.GPUIndex, fmtMB(currentHigh), fmtMB(newHigh), fmtMB(hpMemSlack))
			} else {
				fmt.Printf("[%s] memory_relaxation: LP PID %d GPU %d limit.high %s -> %s (reclaiming, hp_delta=%s)\n",
					ts(), lp.PID, lp.GPUIndex, fmtMB(currentHigh), fmtMB(newHigh), fmtMB(hpMemDelta))
			}
			p.lpCurrentHigh[key] = newHigh
		}
	}

	return actions
}

// TickMemoryRelaxation is an exported entry point used by the combined MemoryAwarePolicy.
func (p *MemoryRelaxationPolicy) TickMemoryRelaxation(ctx SchedulerContext) []SchedulingAction {
	return p.Tick(ctx)
}

func abs64(x int64) int64 {
	if x < 0 {
		return -x
	}
	return x
}

func fmtMB(bytes int64) string {
	mb := float64(bytes) / (1024 * 1024)
	return fmt.Sprintf("%.1fMB", mb)
}
