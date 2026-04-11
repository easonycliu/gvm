package main

import (
	"fmt"
)

// MemoryRelaxationPolicy implements dynamic memory limit adjustment.
// When HP is idle, it lends HP's unused GPU memory to LP so LP can keep
// more of its working set on-device and swap less. When HP load ramps up,
// it reclaims the memory by shrinking LP's limit back.
type MemoryRelaxationPolicy struct {
	borrowFraction      float64
	safetyMarginBytes   int64
	rampDownStepBytes   int64
	growthThreshold     int64
	fastGrowthThreshold int64
	reclaimLeadtimeMS   int
	pollIntervalMS      int

	// Per-LP state keyed by lpKey
	lpBaseLimits   map[lpKey]int64 // base limit at startup (floor)
	lpCurrentLimit map[lpKey]int64 // last-written limit

	// HP memory tracking for growth rate
	hpMemPrevious map[lpKey]int64 // per HP process, previous memory.current
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

	p.lpBaseLimits = make(map[lpKey]int64)
	p.lpCurrentLimit = make(map[lpKey]int64)
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
	var hpMemUsage, hpMemLimit int64
	for _, hp := range ctx.HPProcesses {
		hpMemUsage += hp.MemoryCurrent
		hpMemLimit += hp.MemoryLimit
	}

	// Compute HP memory growth rate (aggregate)
	hpKey := lpKey{PID: -1, GPUIndex: -1} // sentinel key for aggregate HP
	var hpMemDelta int64
	if p.hpMemInitialized {
		hpMemDelta = hpMemUsage - p.hpMemPrevious[hpKey]
	} else {
		// First tick: delta is unknown, treat as zero
		hpMemDelta = 0
		p.hpMemInitialized = true
	}
	p.hpMemPrevious[hpKey] = hpMemUsage

	hpMemSlack := hpMemLimit - hpMemUsage

	for _, lp := range ctx.LPProcesses {
		key := lpKey{PID: lp.PID, GPUIndex: lp.GPUIndex}

		// Record base limit on first sight (the configured limit at startup)
		if _, exists := p.lpBaseLimits[key]; !exists {
			p.lpBaseLimits[key] = lp.MemoryLimit
			p.lpCurrentLimit[key] = lp.MemoryLimit
			fmt.Printf("[%s] memory_relaxation: LP PID %d GPU %d base_limit=%s\n",
				ts(), lp.PID, lp.GPUIndex, fmtMB(lp.MemoryLimit))
		}

		baseLimit := p.lpBaseLimits[key]
		currentLimit := p.lpCurrentLimit[key]
		var newLimit int64

		// Decision logic
		if ctx.HPPending == 0 && hpMemSlack > p.safetyMarginBytes && hpMemDelta <= 0 {
			// HP is idle, not using its memory, and memory isn't growing
			// Lend surplus memory to LP
			borrowable := int64(float64(hpMemSlack-p.safetyMarginBytes) * p.borrowFraction)
			newLimit = baseLimit + borrowable
		} else if ctx.HPPending > 0 || hpMemDelta > p.growthThreshold {
			// HP is active or memory is growing — reclaim gradually
			newLimit = currentLimit - p.rampDownStepBytes
			if newLimit < baseLimit {
				newLimit = baseLimit
			}
		} else {
			newLimit = currentLimit // no change
		}

		// Predictive reclaim: if HP memory is growing fast, shrink LP preemptively
		if hpMemDelta > p.fastGrowthThreshold {
			leadtimeTicks := int64(p.reclaimLeadtimeMS) / int64(p.pollIntervalMS)
			if leadtimeTicks < 1 {
				leadtimeTicks = 1
			}
			projectedHP := hpMemUsage + hpMemDelta*leadtimeTicks

			// Find GPU total memory for this LP's GPU
			gpuTotal := ctx.GPUTotalMem[lp.GPUIndex]
			if gpuTotal > 0 {
				lpCeiling := gpuTotal - projectedHP - p.safetyMarginBytes
				if lpCeiling < baseLimit {
					lpCeiling = baseLimit
				}
				if newLimit > lpCeiling {
					newLimit = lpCeiling
				}
			}
		}

		// Never go below base limit
		if newLimit < baseLimit {
			newLimit = baseLimit
		}

		// Only write if meaningfully changed (epsilon = 1 MB)
		const epsilon = 1024 * 1024
		if abs64(newLimit-currentLimit) > epsilon {
			actions = append(actions, SchedulingAction{
				PID:         lp.PID,
				GPUIndex:    lp.GPUIndex,
				Action:      "set_memory_limit",
				MemoryLimit: newLimit,
			})
			if newLimit > currentLimit {
				fmt.Printf("[%s] memory_relaxation: LP PID %d GPU %d limit %s -> %s (lending, hp_slack=%s)\n",
					ts(), lp.PID, lp.GPUIndex, fmtMB(currentLimit), fmtMB(newLimit), fmtMB(hpMemSlack))
			} else {
				fmt.Printf("[%s] memory_relaxation: LP PID %d GPU %d limit %s -> %s (reclaiming, hp_delta=%s)\n",
					ts(), lp.PID, lp.GPUIndex, fmtMB(currentLimit), fmtMB(newLimit), fmtMB(hpMemDelta))
			}
			p.lpCurrentLimit[key] = newLimit
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
