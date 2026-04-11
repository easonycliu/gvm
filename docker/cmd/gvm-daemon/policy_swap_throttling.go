package main

import (
	"fmt"
)

// SwapThrottlingPolicy monitors LP's memory.swap.current and throttles
// LP compute when swap pressure is high. When LP is actively swapping,
// page faults block GPU context switches and interfere with HP even when
// the compute scheduler assigns HP its timeslice.
type SwapThrottlingPolicy struct {
	freezeRatio   float64
	throttleRatio float64
	clearRatio    float64
	cooldownTicks int

	// Per-LP state
	cooldownRemaining map[lpKey]int
	lastLPSwap        map[lpKey]int64
	lastPriority      map[lpKey]int
	lastFrozen        map[lpKey]bool
}

func (p *SwapThrottlingPolicy) Name() string {
	return "swap_throttling"
}

func (p *SwapThrottlingPolicy) Init(config SchedulerConfig) error {
	p.freezeRatio = config.SwapFreezeRatio
	p.throttleRatio = config.SwapThrottleRatio
	p.clearRatio = config.SwapClearRatio
	p.cooldownTicks = config.SwapCooldownTicks

	if p.throttleRatio >= p.freezeRatio {
		return fmt.Errorf("swap_throttling: throttle_ratio (%.2f) must be less than freeze_ratio (%.2f)",
			p.throttleRatio, p.freezeRatio)
	}

	p.cooldownRemaining = make(map[lpKey]int)
	p.lastLPSwap = make(map[lpKey]int64)
	p.lastPriority = make(map[lpKey]int)
	p.lastFrozen = make(map[lpKey]bool)

	fmt.Printf("[%s] swap_throttling config: freeze_ratio=%.2f throttle_ratio=%.2f clear_ratio=%.2f cooldown_ticks=%d\n",
		ts(), p.freezeRatio, p.throttleRatio, p.clearRatio, p.cooldownTicks)
	return nil
}

func (p *SwapThrottlingPolicy) Tick(ctx SchedulerContext) []SchedulingAction {
	return p.tickSwap(ctx)
}

// tickSwap performs the swap-pressure evaluation. Returns actions and a bool
// indicating whether swap override is active (for use by combined policy).
func (p *SwapThrottlingPolicy) tickSwap(ctx SchedulerContext) []SchedulingAction {
	var actions []SchedulingAction

	for _, lp := range ctx.LPProcesses {
		key := lpKey{PID: lp.PID, GPUIndex: lp.GPUIndex}

		lpSwap := lp.MemorySwapCurrent
		lpMemLimit := lp.MemoryLimit
		if lpMemLimit <= 0 {
			// No limit set — can't compute ratio, skip
			continue
		}

		swapRatio := float64(lpSwap) / float64(lpMemLimit)

		// Track swap velocity
		_ = lpSwap - p.lastLPSwap[key] // swapDelta — for future use
		p.lastLPSwap[key] = lpSwap

		// If in cooldown (previously frozen), count down
		if p.cooldownRemaining[key] > 0 {
			p.cooldownRemaining[key]--
			if p.cooldownRemaining[key] == 0 {
				// Cooldown expired — unfreeze
				actions = append(actions, SchedulingAction{
					PID:      lp.PID,
					GPUIndex: lp.GPUIndex,
					Action:   "unfreeze",
				})
				p.lastFrozen[key] = false
				fmt.Printf("[%s] swap_throttling: LP PID %d GPU %d UNFREEZE (cooldown expired, swap_ratio=%.3f)\n",
					ts(), lp.PID, lp.GPUIndex, swapRatio)
			}
			continue // skip further decisions during cooldown
		}

		// Decision logic
		if swapRatio > p.freezeRatio {
			// Heavy swap pressure — freeze LP to let page faults drain
			if !p.lastFrozen[key] {
				actions = append(actions, SchedulingAction{
					PID:      lp.PID,
					GPUIndex: lp.GPUIndex,
					Action:   "freeze",
				})
				p.lastFrozen[key] = true
			}
			p.cooldownRemaining[key] = p.cooldownTicks
			fmt.Printf("[%s] swap_throttling: LP PID %d GPU %d FREEZE (swap_ratio=%.3f > %.2f, swap=%s, cooldown=%d ticks)\n",
				ts(), lp.PID, lp.GPUIndex, swapRatio, p.freezeRatio,
				fmtMB(lpSwap), p.cooldownTicks)

		} else if swapRatio > p.throttleRatio {
			// Moderate swap pressure — reduce LP priority proportionally
			// Map swap_ratio linearly to priority range [4, 12]
			frac := (swapRatio - p.throttleRatio) / (p.freezeRatio - p.throttleRatio)
			throttlePriority := 4 + int(frac*8)
			if throttlePriority < 4 {
				throttlePriority = 4
			}
			if throttlePriority > 12 {
				throttlePriority = 12
			}

			// Unfreeze if was frozen
			if p.lastFrozen[key] {
				actions = append(actions, SchedulingAction{
					PID:      lp.PID,
					GPUIndex: lp.GPUIndex,
					Action:   "unfreeze",
				})
				p.lastFrozen[key] = false
			}

			// Only write priority if changed
			if p.lastPriority[key] != throttlePriority {
				actions = append(actions, SchedulingAction{
					PID:      lp.PID,
					GPUIndex: lp.GPUIndex,
					Action:   "set_priority",
					Priority: throttlePriority,
				})
				fmt.Printf("[%s] swap_throttling: LP PID %d GPU %d priority -> %d (swap_ratio=%.3f)\n",
					ts(), lp.PID, lp.GPUIndex, throttlePriority, swapRatio)
				p.lastPriority[key] = throttlePriority
			}

		} else if swapRatio < p.clearRatio {
			// Swap is minimal — clear any throttling we applied
			if p.lastFrozen[key] {
				actions = append(actions, SchedulingAction{
					PID:      lp.PID,
					GPUIndex: lp.GPUIndex,
					Action:   "unfreeze",
				})
				p.lastFrozen[key] = false
			}
			// Don't reset priority here — let the compute policy handle it
		}
	}

	return actions
}

// HasSwapOverride returns true if any LP process is currently under
// active swap-based throttling (frozen, in cooldown, or priority-reduced due to swap).
func (p *SwapThrottlingPolicy) HasSwapOverride(ctx SchedulerContext) bool {
	for _, lp := range ctx.LPProcesses {
		key := lpKey{PID: lp.PID, GPUIndex: lp.GPUIndex}
		if p.cooldownRemaining[key] > 0 || p.lastFrozen[key] || p.lastPriority[key] > 0 {
			return true
		}
	}
	return false
}
