package main

import (
	"fmt"
)

// DynamicPriorityPolicy continuously adjusts LP process priority based on
// HP queue pressure. Instead of binary freeze/unfreeze, it graduates through
// priority levels: idle → light → heavy → extreme (freeze).
type DynamicPriorityPolicy struct {
	lightThreshold   int64
	heavyThreshold   int64
	extremeThreshold int64

	idlePriority  int
	lightPriority int
	heavyPriority int

	// Track last-written state per LP process to skip redundant sysfs writes
	lastPriority map[lpKey]int
	lastFrozen   map[lpKey]bool
}

func (p *DynamicPriorityPolicy) Name() string {
	return "dynamic_priority"
}

func (p *DynamicPriorityPolicy) Init(config SchedulerConfig) error {
	p.lightThreshold = config.DynLightThreshold
	p.heavyThreshold = config.DynHeavyThreshold
	p.extremeThreshold = config.DynExtremeThreshold
	p.idlePriority = config.DynIdlePriority
	p.lightPriority = config.DynLightPriority
	p.heavyPriority = config.DynHeavyPriority
	p.lastPriority = make(map[lpKey]int)
	p.lastFrozen = make(map[lpKey]bool)

	// Initialize lastPriority to -1 so the first write always fires
	// (will be set on first Tick for each process)

	fmt.Printf("[%s] dynamic_priority config: thresholds=[light=%d heavy=%d extreme=%d] priorities=[idle=%d light=%d heavy=%d]\n",
		ts(), p.lightThreshold, p.heavyThreshold, p.extremeThreshold,
		p.idlePriority, p.lightPriority, p.heavyPriority)
	return nil
}

func (p *DynamicPriorityPolicy) Tick(ctx SchedulerContext) []SchedulingAction {
	hpPending := ctx.HPPending
	lpProcesses := ctx.LPProcesses

	var actions []SchedulingAction

	if hpPending >= p.extremeThreshold {
		// Extreme: freeze all LP processes
		for _, lp := range lpProcesses {
			key := lpKey{PID: lp.PID, GPUIndex: lp.GPUIndex}
			if !p.lastFrozen[key] {
				actions = append(actions, SchedulingAction{
					PID:      lp.PID,
					GPUIndex: lp.GPUIndex,
					Action:   "freeze",
				})
				p.lastFrozen[key] = true
				delete(p.lastPriority, key)
				fmt.Printf("[%s] dynamic_priority: FREEZE PID %d GPU %d (hp_pending=%d >= extreme=%d)\n",
					ts(), lp.PID, lp.GPUIndex, hpPending, p.extremeThreshold)
			}
		}
		return actions
	}

	// Determine target priority based on HP queue depth
	var targetPriority int
	var level string
	switch {
	case hpPending == 0:
		targetPriority = p.idlePriority
		level = "idle"
	case hpPending < p.lightThreshold:
		targetPriority = p.idlePriority
		level = "idle"
	case hpPending < p.heavyThreshold:
		targetPriority = p.lightPriority
		level = "light"
	default:
		targetPriority = p.heavyPriority
		level = "heavy"
	}

	for _, lp := range lpProcesses {
		key := lpKey{PID: lp.PID, GPUIndex: lp.GPUIndex}

		// If previously frozen, unfreeze first
		if p.lastFrozen[key] {
			actions = append(actions, SchedulingAction{
				PID:      lp.PID,
				GPUIndex: lp.GPUIndex,
				Action:   "unfreeze",
			})
			p.lastFrozen[key] = false
			fmt.Printf("[%s] dynamic_priority: UNFREEZE PID %d GPU %d (hp_pending=%d, level=%s)\n",
				ts(), lp.PID, lp.GPUIndex, hpPending, level)
		}

		// Only write priority if it changed
		lastPri, exists := p.lastPriority[key]
		if !exists || lastPri != targetPriority {
			actions = append(actions, SchedulingAction{
				PID:      lp.PID,
				GPUIndex: lp.GPUIndex,
				Action:   "set_priority",
				Priority: targetPriority,
			})
			p.lastPriority[key] = targetPriority
			if exists {
				fmt.Printf("[%s] dynamic_priority: PID %d GPU %d priority %d -> %d (hp_pending=%d, level=%s)\n",
					ts(), lp.PID, lp.GPUIndex, lastPri, targetPriority, hpPending, level)
			} else {
				fmt.Printf("[%s] dynamic_priority: PID %d GPU %d priority -> %d (hp_pending=%d, level=%s)\n",
					ts(), lp.PID, lp.GPUIndex, targetPriority, hpPending, level)
			}
		}
	}

	return actions
}
