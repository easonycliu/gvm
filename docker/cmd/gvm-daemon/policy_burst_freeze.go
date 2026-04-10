package main

import (
	"fmt"
)

// BurstFreezePolicy implements a binary freeze/unfreeze policy.
// When the HP process's pending kernel count exceeds HighThreshold,
// all LP processes are frozen. When it drops below LowThreshold, they
// are unfrozen. The two separate thresholds provide hysteresis to
// prevent rapid oscillation.
type BurstFreezePolicy struct {
	highThreshold int64
	lowThreshold  int64
	state         burstState
	// Track freeze state per LP process to avoid redundant writes
	frozenPIDs map[lpKey]bool
}

type burstState int

const (
	burstStateIdle       burstState = iota
	burstStateContention
)

type lpKey struct {
	PID      int
	GPUIndex int
}

func (p *BurstFreezePolicy) Name() string {
	return "burst_freeze"
}

func (p *BurstFreezePolicy) Init(config SchedulerConfig) error {
	p.highThreshold = config.BurstHighThreshold
	p.lowThreshold = config.BurstLowThreshold
	p.state = burstStateIdle
	p.frozenPIDs = make(map[lpKey]bool)

	if p.lowThreshold >= p.highThreshold {
		return fmt.Errorf("burst_freeze: LOW_THRESHOLD (%d) must be less than HIGH_THRESHOLD (%d)",
			p.lowThreshold, p.highThreshold)
	}

	fmt.Printf("[%s] burst_freeze config: high_threshold=%d low_threshold=%d\n",
		ts(), p.highThreshold, p.lowThreshold)
	return nil
}

func (p *BurstFreezePolicy) Tick(hpPending int64, lpProcesses []ProcessInfo) []SchedulingAction {

	var actions []SchedulingAction

	switch p.state {
	case burstStateIdle:
		if hpPending > p.highThreshold {
			p.state = burstStateContention
			fmt.Printf("[%s] burst_freeze: IDLE -> CONTENTION (hp_pending=%d > threshold=%d)\n",
				ts(), hpPending, p.highThreshold)

			for _, lp := range lpProcesses {
				key := lpKey{PID: lp.PID, GPUIndex: lp.GPUIndex}
				if !p.frozenPIDs[key] {
					actions = append(actions, SchedulingAction{
						PID:      lp.PID,
						GPUIndex: lp.GPUIndex,
						Action:   "freeze",
					})
					p.frozenPIDs[key] = true
				}
			}
		}

	case burstStateContention:
		if hpPending < p.lowThreshold {
			p.state = burstStateIdle
			fmt.Printf("[%s] burst_freeze: CONTENTION -> IDLE (hp_pending=%d < threshold=%d)\n",
				ts(), hpPending, p.lowThreshold)

			for _, lp := range lpProcesses {
				key := lpKey{PID: lp.PID, GPUIndex: lp.GPUIndex}
				if p.frozenPIDs[key] {
					actions = append(actions, SchedulingAction{
						PID:      lp.PID,
						GPUIndex: lp.GPUIndex,
						Action:   "unfreeze",
					})
					delete(p.frozenPIDs, key)
				}
			}
		} else {
			// Still in contention — freeze any new LP processes that appeared
			for _, lp := range lpProcesses {
				key := lpKey{PID: lp.PID, GPUIndex: lp.GPUIndex}
				if !p.frozenPIDs[key] {
					actions = append(actions, SchedulingAction{
						PID:      lp.PID,
						GPUIndex: lp.GPUIndex,
						Action:   "freeze",
					})
					p.frozenPIDs[key] = true
					fmt.Printf("[%s] burst_freeze: freezing new LP PID %d GPU %d (still in CONTENTION)\n",
						ts(), lp.PID, lp.GPUIndex)
				}
			}
		}
	}

	return actions
}
