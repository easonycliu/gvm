package main

import (
	"fmt"
)

// MemoryAwarePolicy combines memory relaxation (Policy 3) and swap-pressure
// throttling (Policy 4) with an underlying compute policy (burst_freeze or
// dynamic_priority). The tick order is:
//  1. Memory limit relaxation — decide LP's memory budget
//  2. Swap-pressure throttling — decide if LP needs compute throttling due to swap
//  3. If swap override is active, use swap-based compute decisions;
//     otherwise delegate to the underlying compute policy
type MemoryAwarePolicy struct {
	memRelax      *MemoryRelaxationPolicy
	swapThrottle  *SwapThrottlingPolicy
	computePolicy SchedulerPolicy
}

func (p *MemoryAwarePolicy) Name() string {
	return "memory_aware"
}

func (p *MemoryAwarePolicy) Init(config SchedulerConfig) error {
	// Initialize memory relaxation sub-policy
	p.memRelax = &MemoryRelaxationPolicy{}
	if err := p.memRelax.Init(config); err != nil {
		return fmt.Errorf("memory_aware: memory_relaxation init: %w", err)
	}

	// Initialize swap throttling sub-policy
	p.swapThrottle = &SwapThrottlingPolicy{}
	if err := p.swapThrottle.Init(config); err != nil {
		return fmt.Errorf("memory_aware: swap_throttling init: %w", err)
	}

	// Initialize underlying compute policy
	switch config.ComputePolicy {
	case "burst_freeze":
		p.computePolicy = &BurstFreezePolicy{}
	case "dynamic_priority":
		p.computePolicy = &DynamicPriorityPolicy{}
	default:
		return fmt.Errorf("memory_aware: unknown compute policy %q (options: burst_freeze, dynamic_priority)",
			config.ComputePolicy)
	}
	if err := p.computePolicy.Init(config); err != nil {
		return fmt.Errorf("memory_aware: compute policy %q init: %w", config.ComputePolicy, err)
	}

	fmt.Printf("[%s] memory_aware: combined policy initialized (compute_base=%s)\n",
		ts(), p.computePolicy.Name())
	return nil
}

func (p *MemoryAwarePolicy) Tick(ctx SchedulerContext) []SchedulingAction {
	var actions []SchedulingAction

	// Step 1: Memory limit relaxation — always runs
	memActions := p.memRelax.TickMemoryRelaxation(ctx)
	actions = append(actions, memActions...)

	// Step 2: Swap-pressure throttling
	swapActions := p.swapThrottle.tickSwap(ctx)

	// Step 3: If swap override is active, use swap-based compute decisions;
	// otherwise delegate to the underlying compute policy
	if p.swapThrottle.HasSwapOverride(ctx) {
		actions = append(actions, swapActions...)
	} else {
		// Swap pressure is low — delegate to compute policy
		computeActions := p.computePolicy.Tick(ctx)
		actions = append(actions, computeActions...)
	}

	return actions
}
