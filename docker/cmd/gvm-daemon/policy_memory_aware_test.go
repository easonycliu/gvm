package main

import (
	"testing"
)

// Helper to create an LP ProcessInfo with two-level memory stats.
func lpMem(pid, gpu int, memCurrent, limitHigh, limitLow, swapCurrent int64) ProcessInfo {
	return ProcessInfo{
		PID:               pid,
		GPUIndex:          gpu,
		Role:              "lp",
		MemoryCurrent:     memCurrent,
		MemoryLimitHigh:   limitHigh,
		MemoryLimitLow:    limitLow,
		MemorySwapCurrent: swapCurrent,
	}
}

func hpMem(pid, gpu int, memCurrent, limitHigh int64) ProcessInfo {
	return ProcessInfo{
		PID:             pid,
		GPUIndex:        gpu,
		Role:            "hp",
		MemoryCurrent:   memCurrent,
		MemoryLimitHigh: limitHigh,
	}
}

const (
	mb  = 1024 * 1024
	gb  = 1024 * 1024 * 1024
	gpu = 24 * gb // 24 GB total
)

func defaultMemConfig() SchedulerConfig {
	return DefaultSchedulerConfig()
}

// --- MemoryRelaxationPolicy tests ---

func newMemRelaxPolicy() *MemoryRelaxationPolicy {
	p := &MemoryRelaxationPolicy{}
	p.Init(defaultMemConfig())
	return p
}

func TestMemRelax_LendWhenHPIdle(t *testing.T) {
	p := newMemRelaxPolicy()

	// LP: limit.high=6GB, limit.low=4GB (reservation)
	hps := []ProcessInfo{hpMem(1000, 0, 2*gb, 10*gb)}
	lps := []ProcessInfo{lpMem(2000, 0, 4*gb, 6*gb, 4*gb, 0)}

	ctx := SchedulerContext{
		HPPending:   0, // idle
		HPProcesses: hps,
		LPProcesses: lps,
		GPUTotalMem: map[int]int64{0: gpu},
	}

	actions := p.Tick(ctx)

	// HP slack = 10GB - 2GB = 8GB, safety = 50MB
	// Borrowable = (8GB - 50MB) * 0.7 ≈ 5.56GB
	// New limit.high = 6GB + 5.56GB ≈ 11.56GB
	if len(actions) != 1 {
		t.Fatalf("expected 1 set_memory_limit_high action, got %d", len(actions))
	}
	if actions[0].Action != "set_memory_limit_high" {
		t.Fatalf("expected set_memory_limit_high, got %s", actions[0].Action)
	}
	if actions[0].MemoryLimitHigh <= 6*gb {
		t.Fatalf("expected limit.high > 6GB (base), got %d", actions[0].MemoryLimitHigh)
	}
	if actions[0].MemoryLimitHigh > 12*gb {
		t.Fatalf("expected limit.high < 12GB, got %d", actions[0].MemoryLimitHigh)
	}
}

func TestMemRelax_ReclaimWhenHPActive(t *testing.T) {
	p := newMemRelaxPolicy()

	hps := []ProcessInfo{hpMem(1000, 0, 2*gb, 10*gb)}
	lps := []ProcessInfo{lpMem(2000, 0, 4*gb, 6*gb, 4*gb, 0)}

	// First tick: HP idle, lend memory
	ctx := SchedulerContext{
		HPPending:   0,
		HPProcesses: hps,
		LPProcesses: lps,
		GPUTotalMem: map[int]int64{0: gpu},
	}
	p.Tick(ctx)

	// Capture the lent limit.high after first tick
	key := lpKey{PID: 2000, GPUIndex: 0}
	lentHigh := p.lpCurrentHigh[key]

	// Second tick: HP active (pending > 0)
	ctx.HPPending = 5
	actions := p.Tick(ctx)

	// Should reclaim: ramp down by 100MB from the lent limit.high
	if len(actions) != 1 {
		t.Fatalf("expected 1 reclaim action, got %d", len(actions))
	}
	if actions[0].Action != "set_memory_limit_high" {
		t.Fatalf("expected set_memory_limit_high, got %s", actions[0].Action)
	}
	if p.lpCurrentHigh[key] >= lentHigh {
		t.Fatalf("expected limit.high to decrease from lent %d, got %d", lentHigh, p.lpCurrentHigh[key])
	}
	expectedHigh := lentHigh - 100*mb
	if abs64(p.lpCurrentHigh[key]-expectedHigh) > mb {
		t.Fatalf("expected limit.high ~%d (lent - 100MB), got %d", expectedHigh, p.lpCurrentHigh[key])
	}
}

func TestMemRelax_NeverBelowFloor(t *testing.T) {
	p := newMemRelaxPolicy()

	// LP: limit.high=6GB, limit.low=4GB (floor = limit.low)
	hps := []ProcessInfo{hpMem(1000, 0, 9*gb, 10*gb)}
	lps := []ProcessInfo{lpMem(2000, 0, 4*gb, 6*gb, 4*gb, 0)}

	ctx := SchedulerContext{
		HPPending:   10,
		HPProcesses: hps,
		LPProcesses: lps,
		GPUTotalMem: map[int]int64{0: gpu},
	}

	p.Tick(ctx)

	// Many ticks of reclaim should never go below floor (limit.low=4GB)
	for i := 0; i < 30; i++ {
		p.Tick(ctx)
	}

	key := lpKey{PID: 2000, GPUIndex: 0}
	floor := p.lpBaseLimitLow[key]
	if p.lpCurrentHigh[key] < floor {
		t.Fatalf("limit.high %d dropped below floor (limit.low) %d", p.lpCurrentHigh[key], floor)
	}
}

func TestMemRelax_FloorFallsBackToBaseHigh(t *testing.T) {
	p := newMemRelaxPolicy()

	// LP: limit.high=6GB, limit.low=0 (no reservation → floor = base limit.high)
	hps := []ProcessInfo{hpMem(1000, 0, 9*gb, 10*gb)}
	lps := []ProcessInfo{lpMem(2000, 0, 4*gb, 6*gb, 0, 0)}

	ctx := SchedulerContext{
		HPPending:   10,
		HPProcesses: hps,
		LPProcesses: lps,
		GPUTotalMem: map[int]int64{0: gpu},
	}

	p.Tick(ctx)
	for i := 0; i < 30; i++ {
		p.Tick(ctx)
	}

	key := lpKey{PID: 2000, GPUIndex: 0}
	baseHigh := p.lpBaseLimitHigh[key]
	if p.lpCurrentHigh[key] < baseHigh {
		t.Fatalf("limit.high %d dropped below base_high %d (no reservation)", p.lpCurrentHigh[key], baseHigh)
	}
}

func TestMemRelax_SkipSmallChanges(t *testing.T) {
	p := newMemRelaxPolicy()

	hps := []ProcessInfo{hpMem(1000, 0, 2*gb, 10*gb)}
	lps := []ProcessInfo{lpMem(2000, 0, 4*gb, 6*gb, 4*gb, 0)}

	ctx := SchedulerContext{
		HPPending:   0,
		HPProcesses: hps,
		LPProcesses: lps,
		GPUTotalMem: map[int]int64{0: gpu},
	}

	// First tick: lend
	p.Tick(ctx)

	// Same context again — no meaningful change, should skip write
	actions := p.Tick(ctx)
	if len(actions) != 0 {
		t.Fatalf("expected 0 actions (no meaningful change), got %d", len(actions))
	}
}

func TestMemRelax_PredictiveReclaim(t *testing.T) {
	p := newMemRelaxPolicy()

	hps := []ProcessInfo{hpMem(1000, 0, 2*gb, 10*gb)}
	lps := []ProcessInfo{lpMem(2000, 0, 4*gb, 6*gb, 4*gb, 0)}

	ctx := SchedulerContext{
		HPPending:   0,
		HPProcesses: hps,
		LPProcesses: lps,
		GPUTotalMem: map[int]int64{0: gpu},
	}
	p.Tick(ctx)

	// Second tick: HP memory jumped by 200MB (> fast_growth_threshold=50MB)
	hps2 := []ProcessInfo{hpMem(1000, 0, 2*gb+200*mb, 10*gb)}
	ctx2 := SchedulerContext{
		HPPending:   0,
		HPProcesses: hps2,
		LPProcesses: lps,
		GPUTotalMem: map[int]int64{0: gpu},
	}

	actions := p.Tick(ctx2)
	if len(actions) != 1 {
		t.Fatalf("expected 1 action for predictive reclaim, got %d", len(actions))
	}
	if actions[0].Action != "set_memory_limit_high" {
		t.Fatalf("expected set_memory_limit_high, got %s", actions[0].Action)
	}
}

// --- SwapThrottlingPolicy tests ---

func newSwapPolicy() *SwapThrottlingPolicy {
	p := &SwapThrottlingPolicy{}
	p.Init(defaultMemConfig())
	return p
}

func TestSwap_FreezeOnHighSwap(t *testing.T) {
	p := newSwapPolicy()

	// swap_ratio = 2GB / 6GB = 0.33 > freeze_ratio(0.3)
	lps := []ProcessInfo{lpMem(2000, 0, 4*gb, 6*gb, 4*gb, 2*gb)}
	ctx := SchedulerContext{LPProcesses: lps}

	actions := p.Tick(ctx)

	hasFreezeAction := false
	for _, a := range actions {
		if a.Action == "freeze" && a.PID == 2000 {
			hasFreezeAction = true
		}
	}
	if !hasFreezeAction {
		t.Fatal("expected freeze action for high swap ratio")
	}
}

func TestSwap_CooldownCountdown(t *testing.T) {
	p := newSwapPolicy()

	lps := []ProcessInfo{lpMem(2000, 0, 4*gb, 6*gb, 4*gb, 2*gb)}
	ctx := SchedulerContext{LPProcesses: lps}
	p.Tick(ctx)

	key := lpKey{PID: 2000, GPUIndex: 0}
	if p.cooldownRemaining[key] != p.cooldownTicks {
		t.Fatalf("expected cooldown=%d, got %d", p.cooldownTicks, p.cooldownRemaining[key])
	}

	for i := p.cooldownTicks - 1; i > 0; i-- {
		actions := p.Tick(ctx)
		if p.cooldownRemaining[key] != i {
			t.Fatalf("expected cooldown=%d, got %d", i, p.cooldownRemaining[key])
		}
		for _, a := range actions {
			if a.Action == "freeze" || a.Action == "set_priority" {
				t.Fatalf("unexpected action during cooldown: %s", a.Action)
			}
		}
	}

	actions := p.Tick(ctx)
	hasUnfreeze := false
	for _, a := range actions {
		if a.Action == "unfreeze" {
			hasUnfreeze = true
		}
	}
	if !hasUnfreeze {
		t.Fatal("expected unfreeze when cooldown expires")
	}
}

func TestSwap_ThrottlePriority(t *testing.T) {
	p := newSwapPolicy()

	// swap_ratio = 1GB / 6GB ≈ 0.167, between throttle(0.1) and freeze(0.3)
	lps := []ProcessInfo{lpMem(2000, 0, 4*gb, 6*gb, 4*gb, 1*gb)}
	ctx := SchedulerContext{LPProcesses: lps}

	actions := p.Tick(ctx)

	hasPriority := false
	for _, a := range actions {
		if a.Action == "set_priority" {
			hasPriority = true
			if a.Priority < 4 || a.Priority > 12 {
				t.Fatalf("expected priority in [4,12], got %d", a.Priority)
			}
		}
	}
	if !hasPriority {
		t.Fatal("expected set_priority for moderate swap pressure")
	}
}

func TestSwap_NoActionBelowClear(t *testing.T) {
	p := newSwapPolicy()

	// swap_ratio = 100KB / 6GB ≈ 0.00002 < clear_ratio(0.02)
	lps := []ProcessInfo{lpMem(2000, 0, 4*gb, 6*gb, 4*gb, 100*1024)}
	ctx := SchedulerContext{LPProcesses: lps}

	actions := p.Tick(ctx)
	if len(actions) != 0 {
		t.Fatalf("expected 0 actions below clear ratio, got %d", len(actions))
	}
}

func TestSwap_HasSwapOverride(t *testing.T) {
	p := newSwapPolicy()

	lps := []ProcessInfo{lpMem(2000, 0, 4*gb, 6*gb, 4*gb, 2*gb)}
	ctx := SchedulerContext{LPProcesses: lps}

	if p.HasSwapOverride(ctx) {
		t.Fatal("expected no swap override before freeze")
	}

	p.Tick(ctx)
	if !p.HasSwapOverride(ctx) {
		t.Fatal("expected swap override after freeze")
	}
}

// --- MemoryAwarePolicy tests ---

func newMemAwarePolicy(computeBase string) *MemoryAwarePolicy {
	p := &MemoryAwarePolicy{}
	cfg := defaultMemConfig()
	cfg.ComputePolicy = computeBase
	if err := p.Init(cfg); err != nil {
		panic(err)
	}
	return p
}

func TestMemAware_MemRelaxAlwaysRuns(t *testing.T) {
	p := newMemAwarePolicy("dynamic_priority")

	hps := []ProcessInfo{hpMem(1000, 0, 2*gb, 10*gb)}
	lps := []ProcessInfo{lpMem(2000, 0, 4*gb, 6*gb, 4*gb, 0)}

	ctx := SchedulerContext{
		HPPending:   0,
		HPProcesses: hps,
		LPProcesses: lps,
		GPUTotalMem: map[int]int64{0: gpu},
	}

	actions := p.Tick(ctx)

	hasMemLimitHigh := false
	for _, a := range actions {
		if a.Action == "set_memory_limit_high" {
			hasMemLimitHigh = true
		}
	}
	if !hasMemLimitHigh {
		t.Fatal("expected set_memory_limit_high from memory relaxation")
	}
}

func TestMemAware_ComputePolicyRunsWhenNoSwap(t *testing.T) {
	p := newMemAwarePolicy("dynamic_priority")

	hps := []ProcessInfo{hpMem(1000, 0, 2*gb, 10*gb)}
	lps := []ProcessInfo{lpMem(2000, 0, 4*gb, 6*gb, 4*gb, 0)}

	ctx := SchedulerContext{
		HPPending:   0,
		HPProcesses: hps,
		LPProcesses: lps,
		GPUTotalMem: map[int]int64{0: gpu},
	}

	actions := p.Tick(ctx)

	hasPriority := false
	for _, a := range actions {
		if a.Action == "set_priority" {
			hasPriority = true
		}
	}
	if !hasPriority {
		t.Fatal("expected compute policy (set_priority) to run when no swap pressure")
	}
}

func TestMemAware_SwapOverridesCompute(t *testing.T) {
	p := newMemAwarePolicy("dynamic_priority")

	hps := []ProcessInfo{hpMem(1000, 0, 2*gb, 10*gb)}
	// High swap pressure: swap_ratio = 2GB/6GB = 0.33 > 0.3
	lps := []ProcessInfo{lpMem(2000, 0, 4*gb, 6*gb, 4*gb, 2*gb)}

	ctx := SchedulerContext{
		HPPending:   0,
		HPProcesses: hps,
		LPProcesses: lps,
		GPUTotalMem: map[int]int64{0: gpu},
	}

	actions := p.Tick(ctx)

	hasFreeze := false
	hasPriority := false
	for _, a := range actions {
		if a.Action == "freeze" {
			hasFreeze = true
		}
		if a.Action == "set_priority" {
			hasPriority = true
		}
	}
	if !hasFreeze {
		t.Fatal("expected swap-based freeze to override compute policy")
	}
	if hasPriority {
		t.Fatal("expected compute policy NOT to run when swap override active")
	}
}

func TestMemAware_InitInvalidComputePolicy(t *testing.T) {
	p := &MemoryAwarePolicy{}
	cfg := defaultMemConfig()
	cfg.ComputePolicy = "nonexistent"
	if err := p.Init(cfg); err == nil {
		t.Fatal("expected error for invalid compute policy")
	}
}

// --- SelectPolicy tests for new policies ---

func TestSelectPolicy_MemoryPolicies(t *testing.T) {
	if p := SelectPolicy("memory_relaxation"); p == nil || p.Name() != "memory_relaxation" {
		t.Fatal("expected memory_relaxation policy")
	}
	if p := SelectPolicy("swap_throttling"); p == nil || p.Name() != "swap_throttling" {
		t.Fatal("expected swap_throttling policy")
	}
	if p := SelectPolicy("memory_aware"); p == nil || p.Name() != "memory_aware" {
		t.Fatal("expected memory_aware policy")
	}
}
