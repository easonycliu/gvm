package main

import (
	"testing"
)

const (
	adaptiveGPUTotal = int64(24 * gb)
	_128MB           = int64(128 * mb)
	_100MB           = int64(100 * mb)
	_512MB           = int64(512 * mb)
	_GB              = int64(gb)
	_MB              = int64(mb)
)

func adaptiveConfig() SchedulerConfig {
	return SchedulerConfig{
		PollInterval:                  50,
		AdaptiveHPIdleThreshold:       2,
		AdaptiveHPBusyThreshold:       8,
		AdaptiveHPMinCacheFrac:        0.20,  // ~4.8 GB on 24 GB
		AdaptiveHPMaxCacheFrac:        0.80,  // ~19.2 GB on 24 GB
		AdaptiveLPMinReservationBytes: _512MB, // 512 MB
		AdaptiveRampStepBytes:         _128MB, // 128 MB
		AdaptiveSafetyMarginBytes:     _100MB, // 100 MB
		ComputePolicy:                 "dynamic_priority",

		// Required for sub-policy init
		DynLightThreshold:   2,
		DynHeavyThreshold:   8,
		DynExtremeThreshold: 20,
		DynIdlePriority:     0,
		DynLightPriority:    4,
		DynHeavyPriority:    12,
		SwapFreezeRatio:     0.3,
		SwapThrottleRatio:   0.1,
		SwapClearRatio:      0.02,
		SwapCooldownTicks:   4,
	}
}

func adaptiveCtx(hpLoad int64, hp []ProcessInfo, lp []ProcessInfo) SchedulerContext {
	return SchedulerContext{
		HPPending:   hpLoad,
		HPProcesses: hp,
		LPProcesses: lp,
		GPUTotalMem: map[int]int64{0: adaptiveGPUTotal},
	}
}

func findAction(actions []SchedulingAction, pid int, action string) *SchedulingAction {
	for i, a := range actions {
		if a.PID == pid && a.Action == action {
			return &actions[i]
		}
	}
	return nil
}

// Test: HP idle → HP cache shrinks
func TestAdaptive_HPIdle_ShrinkCache(t *testing.T) {
	p := &AdaptiveMemoryPolicy{}
	if err := p.Init(adaptiveConfig()); err != nil {
		t.Fatal(err)
	}

	hpLimitHigh := int64(16 * _GB) // 16 GB current cache
	hpMemCurrent := int64(4 * _GB) // only using 4 GB
	hp := []ProcessInfo{hpMem(100, 0, hpMemCurrent, hpLimitHigh)}
	lp := []ProcessInfo{lpMem(200, 0, 2*_GB, 6*_GB, _512MB, 0)}

	// First tick: init
	ctx := adaptiveCtx(0, hp, lp) // load=0 → idle
	p.Tick(ctx)

	// Second tick: should shrink HP cache (ramp down by 128MB)
	actions := p.Tick(ctx)
	hpAction := findAction(actions, 100, "set_memory_limit_high")
	if hpAction == nil {
		t.Fatal("expected HP set_memory_limit_high action")
	}
	if hpAction.MemoryLimitHigh >= hpLimitHigh {
		t.Errorf("HP limit.high should decrease: got %d, was %d", hpAction.MemoryLimitHigh, hpLimitHigh)
	}
}

// Test: HP busy → HP cache expands
func TestAdaptive_HPBusy_ExpandCache(t *testing.T) {
	p := &AdaptiveMemoryPolicy{}
	if err := p.Init(adaptiveConfig()); err != nil {
		t.Fatal(err)
	}

	hpLimitHigh := int64(8 * _GB)
	hpMemCurrent := int64(7 * _GB) // near limit
	hp := []ProcessInfo{hpMem(100, 0, hpMemCurrent, hpLimitHigh)}
	lp := []ProcessInfo{lpMem(200, 0, 3*_GB, 8*_GB, 2*_GB, 0)}

	// First tick: init
	ctx := adaptiveCtx(15, hp, lp) // load=15 → busy
	p.Tick(ctx)

	// Second tick: HP memory grew
	hp[0].MemoryCurrent = 8 * _GB
	ctx = adaptiveCtx(15, hp, lp)
	actions := p.Tick(ctx)

	hpAction := findAction(actions, 100, "set_memory_limit_high")
	if hpAction == nil {
		t.Fatal("expected HP set_memory_limit_high action")
	}
	if hpAction.MemoryLimitHigh <= hpLimitHigh {
		t.Errorf("HP limit.high should increase: got %d, was %d", hpAction.MemoryLimitHigh, hpLimitHigh)
	}
}

// Test: HP idle → LP gets more reservation (limit.low increases)
func TestAdaptive_HPIdle_LPReservationIncreases(t *testing.T) {
	p := &AdaptiveMemoryPolicy{}
	if err := p.Init(adaptiveConfig()); err != nil {
		t.Fatal(err)
	}

	hpLimitHigh := int64(12 * _GB)
	hpMemCurrent := int64(4 * _GB)
	lpLimitLow := int64(_512MB)
	hp := []ProcessInfo{hpMem(100, 0, hpMemCurrent, hpLimitHigh)}
	lp := []ProcessInfo{lpMem(200, 0, 2*_GB, 6*_GB, lpLimitLow, 0)}

	// First tick: init
	ctx := adaptiveCtx(0, hp, lp)
	p.Tick(ctx)

	// Second tick: HP idle → LP reservation should increase
	actions := p.Tick(ctx)
	lpLowAction := findAction(actions, 200, "set_memory_limit_low")
	if lpLowAction == nil {
		t.Fatal("expected LP set_memory_limit_low action")
	}
	if lpLowAction.MemoryLimitLow <= lpLimitLow {
		t.Errorf("LP limit.low should increase: got %d, was %d", lpLowAction.MemoryLimitLow, lpLimitLow)
	}
}

// Test: HP busy → LP reservation decreases (soft reclaim)
func TestAdaptive_HPBusy_LPReservationDecreases(t *testing.T) {
	p := &AdaptiveMemoryPolicy{}
	if err := p.Init(adaptiveConfig()); err != nil {
		t.Fatal(err)
	}

	hp := []ProcessInfo{hpMem(100, 0, 8*_GB, 12*_GB)}
	lp := []ProcessInfo{lpMem(200, 0, 4*_GB, 8*_GB, 4*_GB, 0)}

	// First tick: init with moderate load
	ctx := adaptiveCtx(5, hp, lp)
	p.Tick(ctx)

	// Second tick: HP gets busy with memory growth
	hp[0].MemoryCurrent = 10 * _GB
	ctx = adaptiveCtx(15, hp, lp)
	actions := p.Tick(ctx)

	lpLowAction := findAction(actions, 200, "set_memory_limit_low")
	if lpLowAction == nil {
		t.Fatal("expected LP set_memory_limit_low action when HP busy")
	}
	if lpLowAction.MemoryLimitLow >= 4*_GB {
		t.Errorf("LP limit.low should decrease: got %d, was %d", lpLowAction.MemoryLimitLow, 4*_GB)
	}
}

// Test: LP limit.low never drops below minimum reservation
func TestAdaptive_LPMinReservationFloor(t *testing.T) {
	p := &AdaptiveMemoryPolicy{}
	if err := p.Init(adaptiveConfig()); err != nil {
		t.Fatal(err)
	}

	hp := []ProcessInfo{hpMem(100, 0, 20*_GB, 22*_GB)}
	lp := []ProcessInfo{lpMem(200, 0, 1*_GB, 2*_GB, 600*_MB, 0)} // just above 512MB min

	// Init
	ctx := adaptiveCtx(0, hp, lp)
	p.Tick(ctx)

	// Busy HP — try to reclaim from LP
	hp[0].MemoryCurrent = 21 * _GB
	ctx = adaptiveCtx(20, hp, lp)
	actions := p.Tick(ctx)

	lpLowAction := findAction(actions, 200, "set_memory_limit_low")
	if lpLowAction != nil && lpLowAction.MemoryLimitLow < _512MB {
		t.Errorf("LP limit.low must not drop below lpMinReservation (%d): got %d",
			_512MB, lpLowAction.MemoryLimitLow)
	}
}

// Test: HP cache respects max fraction
func TestAdaptive_HPMaxCacheCap(t *testing.T) {
	p := &AdaptiveMemoryPolicy{}
	if err := p.Init(adaptiveConfig()); err != nil {
		t.Fatal(err)
	}

	var gpuMem int64 = adaptiveGPUTotal
	hpMaxCache := int64(float64(gpuMem) * 0.80) // ~19.2 GB
	hp := []ProcessInfo{hpMem(100, 0, 18*_GB, hpMaxCache)}
	lp := []ProcessInfo{lpMem(200, 0, 1*_GB, 2*_GB, _512MB, 0)}

	ctx := adaptiveCtx(20, hp, lp) // very busy
	p.Tick(ctx)

	// Second tick
	hp[0].MemoryCurrent = 19 * _GB
	ctx = adaptiveCtx(25, hp, lp)
	actions := p.Tick(ctx)

	hpAction := findAction(actions, 100, "set_memory_limit_high")
	if hpAction != nil && hpAction.MemoryLimitHigh > hpMaxCache {
		t.Errorf("HP limit.high should not exceed max cache (%d): got %d",
			hpMaxCache, hpAction.MemoryLimitHigh)
	}
}

// Test: HP cache respects min fraction
func TestAdaptive_HPMinCacheFloor(t *testing.T) {
	p := &AdaptiveMemoryPolicy{}
	if err := p.Init(adaptiveConfig()); err != nil {
		t.Fatal(err)
	}

	var gpuMem int64 = adaptiveGPUTotal
	hpMinCache := int64(float64(gpuMem) * 0.20) // ~4.8 GB
	hp := []ProcessInfo{hpMem(100, 0, 1*_GB, hpMinCache+_128MB)} // barely above min
	lp := []ProcessInfo{lpMem(200, 0, 2*_GB, 6*_GB, _512MB, 0)}

	// Init at idle
	ctx := adaptiveCtx(0, hp, lp)
	p.Tick(ctx)

	// Multiple idle ticks to try to shrink below min
	for i := 0; i < 20; i++ {
		actions := p.Tick(ctx)
		hpAction := findAction(actions, 100, "set_memory_limit_high")
		if hpAction != nil && hpAction.MemoryLimitHigh < hpMinCache {
			t.Errorf("HP limit.high should not drop below min cache (%d): got %d at tick %d",
				hpMinCache, hpAction.MemoryLimitHigh, i)
			break
		}
	}
}

// Test: Bidirectional flow — HP idle shrinks cache, LP gets reservation;
// then HP busy, LP reservation reduced, HP cache expands
func TestAdaptive_BidirectionalFlow(t *testing.T) {
	p := &AdaptiveMemoryPolicy{}
	if err := p.Init(adaptiveConfig()); err != nil {
		t.Fatal(err)
	}

	hp := []ProcessInfo{hpMem(100, 0, 6*_GB, 16*_GB)}
	lp := []ProcessInfo{lpMem(200, 0, 2*_GB, 4*_GB, _512MB, 0)}

	// Phase 1: HP idle — run several ticks
	for i := 0; i < 10; i++ {
		ctx := adaptiveCtx(0, hp, lp)
		p.Tick(ctx)
	}

	// Verify: HP cache should have shrunk
	hpKey := lpKey{PID: 100, GPUIndex: 0}
	lpK := lpKey{PID: 200, GPUIndex: 0}
	hpHighAfterIdle := p.hpCurrentHigh[hpKey]
	lpLowAfterIdle := p.lpCurrentLow[lpK]

	if hpHighAfterIdle >= 16*_GB {
		t.Errorf("HP cache should have shrunk after idle phase: still %d", hpHighAfterIdle)
	}
	if lpLowAfterIdle <= _512MB {
		t.Errorf("LP reservation should have increased after idle phase: still %d", lpLowAfterIdle)
	}
	t.Logf("After idle: HP.high=%s LP.low=%s", fmtMB(hpHighAfterIdle), fmtMB(lpLowAfterIdle))

	// Phase 2: HP gets busy — run several ticks
	for i := 0; i < 10; i++ {
		hp[0].MemoryCurrent += _128MB // growing
		ctx := adaptiveCtx(15, hp, lp)
		p.Tick(ctx)
	}

	hpHighAfterBusy := p.hpCurrentHigh[hpKey]
	lpLowAfterBusy := p.lpCurrentLow[lpK]

	if hpHighAfterBusy <= hpHighAfterIdle {
		t.Errorf("HP cache should have expanded after busy phase: %d -> %d", hpHighAfterIdle, hpHighAfterBusy)
	}
	if lpLowAfterBusy >= lpLowAfterIdle {
		t.Errorf("LP reservation should have decreased after busy phase: %d -> %d", lpLowAfterIdle, lpLowAfterBusy)
	}
	t.Logf("After busy: HP.high=%s LP.low=%s", fmtMB(hpHighAfterBusy), fmtMB(lpLowAfterBusy))
}

// Test: Compute policy still works through adaptive_memory
func TestAdaptive_ComputePolicyDelegation(t *testing.T) {
	p := &AdaptiveMemoryPolicy{}
	if err := p.Init(adaptiveConfig()); err != nil {
		t.Fatal(err)
	}

	hp := []ProcessInfo{hpMem(100, 0, 8*_GB, 16*_GB)}
	lp := []ProcessInfo{lpMem(200, 0, 2*_GB, 6*_GB, _512MB, 0)}

	// Init tick
	ctx := adaptiveCtx(0, hp, lp)
	p.Tick(ctx)

	// High HP load → dynamic_priority should freeze LP
	ctx = adaptiveCtx(25, hp, lp)
	actions := p.Tick(ctx)

	freezeAction := findAction(actions, 200, "freeze")
	if freezeAction == nil {
		t.Error("expected compute policy to freeze LP at extreme HP load")
	}
}
