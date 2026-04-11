package main

import (
	"testing"
)

func lp(pid, gpu int) ProcessInfo {
	return ProcessInfo{PID: pid, GPUIndex: gpu, Role: "lp"}
}

// mkCtx builds a minimal SchedulerContext for testing.
func mkCtx(hpPending int64, lps []ProcessInfo) SchedulerContext {
	return SchedulerContext{HPPending: hpPending, LPProcesses: lps}
}

// --- BurstFreezePolicy tests ---

func newBurstPolicy(high, low int64) *BurstFreezePolicy {
	p := &BurstFreezePolicy{}
	p.Init(SchedulerConfig{
		BurstHighThreshold: high,
		BurstLowThreshold:  low,
	})
	return p
}

func TestBurstFreeze_InitValidation(t *testing.T) {
	p := &BurstFreezePolicy{}
	err := p.Init(SchedulerConfig{BurstHighThreshold: 3, BurstLowThreshold: 5})
	if err == nil {
		t.Fatal("expected error when low >= high threshold")
	}
	err = p.Init(SchedulerConfig{BurstHighThreshold: 5, BurstLowThreshold: 5})
	if err == nil {
		t.Fatal("expected error when low == high threshold")
	}
}

func TestBurstFreeze_IdleStaysIdle(t *testing.T) {
	p := newBurstPolicy(5, 1)
	lps := []ProcessInfo{lp(100, 0)}

	actions := p.Tick(mkCtx(3, lps)) // below high threshold
	if len(actions) != 0 {
		t.Fatalf("expected 0 actions in idle, got %d", len(actions))
	}
	if p.state != burstStateIdle {
		t.Fatal("expected state to remain IDLE")
	}
}

func TestBurstFreeze_TransitionToContention(t *testing.T) {
	p := newBurstPolicy(5, 1)
	lps := []ProcessInfo{lp(100, 0), lp(200, 0)}

	actions := p.Tick(mkCtx(6, lps)) // above high threshold
	if p.state != burstStateContention {
		t.Fatal("expected state CONTENTION")
	}
	if len(actions) != 2 {
		t.Fatalf("expected 2 freeze actions, got %d", len(actions))
	}
	for _, a := range actions {
		if a.Action != "freeze" {
			t.Fatalf("expected freeze action, got %s", a.Action)
		}
	}
}

func TestBurstFreeze_ContentionStaysContention(t *testing.T) {
	p := newBurstPolicy(5, 1)
	lps := []ProcessInfo{lp(100, 0)}

	p.Tick(mkCtx(6, lps)) // -> CONTENTION

	// Still above low threshold, should stay in CONTENTION with no new actions
	actions := p.Tick(mkCtx(3, lps))
	if p.state != burstStateContention {
		t.Fatal("expected state to remain CONTENTION")
	}
	if len(actions) != 0 {
		t.Fatalf("expected 0 actions (already frozen), got %d", len(actions))
	}
}

func TestBurstFreeze_TransitionBackToIdle(t *testing.T) {
	p := newBurstPolicy(5, 1)
	lps := []ProcessInfo{lp(100, 0)}

	p.Tick(mkCtx(6, lps))  // -> CONTENTION
	actions := p.Tick(mkCtx(0, lps)) // below low threshold -> IDLE

	if p.state != burstStateIdle {
		t.Fatal("expected state IDLE")
	}
	if len(actions) != 1 {
		t.Fatalf("expected 1 unfreeze action, got %d", len(actions))
	}
	if actions[0].Action != "unfreeze" {
		t.Fatalf("expected unfreeze, got %s", actions[0].Action)
	}
}

func TestBurstFreeze_Hysteresis(t *testing.T) {
	p := newBurstPolicy(5, 1)
	lps := []ProcessInfo{lp(100, 0)}

	// Start idle, go to contention
	p.Tick(mkCtx(6, lps))
	if p.state != burstStateContention {
		t.Fatal("expected CONTENTION")
	}

	// Pending drops to 3 (below high but above low) — should NOT unfreeze
	actions := p.Tick(mkCtx(3, lps))
	if p.state != burstStateContention {
		t.Fatal("expected to stay in CONTENTION due to hysteresis")
	}
	if len(actions) != 0 {
		t.Fatalf("expected no actions during hysteresis zone, got %d", len(actions))
	}

	// Pending drops to 0 (below low) — NOW unfreeze
	actions = p.Tick(mkCtx(0, lps))
	if p.state != burstStateIdle {
		t.Fatal("expected IDLE after dropping below low threshold")
	}
	if len(actions) != 1 || actions[0].Action != "unfreeze" {
		t.Fatal("expected unfreeze action")
	}
}

func TestBurstFreeze_NoDuplicateFreeze(t *testing.T) {
	p := newBurstPolicy(5, 1)
	lps := []ProcessInfo{lp(100, 0)}

	actions1 := p.Tick(mkCtx(6, lps)) // -> CONTENTION, freeze
	if len(actions1) != 1 {
		t.Fatalf("expected 1 freeze action, got %d", len(actions1))
	}

	// Still in CONTENTION, same LP process — should NOT re-freeze
	actions2 := p.Tick(mkCtx(10, lps))
	if len(actions2) != 0 {
		t.Fatalf("expected 0 actions (already frozen), got %d", len(actions2))
	}
}

// --- DynamicPriorityPolicy tests ---

func newDynPolicy() *DynamicPriorityPolicy {
	p := &DynamicPriorityPolicy{}
	p.Init(DefaultSchedulerConfig())
	return p
}

func TestDynPriority_IdleState(t *testing.T) {
	p := newDynPolicy()
	lps := []ProcessInfo{lp(100, 0)}

	actions := p.Tick(mkCtx(0, lps))
	if len(actions) != 1 {
		t.Fatalf("expected 1 action (initial priority set), got %d", len(actions))
	}
	if actions[0].Action != "set_priority" || actions[0].Priority != 0 {
		t.Fatalf("expected set_priority to 0 (idle), got %s priority=%d", actions[0].Action, actions[0].Priority)
	}
}

func TestDynPriority_SkipRedundantWrite(t *testing.T) {
	p := newDynPolicy()
	lps := []ProcessInfo{lp(100, 0)}

	p.Tick(mkCtx(0, lps)) // set to idle priority (0)

	// Same state — should produce no actions
	actions := p.Tick(mkCtx(0, lps))
	if len(actions) != 0 {
		t.Fatalf("expected 0 actions (priority unchanged), got %d", len(actions))
	}
}

func TestDynPriority_LightLoad(t *testing.T) {
	p := newDynPolicy()
	lps := []ProcessInfo{lp(100, 0)}

	// Default: light threshold=2, light priority=4
	actions := p.Tick(mkCtx(3, lps)) // >= light, < heavy
	if len(actions) != 1 {
		t.Fatalf("expected 1 action, got %d", len(actions))
	}
	if actions[0].Priority != 4 {
		t.Fatalf("expected light priority 4, got %d", actions[0].Priority)
	}
}

func TestDynPriority_HeavyLoad(t *testing.T) {
	p := newDynPolicy()
	lps := []ProcessInfo{lp(100, 0)}

	// Default: heavy threshold=8, heavy priority=12
	actions := p.Tick(mkCtx(10, lps)) // >= heavy, < extreme
	if len(actions) != 1 {
		t.Fatalf("expected 1 action, got %d", len(actions))
	}
	if actions[0].Priority != 12 {
		t.Fatalf("expected heavy priority 12, got %d", actions[0].Priority)
	}
}

func TestDynPriority_ExtremeFreeze(t *testing.T) {
	p := newDynPolicy()
	lps := []ProcessInfo{lp(100, 0)}

	// Default: extreme threshold=20
	actions := p.Tick(mkCtx(25, lps))
	if len(actions) != 1 {
		t.Fatalf("expected 1 freeze action, got %d", len(actions))
	}
	if actions[0].Action != "freeze" {
		t.Fatalf("expected freeze, got %s", actions[0].Action)
	}
}

func TestDynPriority_UnfreezeAfterExtreme(t *testing.T) {
	p := newDynPolicy()
	lps := []ProcessInfo{lp(100, 0)}

	p.Tick(mkCtx(25, lps)) // extreme -> freeze

	// Drop back to idle
	actions := p.Tick(mkCtx(0, lps))
	// Should unfreeze AND set priority
	hasUnfreeze := false
	hasPriority := false
	for _, a := range actions {
		if a.Action == "unfreeze" {
			hasUnfreeze = true
		}
		if a.Action == "set_priority" {
			hasPriority = true
		}
	}
	if !hasUnfreeze {
		t.Fatal("expected unfreeze action after dropping from extreme")
	}
	if !hasPriority {
		t.Fatal("expected set_priority action after unfreeze")
	}
}

func TestDynPriority_GraduatedTransitions(t *testing.T) {
	p := newDynPolicy()
	lps := []ProcessInfo{lp(100, 0)}

	// idle -> light -> heavy -> extreme -> heavy -> idle
	p.Tick(mkCtx(0, lps))   // idle, priority=0
	actions := p.Tick(mkCtx(3, lps))  // light, priority=4
	if len(actions) != 1 || actions[0].Priority != 4 {
		t.Fatalf("idle->light: expected priority 4, got %v", actions)
	}

	actions = p.Tick(mkCtx(10, lps)) // heavy, priority=12
	if len(actions) != 1 || actions[0].Priority != 12 {
		t.Fatalf("light->heavy: expected priority 12, got %v", actions)
	}

	actions = p.Tick(mkCtx(25, lps)) // extreme -> freeze
	if len(actions) != 1 || actions[0].Action != "freeze" {
		t.Fatalf("heavy->extreme: expected freeze, got %v", actions)
	}

	actions = p.Tick(mkCtx(10, lps)) // back to heavy -> unfreeze + set priority 12
	unfreezeCount := 0
	priorityCount := 0
	for _, a := range actions {
		if a.Action == "unfreeze" {
			unfreezeCount++
		}
		if a.Action == "set_priority" && a.Priority == 12 {
			priorityCount++
		}
	}
	if unfreezeCount != 1 {
		t.Fatalf("extreme->heavy: expected 1 unfreeze, got %d", unfreezeCount)
	}
	if priorityCount != 1 {
		t.Fatalf("extreme->heavy: expected 1 set_priority(12), got %d", priorityCount)
	}

	actions = p.Tick(mkCtx(0, lps)) // back to idle, priority=0
	if len(actions) != 1 || actions[0].Priority != 0 {
		t.Fatalf("heavy->idle: expected priority 0, got %v", actions)
	}
}

func TestDynPriority_MultipleLP(t *testing.T) {
	p := newDynPolicy()
	lps := []ProcessInfo{lp(100, 0), lp(200, 0), lp(300, 1)}

	actions := p.Tick(mkCtx(10, lps)) // heavy
	if len(actions) != 3 {
		t.Fatalf("expected 3 actions for 3 LP processes, got %d", len(actions))
	}
	for _, a := range actions {
		if a.Action != "set_priority" || a.Priority != 12 {
			t.Fatalf("expected set_priority 12 for all, got %s priority=%d", a.Action, a.Priority)
		}
	}
}

// --- SchedulerConfig tests ---

func TestSelectPolicy_Valid(t *testing.T) {
	if p := SelectPolicy("burst_freeze"); p == nil || p.Name() != "burst_freeze" {
		t.Fatal("expected burst_freeze policy")
	}
	if p := SelectPolicy("dynamic_priority"); p == nil || p.Name() != "dynamic_priority" {
		t.Fatal("expected dynamic_priority policy")
	}
}

func TestSelectPolicy_Invalid(t *testing.T) {
	if p := SelectPolicy(""); p != nil {
		t.Fatal("expected nil for empty policy name")
	}
	if p := SelectPolicy("nonexistent"); p != nil {
		t.Fatal("expected nil for unknown policy name")
	}
}

func TestProcessRegistry_ThreadSafety(t *testing.T) {
	reg := NewProcessRegistry()
	hp := []ProcessInfo{{PID: 1, GPUIndex: 0, Role: "hp"}}
	lpList := []ProcessInfo{{PID: 2, GPUIndex: 0, Role: "lp"}}

	done := make(chan bool)
	go func() {
		for i := 0; i < 1000; i++ {
			reg.Update(hp, lpList)
		}
		done <- true
	}()
	go func() {
		for i := 0; i < 1000; i++ {
			reg.Get()
		}
		done <- true
	}()
	<-done
	<-done
}
