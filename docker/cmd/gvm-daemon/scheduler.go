package main

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/easonycliu/GCgroupProject/docker/pkg/gvm"
)

// SchedulerPolicy defines the interface for dynamic GPU scheduling policies.
type SchedulerPolicy interface {
	Name() string
	Init(config SchedulerConfig) error
	// Tick is called every poll interval with the full scheduler context.
	// Returns actions to apply.
	Tick(ctx SchedulerContext) []SchedulingAction
}

// ProcessInfo holds runtime info about a GPU process tracked by the scheduler.
type ProcessInfo struct {
	PID           int
	GPUIndex      int
	ContainerID   string
	ContainerName string
	Role          string // "hp" or "lp"
	// Memory stats (populated by scheduler loop each tick)
	MemoryCurrent     int64 // bytes on GPU device (memory.current)
	MemoryLimitHigh   int64 // hard ceiling in bytes (memory.limit.high)
	MemoryLimitLow    int64 // soft reservation in bytes (memory.limit.low)
	MemorySwapCurrent int64 // bytes swapped to host (memory.swap.current)
}

// SchedulerContext is passed to each policy Tick with all process stats.
type SchedulerContext struct {
	HPPending   int64            // aggregate HP pending kernel count (or nvidia-smi fallback)
	HPProcesses []ProcessInfo    // HP processes with memory stats
	LPProcesses []ProcessInfo    // LP processes with memory stats
	GPUTotalMem map[int]int64    // GPU index -> total memory in bytes
}

// SchedulingAction represents a single scheduling action to apply.
type SchedulingAction struct {
	PID             int
	GPUIndex        int
	Action          string // "set_priority", "freeze", "unfreeze", "set_memory_limit_high", "set_memory_limit_low"
	Priority        int    // only used for "set_priority"
	MemoryLimitHigh int64  // only used for "set_memory_limit_high" (bytes)
	MemoryLimitLow  int64  // only used for "set_memory_limit_low" (bytes)
}

// SchedulerConfig holds all configurable parameters for scheduling policies.
type SchedulerConfig struct {
	PollInterval time.Duration

	// Policy 1: Burst Freeze
	BurstHighThreshold int64
	BurstLowThreshold  int64

	// Policy 2: Dynamic Priority
	DynLightThreshold   int64
	DynHeavyThreshold   int64
	DynExtremeThreshold int64
	DynIdlePriority     int
	DynLightPriority    int
	DynHeavyPriority    int

	// Policy 3: Memory Relaxation
	MemBorrowFraction      float64
	MemSafetyMarginBytes   int64
	MemRampDownStepBytes   int64
	MemGrowthThresholdBytes int64
	MemFastGrowthThresholdBytes int64
	MemReclaimLeadtimeMS   int

	// Policy 4: Swap Throttling
	SwapFreezeRatio   float64
	SwapThrottleRatio float64
	SwapClearRatio    float64
	SwapCooldownTicks int

	// Combined memory_aware: which compute policy to use as base
	ComputePolicy string
}

// DefaultSchedulerConfig returns the default scheduler configuration.
func DefaultSchedulerConfig() SchedulerConfig {
	return SchedulerConfig{
		PollInterval:        50 * time.Millisecond,
		BurstHighThreshold:  5,
		BurstLowThreshold:   1,
		DynLightThreshold:   2,
		DynHeavyThreshold:   8,
		DynExtremeThreshold: 20,
		DynIdlePriority:     0,
		DynLightPriority:    4,
		DynHeavyPriority:    12,

		MemBorrowFraction:           0.7,
		MemSafetyMarginBytes:        50 * 1024 * 1024,  // 50 MB
		MemRampDownStepBytes:         100 * 1024 * 1024, // 100 MB
		MemGrowthThresholdBytes:      10 * 1024 * 1024,  // 10 MB
		MemFastGrowthThresholdBytes:  50 * 1024 * 1024,  // 50 MB
		MemReclaimLeadtimeMS:         200,

		SwapFreezeRatio:   0.3,
		SwapThrottleRatio: 0.1,
		SwapClearRatio:    0.02,
		SwapCooldownTicks: 4,

		ComputePolicy: "dynamic_priority",
	}
}

// SchedulerConfigFromEnv overrides defaults with environment variables.
func SchedulerConfigFromEnv() SchedulerConfig {
	config := DefaultSchedulerConfig()

	if v := os.Getenv("GVM_POLL_INTERVAL_MS"); v != "" {
		if ms, err := strconv.Atoi(v); err == nil && ms > 0 {
			config.PollInterval = time.Duration(ms) * time.Millisecond
		}
	}
	if v := os.Getenv("GVM_BURST_HIGH"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			config.BurstHighThreshold = n
		}
	}
	if v := os.Getenv("GVM_BURST_LOW"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			config.BurstLowThreshold = n
		}
	}
	if v := os.Getenv("GVM_DYN_LIGHT_THRESHOLD"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			config.DynLightThreshold = n
		}
	}
	if v := os.Getenv("GVM_DYN_HEAVY_THRESHOLD"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			config.DynHeavyThreshold = n
		}
	}
	if v := os.Getenv("GVM_DYN_EXTREME_THRESHOLD"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			config.DynExtremeThreshold = n
		}
	}
	if v := os.Getenv("GVM_DYN_IDLE_PRIORITY"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 && n <= 15 {
			config.DynIdlePriority = n
		}
	}
	if v := os.Getenv("GVM_DYN_LIGHT_PRIORITY"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 && n <= 15 {
			config.DynLightPriority = n
		}
	}
	if v := os.Getenv("GVM_DYN_HEAVY_PRIORITY"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 && n <= 15 {
			config.DynHeavyPriority = n
		}
	}

	// Memory relaxation (Policy 3)
	if v := os.Getenv("GVM_MEM_BORROW_FRACTION"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f > 0 && f <= 1 {
			config.MemBorrowFraction = f
		}
	}
	if v := os.Getenv("GVM_MEM_SAFETY_MARGIN_MB"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n >= 0 {
			config.MemSafetyMarginBytes = n * 1024 * 1024
		}
	}
	if v := os.Getenv("GVM_MEM_RAMP_DOWN_STEP_MB"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n > 0 {
			config.MemRampDownStepBytes = n * 1024 * 1024
		}
	}
	if v := os.Getenv("GVM_MEM_GROWTH_THRESHOLD_MB"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n > 0 {
			config.MemGrowthThresholdBytes = n * 1024 * 1024
		}
	}
	if v := os.Getenv("GVM_MEM_FAST_GROWTH_THRESHOLD_MB"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n > 0 {
			config.MemFastGrowthThresholdBytes = n * 1024 * 1024
		}
	}
	if v := os.Getenv("GVM_MEM_RECLAIM_LEADTIME_MS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			config.MemReclaimLeadtimeMS = n
		}
	}

	// Swap throttling (Policy 4)
	if v := os.Getenv("GVM_SWAP_FREEZE_RATIO"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f > 0 && f <= 1 {
			config.SwapFreezeRatio = f
		}
	}
	if v := os.Getenv("GVM_SWAP_THROTTLE_RATIO"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f > 0 && f <= 1 {
			config.SwapThrottleRatio = f
		}
	}
	if v := os.Getenv("GVM_SWAP_CLEAR_RATIO"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 && f <= 1 {
			config.SwapClearRatio = f
		}
	}
	if v := os.Getenv("GVM_SWAP_COOLDOWN_TICKS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			config.SwapCooldownTicks = n
		}
	}

	// Combined policy: underlying compute policy
	if v := os.Getenv("GVM_COMPUTE_POLICY"); v != "" {
		config.ComputePolicy = v
	}

	return config
}

// SelectPolicy returns the appropriate SchedulerPolicy based on the policy name.
// Returns nil if no dynamic scheduling is requested (empty string or "none").
func SelectPolicy(name string) SchedulerPolicy {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "burst_freeze":
		return &BurstFreezePolicy{}
	case "dynamic_priority":
		return &DynamicPriorityPolicy{}
	case "memory_relaxation":
		return &MemoryRelaxationPolicy{}
	case "swap_throttling":
		return &SwapThrottlingPolicy{}
	case "memory_aware":
		return &MemoryAwarePolicy{}
	default:
		return nil
	}
}

// ProcessRegistry is a thread-safe registry of HP and LP processes
// shared between the discovery goroutine and the scheduling goroutine.
type ProcessRegistry struct {
	mu          sync.RWMutex
	hpProcesses []ProcessInfo
	lpProcesses []ProcessInfo
}

func NewProcessRegistry() *ProcessRegistry {
	return &ProcessRegistry{}
}

func (r *ProcessRegistry) Update(hp, lp []ProcessInfo) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.hpProcesses = hp
	r.lpProcesses = lp
}

func (r *ProcessRegistry) Get() ([]ProcessInfo, []ProcessInfo) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	// Return copies to avoid data races
	hp := make([]ProcessInfo, len(r.hpProcesses))
	copy(hp, r.hpProcesses)
	lp := make([]ProcessInfo, len(r.lpProcesses))
	copy(lp, r.lpProcesses)
	return hp, lp
}

// RunScheduler starts the fast scheduling loop in a goroutine. It reads
// HP process stats and invokes the policy to generate actions for LP processes.
func RunScheduler(policy SchedulerPolicy, registry *ProcessRegistry, config SchedulerConfig) {
	fmt.Printf("[%s] Scheduler started: policy=%s poll_interval=%s\n", ts(), policy.Name(), config.PollInterval)

	ticker := time.NewTicker(config.PollInterval)
	defer ticker.Stop()

	// Query GPU total memory once at startup
	gpuTotalMem, err := gvm.QueryGPUTotalMemory()
	if err != nil {
		fmt.Fprintf(os.Stderr, "[%s] Warning: could not query GPU total memory: %v\n", ts(), err)
		gpuTotalMem = make(map[int]int64)
	} else {
		for idx, mem := range gpuTotalMem {
			fmt.Printf("[%s] GPU %d total memory: %s\n", ts(), idx, gvm.FormatMemoryValue(mem))
		}
	}

	useGPUUtil := false
	for range ticker.C {
		hp, lp := registry.Get()
		if len(hp) == 0 || len(lp) == 0 {
			continue
		}

		// Read memory stats for all processes
		populateMemoryStats(hp)
		populateMemoryStats(lp)

		hpLoad := ReadHPLoad(hp, &useGPUUtil)

		ctx := SchedulerContext{
			HPPending:   hpLoad,
			HPProcesses: hp,
			LPProcesses: lp,
			GPUTotalMem: gpuTotalMem,
		}

		actions := policy.Tick(ctx)
		for _, action := range actions {
			applySchedulingAction(action)
		}
	}
}

// populateMemoryStats reads memory.current, memory.limit.high, memory.limit.low,
// and memory.swap.current for each process. Errors are silently ignored (process may have exited).
func populateMemoryStats(procs []ProcessInfo) {
	for i := range procs {
		p := &procs[i]
		if v, err := gvm.GetMemoryCurrent(p.PID, p.GPUIndex); err == nil {
			p.MemoryCurrent = v
		}
		if v, err := gvm.GetMemoryLimitHigh(p.PID, p.GPUIndex); err == nil {
			p.MemoryLimitHigh = v
		}
		if v, err := gvm.GetMemoryLimitLow(p.PID, p.GPUIndex); err == nil {
			p.MemoryLimitLow = v
		}
		if v, err := gvm.GetMemorySwapCurrent(p.PID, p.GPUIndex); err == nil {
			p.MemorySwapCurrent = v
		}
	}
}

// applySchedulingAction applies a single scheduling action via sysfs.
func applySchedulingAction(action SchedulingAction) {
	var err error
	switch action.Action {
	case "set_priority":
		err = gvm.SetComputePriority(action.PID, action.GPUIndex, action.Priority)
	case "freeze":
		err = gvm.SetComputeFreeze(action.PID, action.GPUIndex, true)
	case "unfreeze":
		err = gvm.SetComputeFreeze(action.PID, action.GPUIndex, false)
	case "set_memory_limit_high":
		err = gvm.SetMemoryLimitHigh(action.PID, action.GPUIndex, action.MemoryLimitHigh)
	case "set_memory_limit_low":
		err = gvm.SetMemoryLimitLow(action.PID, action.GPUIndex, action.MemoryLimitLow)
	default:
		fmt.Fprintf(os.Stderr, "[%s] Unknown scheduling action: %s\n", ts(), action.Action)
		return
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "[%s] Error applying %s to PID %d GPU %d: %v\n",
			ts(), action.Action, action.PID, action.GPUIndex, err)
	}
}

// ReadHPPendingKernels reads the total pending kernels across all HP processes.
// Returns 0 on any read error (graceful degradation).
func ReadHPPendingKernels(hpProcesses []ProcessInfo) int64 {
	var total int64
	for _, hp := range hpProcesses {
		stat, err := gvm.ReadGcgroupStat(hp.PID, hp.GPUIndex)
		if err != nil {
			// Process may have exited — skip gracefully
			continue
		}
		total += stat.NrPendingKernels
	}
	return total
}

// ReadGPUUtilization queries nvidia-smi for GPU utilization percentage (0-100).
// Returns -1 on error. This is used as a fallback when gcgroup.stat counters
// are not populated by the kernel module.
func ReadGPUUtilization() int64 {
	cmd := exec.Command("nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits")
	output, err := cmd.Output()
	if err != nil {
		return -1
	}
	val := strings.TrimSpace(string(output))
	// Multi-GPU: take first GPU for now
	lines := strings.Split(val, "\n")
	if len(lines) == 0 {
		return -1
	}
	util, err := strconv.ParseInt(strings.TrimSpace(lines[0]), 10, 64)
	if err != nil {
		return -1
	}
	return util
}

// ReadHPLoad returns the HP load signal. It first tries gcgroup.stat pending
// kernels. If those are zero (kernel module doesn't populate them), it falls
// back to nvidia-smi GPU utilization, scaled to a pending-kernel-equivalent:
// utilization 0-100% → scaled 0-25 (so thresholds work similarly).
func ReadHPLoad(hpProcesses []ProcessInfo, useGPUUtil *bool) int64 {
	if !*useGPUUtil {
		pending := ReadHPPendingKernels(hpProcesses)
		if pending > 0 {
			return pending
		}
		// First time we see zero, check if gcgroup.stat is always zero
		// by also reading GPU utilization. If GPU is active but pending=0,
		// switch to utilization mode permanently.
		util := ReadGPUUtilization()
		if util > 10 {
			// GPU is clearly active but gcgroup.stat says 0 → switch mode
			*useGPUUtil = true
			fmt.Printf("[%s] gcgroup.stat counters are zero despite GPU activity (%d%%). Switching to nvidia-smi utilization mode (effective poll ~200ms).\n", ts(), util)
			return util / 4 // scale 0-100 → 0-25
		}
		return 0
	}
	// GPU utilization mode
	util := ReadGPUUtilization()
	if util < 0 {
		return 0
	}
	return util / 4 // scale 0-100 → 0-25
}
