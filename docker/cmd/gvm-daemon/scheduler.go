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
	// Tick is called every poll interval with the aggregated HP pending kernel
	// count and the current LP processes. Returns actions to apply.
	Tick(hpPending int64, lpProcesses []ProcessInfo) []SchedulingAction
}

// ProcessInfo holds runtime info about a GPU process tracked by the scheduler.
type ProcessInfo struct {
	PID          int
	GPUIndex     int
	ContainerID  string
	ContainerName string
	Role         string // "hp" or "lp"
}

// SchedulingAction represents a single scheduling action to apply.
type SchedulingAction struct {
	PID      int
	GPUIndex int
	Action   string // "set_priority", "freeze", "unfreeze"
	Priority int    // only used for "set_priority"
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

	useGPUUtil := false
	for range ticker.C {
		hp, lp := registry.Get()
		if len(hp) == 0 || len(lp) == 0 {
			continue
		}

		hpLoad := ReadHPLoad(hp, &useGPUUtil)
		actions := policy.Tick(hpLoad, lp)
		for _, action := range actions {
			applySchedulingAction(action)
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
