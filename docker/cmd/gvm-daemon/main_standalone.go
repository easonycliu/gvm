//go:build standalone

package main

import (
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/easonycliu/GCgroupProject/docker/pkg/gvm"
)

// intSlice is a flag type for comma-separated ints (e.g. "1234,5678").
type intSlice []int

func (s *intSlice) String() string {
	parts := make([]string, len(*s))
	for i, v := range *s {
		parts[i] = strconv.Itoa(v)
	}
	return strings.Join(parts, ",")
}

func (s *intSlice) Set(value string) error {
	for _, part := range strings.Split(value, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil {
			return fmt.Errorf("invalid PID %q: %v", part, err)
		}
		*s = append(*s, n)
	}
	return nil
}

func main() {
	var (
		hpPIDs     intSlice
		lpPIDs     intSlice
		gpuIndex   int
		policyName string
		intervalMS int

		// Initial memory limits (bytes, -1 = unlimited)
		hpMemLimitHigh int64
		hpMemLimitLow  int64
		lpMemLimitHigh int64
		lpMemLimitLow  int64
		hpPriority     int
		lpPriority     int

		// Burst freeze
		burstHigh int64
		burstLow  int64

		// Dynamic priority
		dynLight         int64
		dynHeavy         int64
		dynExtreme       int64
		dynIdlePriority  int
		dynLightPriority int
		dynHeavyPriority int

		// Memory relaxation
		memBorrowFraction float64
		memSafetyMB       int64
		memRampDownMB     int64
		memGrowthMB       int64
		memFastGrowthMB   int64
		memLeadtimeMS     int

		// Swap throttling
		swapFreezeRatio   float64
		swapThrottleRatio float64
		swapClearRatio    float64
		swapCooldownTicks int

		// Combined policy
		computePolicy string

		// Adaptive memory
		adaptiveHPIdle       int64
		adaptiveHPBusy       int64
		adaptiveHPMinCache   float64
		adaptiveHPMaxCache   float64
		adaptiveLPMinResMB   int64
		adaptiveRampMB       int64
		adaptiveSafetyMB     int64
	)

	flag.Var(&hpPIDs, "hppid", "HP (high-priority / latency-critical) PID(s), comma-separated")
	flag.Var(&lpPIDs, "lppid", "LP (low-priority / best-effort) PID(s), comma-separated")
	flag.IntVar(&gpuIndex, "gpu", 0, "GPU index (default: 0)")
	flag.StringVar(&policyName, "policy", "dynamic_priority",
		"Scheduling policy: burst_freeze, dynamic_priority, memory_relaxation, swap_throttling, memory_aware, adaptive_memory")
	flag.IntVar(&intervalMS, "interval", 100, "Poll interval in ms (default: 100)")

	flag.Int64Var(&hpMemLimitHigh, "hp-memlimit-high", -1, "HP memory.limit.high in bytes (-1 = unlimited)")
	flag.Int64Var(&hpMemLimitLow, "hp-memlimit-low", 0, "HP memory.limit.low in bytes (0 = no reservation)")
	flag.Int64Var(&lpMemLimitHigh, "lp-memlimit-high", -1, "LP memory.limit.high in bytes (-1 = unlimited)")
	flag.Int64Var(&lpMemLimitLow, "lp-memlimit-low", 0, "LP memory.limit.low in bytes (0 = no reservation)")
	flag.IntVar(&hpPriority, "hp-priority", 0, "HP compute.priority (0-15, default: 0 = highest)")
	flag.IntVar(&lpPriority, "lp-priority", 8, "LP initial compute.priority (0-15, default: 8)")

	flag.Int64Var(&burstHigh, "burst-high", 5, "Burst freeze: high threshold")
	flag.Int64Var(&burstLow, "burst-low", 1, "Burst freeze: low threshold")

	flag.Int64Var(&dynLight, "dyn-light", 2, "Dynamic priority: light load threshold")
	flag.Int64Var(&dynHeavy, "dyn-heavy", 8, "Dynamic priority: heavy load threshold")
	flag.Int64Var(&dynExtreme, "dyn-extreme", 20, "Dynamic priority: extreme/freeze threshold")
	flag.IntVar(&dynIdlePriority, "dyn-idle-priority", 0, "Dynamic priority: idle priority (default: 0)")
	flag.IntVar(&dynLightPriority, "dyn-light-priority", 4, "Dynamic priority: light priority (default: 4)")
	flag.IntVar(&dynHeavyPriority, "dyn-heavy-priority", 12, "Dynamic priority: heavy priority (default: 12)")

	flag.Float64Var(&memBorrowFraction, "mem-borrow-fraction", 0.7, "Memory relaxation: borrow fraction (0-1)")
	flag.Int64Var(&memSafetyMB, "mem-safety-mb", 50, "Memory relaxation: safety margin in MB")
	flag.Int64Var(&memRampDownMB, "mem-ramp-down-mb", 100, "Memory relaxation: ramp-down step in MB")
	flag.Int64Var(&memGrowthMB, "mem-growth-mb", 10, "Memory relaxation: growth threshold in MB")
	flag.Int64Var(&memFastGrowthMB, "mem-fast-growth-mb", 50, "Memory relaxation: fast growth threshold in MB")
	flag.IntVar(&memLeadtimeMS, "mem-leadtime-ms", 200, "Memory relaxation: reclaim leadtime in ms")

	flag.Float64Var(&swapFreezeRatio, "swap-freeze-ratio", 0.3, "Swap throttling: freeze ratio")
	flag.Float64Var(&swapThrottleRatio, "swap-throttle-ratio", 0.1, "Swap throttling: throttle ratio")
	flag.Float64Var(&swapClearRatio, "swap-clear-ratio", 0.02, "Swap throttling: clear ratio")
	flag.IntVar(&swapCooldownTicks, "swap-cooldown-ticks", 4, "Swap throttling: cooldown ticks")

	flag.StringVar(&computePolicy, "compute-policy", "dynamic_priority",
		"For memory_aware/adaptive_memory: underlying compute policy (burst_freeze or dynamic_priority)")

	flag.Int64Var(&adaptiveHPIdle, "adaptive-hp-idle", 2, "Adaptive memory: HP load idle threshold")
	flag.Int64Var(&adaptiveHPBusy, "adaptive-hp-busy", 8, "Adaptive memory: HP load busy threshold")
	flag.Float64Var(&adaptiveHPMinCache, "adaptive-hp-min-cache", 0.20, "Adaptive memory: HP min cache fraction of GPU (0-1)")
	flag.Float64Var(&adaptiveHPMaxCache, "adaptive-hp-max-cache", 0.80, "Adaptive memory: HP max cache fraction of GPU (0-1)")
	flag.Int64Var(&adaptiveLPMinResMB, "adaptive-lp-min-res-mb", 512, "Adaptive memory: LP minimum reservation in MB")
	flag.Int64Var(&adaptiveRampMB, "adaptive-ramp-mb", 128, "Adaptive memory: ramp step per tick in MB")
	flag.Int64Var(&adaptiveSafetyMB, "adaptive-safety-mb", 100, "Adaptive memory: safety margin in MB")

	flag.Parse()

	if len(hpPIDs) == 0 || len(lpPIDs) == 0 {
		fmt.Fprintf(os.Stderr, "Error: --hppid and --lppid are required\n")
		fmt.Fprintf(os.Stderr, "Usage: gvm-scheduler --hppid=<pid> --lppid=<pid> [--policy=<name>] [options...]\n")
		flag.PrintDefaults()
		os.Exit(1)
	}

	fmt.Println("GVM Standalone Scheduler starting...")
	fmt.Printf("  Policy:       %s\n", policyName)
	fmt.Printf("  HP PIDs:      %v\n", []int(hpPIDs))
	fmt.Printf("  LP PIDs:      %v\n", []int(lpPIDs))
	fmt.Printf("  GPU:          %d\n", gpuIndex)
	fmt.Printf("  Interval:     %dms\n", intervalMS)
	fmt.Printf("  Sysfs path:   %s\n", gvm.GVMProcessesPath)

	// Build SchedulerConfig from CLI flags
	config := SchedulerConfig{
		PollInterval:                time.Duration(intervalMS) * time.Millisecond,
		BurstHighThreshold:          burstHigh,
		BurstLowThreshold:           burstLow,
		DynLightThreshold:           dynLight,
		DynHeavyThreshold:           dynHeavy,
		DynExtremeThreshold:         dynExtreme,
		DynIdlePriority:             dynIdlePriority,
		DynLightPriority:            dynLightPriority,
		DynHeavyPriority:            dynHeavyPriority,
		MemBorrowFraction:           memBorrowFraction,
		MemSafetyMarginBytes:        memSafetyMB * 1024 * 1024,
		MemRampDownStepBytes:        memRampDownMB * 1024 * 1024,
		MemGrowthThresholdBytes:     memGrowthMB * 1024 * 1024,
		MemFastGrowthThresholdBytes: memFastGrowthMB * 1024 * 1024,
		MemReclaimLeadtimeMS:        memLeadtimeMS,
		SwapFreezeRatio:             swapFreezeRatio,
		SwapThrottleRatio:           swapThrottleRatio,
		SwapClearRatio:              swapClearRatio,
		SwapCooldownTicks:           swapCooldownTicks,
		ComputePolicy:                    computePolicy,
		AdaptiveHPIdleThreshold:          adaptiveHPIdle,
		AdaptiveHPBusyThreshold:          adaptiveHPBusy,
		AdaptiveHPMinCacheFrac:           adaptiveHPMinCache,
		AdaptiveHPMaxCacheFrac:           adaptiveHPMaxCache,
		AdaptiveLPMinReservationBytes:    adaptiveLPMinResMB * 1024 * 1024,
		AdaptiveRampStepBytes:            adaptiveRampMB * 1024 * 1024,
		AdaptiveSafetyMarginBytes:        adaptiveSafetyMB * 1024 * 1024,
	}

	// Select and initialize policy
	policy := SelectPolicy(policyName)
	if policy == nil {
		fmt.Fprintf(os.Stderr, "Error: unknown policy %q. Valid: burst_freeze, dynamic_priority, memory_relaxation, swap_throttling, memory_aware, adaptive_memory\n", policyName)
		os.Exit(1)
	}
	if err := policy.Init(config); err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing policy %q: %v\n", policyName, err)
		os.Exit(1)
	}

	// Apply initial memory limits and priorities
	for _, pid := range hpPIDs {
		if hpMemLimitHigh > 0 {
			if err := gvm.SetMemoryLimitHigh(pid, gpuIndex, hpMemLimitHigh); err != nil {
				fmt.Fprintf(os.Stderr, "[%s] Warning: set HP PID %d memory.limit.high: %v\n", ts(), pid, err)
			}
		}
		if hpMemLimitLow > 0 {
			if err := gvm.SetMemoryLimitLow(pid, gpuIndex, hpMemLimitLow); err != nil {
				fmt.Fprintf(os.Stderr, "[%s] Warning: set HP PID %d memory.limit.low: %v\n", ts(), pid, err)
			}
		}
		if err := gvm.SetComputePriority(pid, gpuIndex, hpPriority); err != nil {
			fmt.Fprintf(os.Stderr, "[%s] Warning: set HP PID %d compute.priority: %v\n", ts(), pid, err)
		}
		fmt.Printf("[%s] HP PID %d GPU %d: limit.high=%d limit.low=%d priority=%d\n",
			ts(), pid, gpuIndex, hpMemLimitHigh, hpMemLimitLow, hpPriority)
	}
	for _, pid := range lpPIDs {
		if lpMemLimitHigh > 0 {
			if err := gvm.SetMemoryLimitHigh(pid, gpuIndex, lpMemLimitHigh); err != nil {
				fmt.Fprintf(os.Stderr, "[%s] Warning: set LP PID %d memory.limit.high: %v\n", ts(), pid, err)
			}
		}
		if lpMemLimitLow > 0 {
			if err := gvm.SetMemoryLimitLow(pid, gpuIndex, lpMemLimitLow); err != nil {
				fmt.Fprintf(os.Stderr, "[%s] Warning: set LP PID %d memory.limit.low: %v\n", ts(), pid, err)
			}
		}
		if err := gvm.SetComputePriority(pid, gpuIndex, lpPriority); err != nil {
			fmt.Fprintf(os.Stderr, "[%s] Warning: set LP PID %d compute.priority: %v\n", ts(), pid, err)
		}
		fmt.Printf("[%s] LP PID %d GPU %d: limit.high=%d limit.low=%d priority=%d\n",
			ts(), pid, gpuIndex, lpMemLimitHigh, lpMemLimitLow, lpPriority)
	}

	// Build static process lists
	hpProcesses := make([]ProcessInfo, len(hpPIDs))
	for i, pid := range hpPIDs {
		hpProcesses[i] = ProcessInfo{PID: pid, GPUIndex: gpuIndex, Role: "hp"}
	}
	lpProcesses := make([]ProcessInfo, len(lpPIDs))
	for i, pid := range lpPIDs {
		lpProcesses[i] = ProcessInfo{PID: pid, GPUIndex: gpuIndex, Role: "lp"}
	}

	// Register with the ProcessRegistry and start the scheduler
	registry := NewProcessRegistry()
	registry.Update(hpProcesses, lpProcesses)

	// Graceful shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		fmt.Printf("\n[%s] Shutting down (signal=%v)\n", ts(), sig)
		os.Exit(0)
	}()

	// Run scheduler loop (blocking)
	RunScheduler(policy, registry, config)
}
