package gvm

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	// GVMProcessesPath is the sysfs base path for GVM GPU process controls.
	GVMProcessesPath = "/sys/kernel/debug/nvidia-uvm/processes"
)

// GPUProcessPath returns the sysfs base path for a specific PID and GPU index.
// e.g., /sys/kernel/debug/nvidia-uvm/processes/12345/0/
func GPUProcessPath(pid int, gpuIndex int) string {
	return filepath.Join(GVMProcessesPath, strconv.Itoa(pid), strconv.Itoa(gpuIndex))
}

// WriteSysfs writes a value to a sysfs file. Requires root privileges.
func WriteSysfs(path, value string) error {
	file, err := os.OpenFile(path, os.O_WRONLY, 0)
	if err != nil {
		return fmt.Errorf("open %s: %w", path, err)
	}
	defer file.Close()

	_, err = file.WriteString(value)
	if err != nil {
		return fmt.Errorf("write %s to %s: %w", value, path, err)
	}
	return nil
}

// ReadSysfs reads and trims the content of a sysfs file.
func ReadSysfs(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read %s: %w", path, err)
	}
	return strings.TrimSpace(string(data)), nil
}

// SetMemoryLimitHigh writes the hard memory ceiling (memory.limit.high) in bytes.
// If the process exceeds this limit, the kernel force-evicts pages synchronously.
// The write blocks until eviction completes.
func SetMemoryLimitHigh(pid int, gpuIndex int, limitBytes int64) error {
	path := filepath.Join(GPUProcessPath(pid, gpuIndex), "memory.limit.high")
	return WriteSysfs(path, strconv.FormatInt(limitBytes, 10))
}

// GetMemoryLimitHigh reads the current hard memory ceiling (memory.limit.high).
// Returns -1 for unlimited.
func GetMemoryLimitHigh(pid int, gpuIndex int) (int64, error) {
	path := filepath.Join(GPUProcessPath(pid, gpuIndex), "memory.limit.high")
	val, err := ReadSysfs(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseInt(val, 10, 64)
}

// SetMemoryLimitLow writes the soft memory reservation (memory.limit.low) in bytes.
// This is the guaranteed safe zone — no eviction notices below this.
// If set above memory.current, triggers a reallocation notice to the process.
func SetMemoryLimitLow(pid int, gpuIndex int, limitBytes int64) error {
	path := filepath.Join(GPUProcessPath(pid, gpuIndex), "memory.limit.low")
	return WriteSysfs(path, strconv.FormatInt(limitBytes, 10))
}

// GetMemoryLimitLow reads the current soft memory reservation (memory.limit.low).
func GetMemoryLimitLow(pid int, gpuIndex int) (int64, error) {
	path := filepath.Join(GPUProcessPath(pid, gpuIndex), "memory.limit.low")
	val, err := ReadSysfs(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseInt(val, 10, 64)
}

// SetMemoryLimit is a backwards-compatible alias that sets memory.limit.high.
func SetMemoryLimit(pid int, gpuIndex int, limitBytes int64) error {
	return SetMemoryLimitHigh(pid, gpuIndex, limitBytes)
}

// GetMemoryLimit is a backwards-compatible alias that reads memory.limit.high.
func GetMemoryLimit(pid int, gpuIndex int) (int64, error) {
	return GetMemoryLimitHigh(pid, gpuIndex)
}

// GetMemoryCurrent reads the current GPU memory usage for a process.
func GetMemoryCurrent(pid int, gpuIndex int) (int64, error) {
	path := filepath.Join(GPUProcessPath(pid, gpuIndex), "memory.current")
	val, err := ReadSysfs(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseInt(val, 10, 64)
}

// GetMemorySwapCurrent reads the current swapped memory for a GPU process.
func GetMemorySwapCurrent(pid int, gpuIndex int) (int64, error) {
	path := filepath.Join(GPUProcessPath(pid, gpuIndex), "memory.swap.current")
	val, err := ReadSysfs(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseInt(val, 10, 64)
}

// SetComputePriority writes the compute priority (0-15) for a GPU process.
func SetComputePriority(pid int, gpuIndex int, priority int) error {
	if priority < 0 || priority > 15 {
		return fmt.Errorf("priority must be 0-15, got %d", priority)
	}
	path := filepath.Join(GPUProcessPath(pid, gpuIndex), "compute.priority")
	return WriteSysfs(path, strconv.Itoa(priority))
}

// GetComputePriority reads the current compute priority for a GPU process.
func GetComputePriority(pid int, gpuIndex int) (int, error) {
	path := filepath.Join(GPUProcessPath(pid, gpuIndex), "compute.priority")
	val, err := ReadSysfs(path)
	if err != nil {
		return 0, err
	}
	return strconv.Atoi(val)
}

// SetComputeFreeze freezes (true) or unfreezes (false) GPU compute for a process.
func SetComputeFreeze(pid int, gpuIndex int, freeze bool) error {
	path := filepath.Join(GPUProcessPath(pid, gpuIndex), "compute.freeze")
	val := "0"
	if freeze {
		val = "1"
	}
	return WriteSysfs(path, val)
}

// GetComputeFreeze reads whether GPU compute is frozen for a process.
func GetComputeFreeze(pid int, gpuIndex int) (bool, error) {
	path := filepath.Join(GPUProcessPath(pid, gpuIndex), "compute.freeze")
	val, err := ReadSysfs(path)
	if err != nil {
		return false, err
	}
	return val == "1", nil
}

// GetGCGroupStat reads the gcgroup stats for a GPU process.
func GetGCGroupStat(pid int, gpuIndex int) (string, error) {
	path := filepath.Join(GPUProcessPath(pid, gpuIndex), "gcgroup.stat")
	return ReadSysfs(path)
}

// GcgroupStat holds the parsed contents of a gcgroup.stat sysfs file.
type GcgroupStat struct {
	NrSubmittedKernels int64
	NrEndedKernels     int64
	NrPendingKernels   int64
}

// ReadGcgroupStat reads and parses the gcgroup.stat file for a GPU process.
func ReadGcgroupStat(pid int, gpuIndex int) (*GcgroupStat, error) {
	path := filepath.Join(GPUProcessPath(pid, gpuIndex), "gcgroup.stat")
	content, err := ReadSysfs(path)
	if err != nil {
		return nil, err
	}
	return ParseGcgroupStat(content)
}

// ParseGcgroupStat parses the text content of a gcgroup.stat file.
// Expected format (one key-value pair per line):
//
//	nr_submitted_kernels 12345
//	nr_ended_kernels 12340
//	nr_pending_kernels 5
func ParseGcgroupStat(content string) (*GcgroupStat, error) {
	stat := &GcgroupStat{}
	for _, line := range strings.Split(content, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) != 2 {
			continue
		}
		val, err := strconv.ParseInt(parts[1], 10, 64)
		if err != nil {
			continue
		}
		key := strings.TrimRight(parts[0], ":")
		switch key {
		case "nr_submitted_kernels":
			stat.NrSubmittedKernels = val
		case "nr_ended_kernels":
			stat.NrEndedKernels = val
		case "nr_pending_kernels":
			stat.NrPendingKernels = val
		}
	}
	return stat, nil
}

// ListGPUIndices returns the GPU indices available for a given PID.
func ListGPUIndices(pid int) ([]int, error) {
	pidPath := filepath.Join(GVMProcessesPath, strconv.Itoa(pid))
	entries, err := os.ReadDir(pidPath)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", pidPath, err)
	}

	var indices []int
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		idx, err := strconv.Atoi(entry.Name())
		if err != nil {
			continue
		}
		indices = append(indices, idx)
	}
	return indices, nil
}

// QueryGPUTotalMemory returns a map of GPU index → total memory in bytes.
// Uses nvidia-smi to query device memory. Returns an error if nvidia-smi is unavailable.
func QueryGPUTotalMemory() (map[int]int64, error) {
	cmd := exec.Command("nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits")
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi query failed: %w", err)
	}

	result := make(map[int]int64)
	for _, line := range strings.Split(strings.TrimSpace(string(output)), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, ",", 2)
		if len(parts) != 2 {
			continue
		}
		idx, err := strconv.Atoi(strings.TrimSpace(parts[0]))
		if err != nil {
			continue
		}
		// nvidia-smi reports memory in MiB
		mib, err := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
		if err != nil {
			continue
		}
		result[idx] = mib * 1024 * 1024 // Convert MiB to bytes
	}

	return result, nil
}

// ApplyConfig applies a full GVM Config to a specific GPU process.
// totalGPUMemory is needed for percentage-based limits (pass 0 if unknown).
func ApplyConfig(pid int, gpuIndex int, config *Config, totalGPUMemory int64) error {
	if !config.Enabled {
		return nil
	}

	// Apply memory limits (two-level: limit.high = hard ceiling, limit.low = reservation)
	memLimit := config.MemoryLimitForDevice(gpuIndex, totalGPUMemory)
	if memLimit > 0 {
		if err := SetMemoryLimitHigh(pid, gpuIndex, memLimit); err != nil {
			return fmt.Errorf("set memory.limit.high for PID %d GPU %d: %w", pid, gpuIndex, err)
		}
	}

	// Apply memory reservation if configured
	memReservation := config.MemoryReservationForDevice(gpuIndex, totalGPUMemory)
	if memReservation > 0 {
		if err := SetMemoryLimitLow(pid, gpuIndex, memReservation); err != nil {
			return fmt.Errorf("set memory.limit.low for PID %d GPU %d: %w", pid, gpuIndex, err)
		}
	}

	// Apply compute priority
	if err := SetComputePriority(pid, gpuIndex, config.ComputePriority); err != nil {
		return fmt.Errorf("set compute.priority for PID %d GPU %d: %w", pid, gpuIndex, err)
	}

	// Apply compute freeze
	if config.ComputeFreeze {
		if err := SetComputeFreeze(pid, gpuIndex, true); err != nil {
			return fmt.Errorf("set compute.freeze for PID %d GPU %d: %w", pid, gpuIndex, err)
		}
	}

	return nil
}
