#!/bin/bash

set -o pipefail

script_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
project_dir=$(realpath "$script_dir/..")
failures=0

pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1" >&2; failures=$((failures + 1)); }

check_python_package() {
	local name=$1 venv_name=$2 module=$3 expected=$4
	local python=$project_dir/venv/$venv_name/bin/python3
	local version
	if [ ! -x "$python" ]; then
		fail "$name: missing $venv_name"
		return
	fi
	if ! version=$("$python" -c "import $module; print($module.__version__)" 2>&1); then
		fail "$name: import failed in $venv_name ($version)"
		return
	fi
	if [ -n "$expected" ] && [[ "$version" != "$expected"* ]]; then
		fail "$name: expected $expected, found $version"
		return
	fi
	pass "$name $version ($venv_name)"
}

echo "GVM kick-the-tire checks"
echo "Project: $project_dir"

uvm_debugfs=/sys/kernel/debug/nvidia-uvm
if [ -d "$uvm_debugfs" ]; then
	pass "GVM NVIDIA UVM debugfs interface"
elif command -v sudo >/dev/null 2>&1; then
	echo "INFO  checking the GVM NVIDIA UVM debugfs interface with sudo"
	if sudo test -d "$uvm_debugfs"; then
		pass "GVM NVIDIA UVM debugfs interface"
	else
		fail "GVM NVIDIA UVM debugfs interface: $uvm_debugfs is missing"
	fi
else
	fail "GVM NVIDIA UVM debugfs interface: $uvm_debugfs is inaccessible and sudo is unavailable"
fi

check_python_package "Diffusers" "DiffusionVenv" "diffusers" "0.34.0"
check_python_package "LLaMA-Factory" "LFVenv" "llamafactory" "0.9.4"
check_python_package "vLLM" "VllmVenv" "vllm" "0.10.1.1"

ac_python=$project_dir/venv/VllmACVenv/bin/python3
if [ ! -x "$ac_python" ]; then
	fail "adapted vLLM: missing VllmACVenv"
elif ! ac_info=$("$ac_python" -c 'import vllm; print(vllm.__version__); print(vllm.__file__)' 2>&1); then
	fail "adapted vLLM: import failed in VllmACVenv ($ac_info)"
else
	ac_version=$(sed -n '1p' <<<"$ac_info")
	ac_path=$(sed -n '2p' <<<"$ac_info")
	case $(realpath "$ac_path") in
		"$project_dir"/apps/vllm/*)
			pass "adapted vLLM $ac_version from apps/vllm (VllmACVenv)"
			;;
		*)
			fail "adapted vLLM: VllmACVenv imports $ac_path instead of apps/vllm"
			;;
	esac
fi
check_python_package "GVMAC Transformers" "VllmACVenv" "transformers" "4.53.2"
if [ -x "$ac_python" ]; then
	if gvm_notify_path=$("$ac_python" -c 'import gvm_notify; print(gvm_notify.__file__)' 2>&1); then
		pass "gvm-notify import ($gvm_notify_path)"
	else
		fail "gvm-notify: import failed in VllmACVenv; run 'cd gvm-notify && make install-python' ($gvm_notify_path)"
	fi
fi

vllm_python=$project_dir/venv/VllmVenv/bin/python3
interposer_dir=$project_dir/gvm-cuda-driver/install
if [ ! -x "$vllm_python" ]; then
	fail "GVM CUDA intercept: missing VllmVenv"
elif [ ! -e "$interposer_dir/libcuda.so.1" ]; then
	fail "GVM CUDA intercept: missing $interposer_dir/libcuda.so.1"
else
	echo "INFO  allocating a 1 GiB CUDA tensor through the GVM intercept layer"
	if allocation_output=$(
		env LD_LIBRARY_PATH="$interposer_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
			"$vllm_python" -c 'import torch; assert torch.cuda.is_available(); x = torch.empty(256 * 1024 * 1024, dtype=torch.float32, device="cuda"); x.fill_(1); torch.cuda.synchronize(); print("allocated tensor: 1024 MiB")' 2>&1
	); then
		printf '%s\n' "$allocation_output"
		if grep -q 'total cuda memory allocated: [0-9][0-9]*MB' <<<"$allocation_output"; then
			pass "GVM CUDA allocation interception"
		else
			fail "GVM CUDA allocation interception: no GVM accounting message was observed"
		fi
	else
		printf '%s\n' "$allocation_output" >&2
		fail "GVM CUDA allocation interception: CUDA test failed"
	fi
fi

if [ "$failures" -eq 0 ]; then
	echo "All checks passed."
	exit 0
fi
echo "$failures check(s) failed." >&2
exit 1
