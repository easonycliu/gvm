#include <cuda_runtime.h>
#include <atomic>
#include <filesystem>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <sys/mman.h>

#define KERNEL_NOTIFICATION_SHM "/kernel_notification"
#define KERNEL_NOTIFICATION_SHM_SIZE 4096
struct kernel_notification {
	std::atomic<bool> running;
};

#define NVA06C_CTRL_CMD_SET_TIMESLICE (0xa06c0103) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GROUP_A_GPFIFO_INTERFACE_ID << 8) | NVA06C_CTRL_SET_TIMESLICE_PARAMS_MESSAGE_ID" */
#define NVA06C_CTRL_CMD_GET_TIMESLICE (0xa06c0104) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GROUP_A_GPFIFO_INTERFACE_ID << 8) | NVA06C_CTRL_GET_TIMESLICE_PARAMS_MESSAGE_ID" */
#define NVA06F_CTRL_CMD_STOP_CHANNEL (0xa06f0112) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GPFIFO_A_GPFIFO_INTERFACE_ID << 8) | NVA06F_CTRL_STOP_CHANNEL_PARAMS_MESSAGE_ID" */
#define NVA06C_CTRL_CMD_PREEMPT (0xa06c0105) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GROUP_A_GPFIFO_INTERFACE_ID << 8) | NVA06C_CTRL_PREEMPT_PARAMS_MESSAGE_ID" */
#define NVA06F_CTRL_CMD_GPFIFO_SCHEDULE (0xa06f0103) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GPFIFO_A_GPFIFO_INTERFACE_ID << 8) | NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS_MESSAGE_ID" */
#define NVA06F_CTRL_CMD_RESTART_RUNLIST (0xa06f0111) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GPFIFO_A_GPFIFO_INTERFACE_ID << 8) | NVA06F_CTRL_RESTART_RUNLIST_PARAMS_MESSAGE_ID" */
#define NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL (0xa06c0107) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GROUP_A_GPFIFO_INTERFACE_ID << 8) | NVA06C_CTRL_SET_INTERLEAVE_LEVEL_PARAMS_MESSAGE_ID" */
#define NVA06F_CTRL_CMD_BIND (0xa06f0104) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GPFIFO_A_GPFIFO_INTERFACE_ID << 8) | NVA06F_CTRL_BIND_PARAMS_MESSAGE_ID" */

#define UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP 81
#define UVM_CTRL_CMD_OPERATE_CHANNEL 82

typedef struct {
	std::uint64_t       timesliceUs;
} NVA06C_CTRL_TIMESLICE_PARAMS;

typedef struct {
    bool bWait;
    bool bManualTimeout;
	std::uint32_t  timeoutUs;
} NVA06C_CTRL_PREEMPT_PARAMS;

typedef struct {
    std::uint32_t tsgInterleaveLevel;     // IN
} NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS;

typedef union {
	NVA06C_CTRL_TIMESLICE_PARAMS NVA06C_CTRL_TIMESLICE_PARAMS;
	NVA06C_CTRL_PREEMPT_PARAMS NVA06C_CTRL_PREEMPT_PARAMS;
	NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS;
} UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP_PARAMS_DATA;

typedef struct {
    bool bEnable;               // IN
    bool bSkipSubmit;           // IN
    bool bSkipEnable;           // IN
} NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS;

typedef struct {
    bool bForceRestart;
    bool bBypassWait;
} NVA06F_CTRL_RESTART_RUNLIST_PARAMS;

typedef struct {
    bool bImmediate;
} NVA06F_CTRL_STOP_CHANNEL_PARAMS;

typedef struct NVA06F_CTRL_BIND_PARAMS {
	std::uint32_t engineType;
} NVA06F_CTRL_BIND_PARAMS;

typedef union {
	NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS;
	NVA06F_CTRL_RESTART_RUNLIST_PARAMS NVA06F_CTRL_RESTART_RUNLIST_PARAMS;
	NVA06F_CTRL_STOP_CHANNEL_PARAMS NVA06F_CTRL_STOP_CHANNEL_PARAMS;
	NVA06F_CTRL_BIND_PARAMS NVA06F_CTRL_BIND_PARAMS;
} UVM_CTRL_CMD_OPERATE_CHANNEL_PARAMS_DATA;

typedef struct
{
	unsigned int			cmd;       // IN
	UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP_PARAMS_DATA data;							   // IN
	size_t 					dataSize;  // IN
	int						rmStatus;  // OUT
} UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP_PARAMS;

typedef struct
{
	unsigned int			cmd;       // IN
	UVM_CTRL_CMD_OPERATE_CHANNEL_PARAMS_DATA data;							   // IN
	size_t 					dataSize;  // IN
	int						rmStatus;  // OUT
} UVM_CTRL_CMD_OPERATE_CHANNEL_PARAMS;

#define UVM_IS_INITIALIZED 80
typedef struct
{
	bool					initialized;    // IN/OUT
	int						rmStatus;		// OUT
} UVM_IS_INITIALIZED_PARAMS;

#define UVM_SET_GMEMCG 83
typedef struct
{
	std::size_t             size;         // IN
	int                     rmStatus;     // OUT
} UVM_SET_GMEMCG_PARAMS;

int find_initialized_uvm() {
	pid_t pid = getpid();
	const std::string target_path = "/dev/nvidia-uvm";

    std::string fd_dir = "/proc/" + std::to_string(pid) + "/fd";
    for (const auto& entry : std::filesystem::directory_iterator(fd_dir)) {
        std::error_code ec;
        std::string fd_path = std::filesystem::read_symlink(entry, ec).string();
        if (ec) continue;

		if (fd_path == target_path) {
			int fd = std::stoi(entry.path().filename());
			if (fd >= 0) {
				UVM_IS_INITIALIZED_PARAMS params = {
					.initialized = false,
					.rmStatus = 0
				};
				if (ioctl(fd, UVM_IS_INITIALIZED, &params) == 0) {
					if (params.initialized) {
						return fd;
					}
				} else {
					std::cerr << "Failed to check UVM initialization: " << strerror(errno) << "\n";
				}
			} else {
				std::cerr << "Invalid file descriptor: " << fd << "\n";
			}
		}
    }
    return -1;
}

void set_timeslice(int fd, long long unsigned timesliceUs) {
	if (fd < 0) {
		std::cerr << "Invalid file descriptor\n";
		return;
	}

	UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP_PARAMS params = {
		.cmd = NVA06C_CTRL_CMD_SET_TIMESLICE,
		.data = {
			.NVA06C_CTRL_TIMESLICE_PARAMS = {
				.timesliceUs = timesliceUs
			}
		},
		.dataSize = sizeof(NVA06C_CTRL_TIMESLICE_PARAMS),
		.rmStatus = 0
	};
	int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP, &params);
	if (ret < 0) {
		std::cerr << "Failed to set timeslice: " << strerror(errno) << "\n";
	} else {
		std::cout << "Timeslice: " << params.data.NVA06C_CTRL_TIMESLICE_PARAMS.timesliceUs << "\n";
	}
}

long long unsigned get_timeslice(int fd) {
	if (fd < 0) {
		std::cerr << "Invalid file descriptor\n";
		return 0;
	}

	long long unsigned timesliceUs = 0;

	UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP_PARAMS params = {
		.cmd = NVA06C_CTRL_CMD_GET_TIMESLICE,
		.data = {
			.NVA06C_CTRL_TIMESLICE_PARAMS = {
				.timesliceUs = timesliceUs
			}
		},
		.dataSize = sizeof(NVA06C_CTRL_TIMESLICE_PARAMS),
		.rmStatus = 0
	};
	int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP, &params);
	if (ret < 0) {
		std::cerr << "Failed to get timeslice: " << strerror(errno) << "\n";
	}

	return params.data.NVA06C_CTRL_TIMESLICE_PARAMS.timesliceUs;
}

void preempt(int fd) {
	if (fd < 0) {
		std::cerr << "Invalid file descriptor\n";
		return;
	}

	UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP_PARAMS params = {
		.cmd = NVA06C_CTRL_CMD_PREEMPT,
		.data = {
			.NVA06C_CTRL_PREEMPT_PARAMS = {
				.bWait = true,
				.bManualTimeout = false,
				.timeoutUs = 1000
			}
		},
		.dataSize = sizeof(NVA06C_CTRL_PREEMPT_PARAMS),
		.rmStatus = 0
	};
	int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP, &params);
	if (ret < 0) {
		std::cerr << "Failed to preempt: " << strerror(errno) << "\n";
	} else {
		std::cout << "Preempted successfully" << "\n";
	}
}

void restart(int fd) {
	if (fd < 0) {
		std::cerr << "Invalid file descriptor\n";
		return;
	}

	UVM_CTRL_CMD_OPERATE_CHANNEL_PARAMS params = {
		.cmd = NVA06F_CTRL_CMD_RESTART_RUNLIST,
		.data = {
			.NVA06F_CTRL_RESTART_RUNLIST_PARAMS = {
				.bForceRestart = true,
				.bBypassWait = false
			}
		},
		.dataSize = sizeof(NVA06F_CTRL_CMD_RESTART_RUNLIST),
		.rmStatus = 0
	};
	int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL, &params);
	if (ret < 0) {
		std::cerr << "Failed to restart: " << strerror(errno) << "\n";
	} else {
		std::cout << "Restarted successfully" << "\n";
	}
}

void schedule(int fd, bool enable) {
	if (fd < 0) {
		std::cerr << "Invalid file descriptor\n";
		return;
	}

	UVM_CTRL_CMD_OPERATE_CHANNEL_PARAMS params = {
		.cmd = NVA06F_CTRL_CMD_GPFIFO_SCHEDULE,
		.data = {
			.NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS = {
				.bEnable = enable,
				.bSkipSubmit = !enable,
				.bSkipEnable =!enable 
			}
		},
		.dataSize = sizeof(	NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS),
		.rmStatus = 0
	};
	int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL, &params);
	if (ret < 0) {
		std::cerr << "Failed to schedule: " << strerror(errno) << "\n";
	} else {
		std::cout << "Scheduled successfully" << "\n";
	}
}

void stop(int fd) {
	if (fd < 0) {
		std::cerr << "Invalid file descriptor\n";
		return;
	}

	UVM_CTRL_CMD_OPERATE_CHANNEL_PARAMS params = {
		.cmd = NVA06F_CTRL_CMD_STOP_CHANNEL,
		.data = {
			.NVA06F_CTRL_STOP_CHANNEL_PARAMS = {
				.bImmediate = true
			}
		},
		.dataSize = sizeof(NVA06F_CTRL_STOP_CHANNEL_PARAMS),
		.rmStatus = 0
	};
	int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL, &params);
	if (ret < 0) {
		std::cerr << "Failed to stop: " << strerror(errno) << "\n";
	} else {
		std::cout << "Stopped successfully" << "\n";
	}
}

void set_interleave(int fd, std::uint32_t interleave) {
	if (fd < 0) {
		std::cerr << "Invalid file descriptor\n";
		return;
	}

	UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP_PARAMS params = {
		.cmd = NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL,
		.data = {
			.NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS = {
				.tsgInterleaveLevel = interleave
			}
		},
		.dataSize = sizeof(NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS),
		.rmStatus = 0
	};
	int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP, &params);
	if (ret < 0) {
		std::cerr << "Failed to set interleave level: " << strerror(errno) << "\n";
	} else {
		std::cout << "Set interleave level to " << interleave << " successfully" << "\n";
	}
}

void bind(int fd) {
	if (fd < 0) {
		std::cerr << "Invalid file descriptor\n";
		return;
	}

	UVM_CTRL_CMD_OPERATE_CHANNEL_PARAMS params = {
		.cmd = NVA06F_CTRL_CMD_BIND,
		.data = {
			.NVA06F_CTRL_BIND_PARAMS = {
				// Will be filled in kernel module using restored type of engine
				.engineType = 0
			}
		},
		.dataSize = sizeof(NVA06F_CTRL_BIND_PARAMS),
		.rmStatus = 0
	};
	int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL, &params);
	if (ret < 0) {
		std::cerr << "Failed to bind: " << strerror(errno) << "\n";
	} else {
		std::cout << "Binded successfully" << "\n";
	}
}

void set_gmemcg(int fd, std::size_t size) {
	if (fd < 0) {
		std::cerr << "Invalid file descriptor\n";
		return;
	}

	UVM_SET_GMEMCG_PARAMS params = {
		.size = size,
		.rmStatus = 0
	};
	int ret = ioctl(fd, UVM_SET_GMEMCG, &params);
	if (ret < 0 || params.rmStatus != 0) {
		std::cerr << "Failed to set gmemcg" << std::endl;
	} else {
		std::cout << "Set gmemcg successfully" << std::endl;
	}
}

__global__ void touch_kernel(volatile int *stop_flag, volatile char *dummy, std::size_t bytes) {
    int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

	std::size_t start_index = (bytes / total_threads) * global_id;
	std::size_t end_index = min((bytes / total_threads) * (global_id + 1), bytes);
	while (!(*stop_flag)) {
		for (std::size_t i = start_index; i < end_index; ++i) {
    	    (reinterpret_cast<char *>(reinterpret_cast<std::uintptr_t>(dummy) + i))[0] = 1;
    	}
		if (global_id == 0) {
			printf("Finished one round touch\n");
		}
	}
}

int main(int argc, char **argv) {
	if (argc != 2) {
		std::cerr << "Usage: ./uvmalloc <bytes>" << std::endl;
		return 1;
	}

    int device = 0;
	cudaGetDevice(&device);

	int shmfd = shm_open(KERNEL_NOTIFICATION_SHM, O_CREAT | O_RDWR, 0666);
	if (shmfd < 0) {
		std::cerr << "Failed to open shm" << std::endl;
		return 1;
	}
	ftruncate(shmfd, KERNEL_NOTIFICATION_SHM_SIZE);

	struct kernel_notification *notification = (struct kernel_notification *) mmap(NULL, KERNEL_NOTIFICATION_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shmfd, 0);
	if (!notification) {
		std::cerr << "Failed to map shm" << std::endl;
		return 1;
	}

    std::size_t bytes = std::stoull(argv[1]);
    if (bytes <= 0) {
        std::cerr << "Bytes must be a positive value" << std::endl;
        return 1;
    }
    char *dummy;

	std::cerr << "To allocate " << bytes << " bytes" << std::endl;
    cudaMallocManaged((void **)&dummy, bytes);

    volatile int* stop_flag;
    cudaMallocManaged((void**)&stop_flag, sizeof(int));
    *stop_flag = 0;

    std::cerr << "Allocated " << bytes << " bytes" << std::endl;
					
	int uvmfd = find_initialized_uvm();

	touch_kernel<<<16, 64>>>(stop_flag, dummy, bytes);

	std::thread t = std::thread([notification, stop_flag, uvmfd, dummy, bytes, device] () {
		std::cerr << "My thread id is " << gettid() << std::endl;
		bool stopping = false;
		while (!*stop_flag) {
			if (notification->running.load()) {
				if (!stopping) {
					stopping = true;
					std::cerr << "New kernel inited, preempting myself" << std::endl;
					stop(uvmfd);
					// std::this_thread::sleep_for(std::chrono::milliseconds(2000));
					// std::cerr << "Swapping out memory";
					// std::chrono::high_resolution_clock::time_point start_setmemcg = std::chrono::high_resolution_clock::now();
					// set_gmemcg(uvmfd, 4096L);
					// std::chrono::high_resolution_clock::time_point end_setmemcg = std::chrono::high_resolution_clock::now();
					// std::cerr << "Set memcg out in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_setmemcg - start_setmemcg).count() << " ms" << std::endl;
				}
			} else {
				if (stopping) {
					stopping = false;
					// std::cerr << "New kernel finished, restarting myself" << std::endl;
					// set_gmemcg(uvmfd, 1073741824L * 24);
					bind(uvmfd);
					schedule(uvmfd, true);
				}
			}
		}
		std::cerr << "Standby thread exit" << std::endl;
	});

    std::cerr << "Press enter to exit" << std::endl;

	std::cin.get();

	*stop_flag = true;
	t.join();

	std::cerr << "Touched " << bytes << " bytes, press to swapout" << std::endl;
	std::cin.get();

	std::chrono::high_resolution_clock::time_point start_swapout = std::chrono::high_resolution_clock::now();
	cudaMemPrefetchAsync(dummy, bytes, cudaCpuDeviceId, 0);
	std::chrono::high_resolution_clock::time_point end_swapout = std::chrono::high_resolution_clock::now();
	std::cerr << "Swapped out in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_swapout - start_swapout).count() << " ms" << std::endl;

	std::cerr << "Swapping out async" << std::endl;

    cudaDeviceSynchronize();
    cudaFree(dummy);

    return 0;
}
