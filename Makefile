
.PHONY: install_deps install_cuda install_linux install_driver

install_deps:
	sudo apt install flex bison libssl-dev libelf-dev

install_cuda:
	mkdir -p cuda_deb
	cd cuda_deb && sudo pwd
	@$(foreach file,$(shell ls cuda),cat cuda/$(file)/part_* > cuda_deb/$(file).deb;)
	cd cuda_deb && sudo dpkg -i $(shell ls cuda_deb)

install_linux:
	cp configs/config-gcgroup-seoul-l4 linux/.config
	cd linux && make olddefconfig && make -j$(shell nproc) && sudo make modules_install -j$(shell nproc) && sudo make install

install_driver:
	cd open-gpu-kernel-modules && make -j$(shell nproc)
	cd open-gpu-kernel-modules/kernel-open && sudo rmmod nvidia_uvm
	cd open-gpu-kernel-modules/kernel-open && sudo rmmod nvidia_drm
	cd open-gpu-kernel-modules/kernel-open && sudo rmmod nvidia_modeset
	cd open-gpu-kernel-modules/kernel-open && sudo rmmod nvidia
	sudo insmod /lib/modules/6.11.0-gcgroup+/kernel/drivers/platform/x86/wmi.ko
	sudo insmod /lib/modules/6.11.0-gcgroup+/kernel/drivers/acpi/video.ko
	sudo insmod /lib/modules/6.11.0-gcgroup+/kernel/drivers/gpu/drm/ttm/ttm.ko
	sudo insmod /lib/modules/6.11.0-gcgroup+/kernel/drivers/gpu/drm/drm_ttm_helper.ko
	cd open-gpu-kernel-modules/kernel-open && sudo insmod nvidia.ko && sudo insmod nvidia-modeset.ko && sudo insmod nvidia-drm.ko && sudo insmod nvidia-uvm.ko
	sudo mknod -m 666 /dev/nvidia0 c 195 0
	sudo mknod -m 666 /dev/nvidiactl c 195 255
	sudo mknod -m 666 /dev/nvidia-modeset c 195 254
	sudo mknod -m 666 /dev/nvidia-uvm c 508 0
	sudo mknod -m 666 /dev/nvidia-uvm-tools c 508 1
