# 0. Edit constants.h to change the number of batches if needed:
	constexpr int NUM_BATCHES       = 5;	// set = 1 for 10k train, = 5 for 50k train

# 1. Edit main.cu to turn off just using first samples to run fully:
	int TRAIN_SAMPLES = -1;	// set = -1 < 0 to off
	int TEST_SAMPLES = -1;	// set = -1 < 0 to off

# 2. Create run_job.sh to run in background:
	nano run_job.sh

	## Paste these compile and run commands	
	#!/bin/bash
	# 1. Get the GPU architecture
	ARCH=$(python3 -c "from numba import cuda; m, n = cuda.get_current_device().compute_capability; print(f'{m}{n}')")

	# 2. Compile
	nvcc -arch=sm_$ARCH \
  main.cu data_loader.cu gpu_autoencoder.cu gpu_layers.cu \
  cpu_autoencoder.cpp cpu_layers.cpp model.cu progress_bar.cu \
  timer.cu utils.cu libsvm/svm.cpp \
  -Xcompiler -fopenmp -lgomp \
  -std=c++20 \
  -o main

	# 3. Run the program
	./main cpu train train	// 6 available modes: ./main cpu/gpu train/load train/load

# 3. Make run_job.sh executable:
	chmod +x run_job.sh

# 4. Run with nohup:
	nohup ./run_job.sh > output.log 2>&1 &

## View live output:
	tail -f output.log

## Check if it's still running:
	ps -ef | grep main

## Check GPU usage:
	nvidia-smi

## Kill nohup sections
### See PIDs If still in the same SSH session:
	jobs -l
### Find the PIDs If logged out and came back
	ps -ef | grep main

### Kill the process:
	kill <PID>
