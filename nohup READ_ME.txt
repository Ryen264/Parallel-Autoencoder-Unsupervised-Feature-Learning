# 0. Edit constants.h to change the number of batches if needed:
nano constants.h
	constexpr int NUM_BATCHES       = 5;	// set = 1 for 10k train, = 5 for 50k train

# 1. Edit cpu_main.cpp to turn off just using first samples to run fully:
nano cpu_main.cpp
	int TRAIN_SAMPLES = -1;	// set = -1 < 0 to off
	int TEST_SAMPLES = -1;	// set = -1 < 0 to off

# 2. Create run_job.sh to run in background:
nano run_job.sh

	## Paste these compile and run commands	
#!/bin/bash
# Step 1. Compile
g++ cpu_main.cpp data_loader.cpp cpu_autoencoder.cpp cpu_layers.cpp timer.cpp progress_bar.cpp utils.cpp -std=c++20 -O2 -o cpu_main.exe

# Step 2. Run exe (2 available modes: ./cpu_main train/load)
./cpu_main.exe train

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
