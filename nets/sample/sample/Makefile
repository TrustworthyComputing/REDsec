CC=/usr/bin/g++
LIB_DIR = ../../../lib
JOIN_FLAGS= -I. -I$(LIB_DIR) -Wall -g -fopenmp
JOIN_FLAGS_GPU = -I. -I$(LIB_DIR) -g
ENC_FLAGS = -ltfhe-spqlios-fma -fopenmp 
DEPS = %.h

weight_convert: ## Run offline weight compression and optimization
	$(CC) -c -o net_wght.o net.cpp $(JOIN_FLAGS) -fopenmp -D_WEIGHT_CONVERT_
	$(CC) -c -o weight_convert.o weight_convert.cpp $(JOIN_FLAGS) -fopenmp -D_WEIGHT_CONVERT_
	$(CC) -o $@.out $(LIB_DIR)/wght/*.o net_wght.o weight_convert.o $(JOIN_FLAGS) -D_WEIGHT_CONVERT_
	./$@.out

ptxt: ## Run inference on plaintext to verify accuracy
	$(CC) -c -o net_unenc.o net.cpp $(JOIN_FLAGS) -fopenmp
	$(CC) -c -o main_unenc.o main.cpp $(JOIN_FLAGS) -fopenmp
	$(CC) -o $@.out $(LIB_DIR)/unenc/*.o net_unenc.o main_unenc.o $(JOIN_FLAGS) -fopenmp
	./$@.out

cpu-encrypt: ## Run inference on encrypted data with TFHE on CPU-based systems 
	$(CC) -c -o net_enc.o net.cpp $(JOIN_FLAGS) $(ENC_FLAGS) -fopenmp -DENCRYPTED
	$(CC) -c -o main_enc.o main.cpp $(JOIN_FLAGS) $(ENC_FLAGS) -fopenmp -DENCRYPTED
	$(CC) -o $@.out $(LIB_DIR)/enc/*.o net_enc.o main_enc.o $(JOIN_FLAGS) $(ENC_FLAGS) -DENCRYPTED
	./$@.out

gpu-encrypt: ## Run inference on encrypted data with REDcuFHE on GPU-based systems
	nvcc -c -o net_gpu.o net.cu $(JOIN_FLAGS_GPU) -lredcufhe -Xcompiler -fopenmp -Xcompiler -Wall -Xcompiler -DGPU_ENC
	nvcc -c -o main_gpu.o main.cu $(JOIN_FLAGS_GPU) -lredcufhe -Xcompiler -fopenmp -Xcompiler -Wall -Xcompiler -DGPU_ENC
	nvcc -o $@.out $(LIB_DIR)/GPU/*.o net_gpu.o main_gpu.o $(JOIN_FLAGS_GPU) -lredcufhe -Xcompiler -fopenmp -Xcompiler -Wall -Xcompiler -DGPU_ENC
	./$@.out

clean: ## Delete object files and executables
	@rm  -f *.o *.out

.PHONY: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help