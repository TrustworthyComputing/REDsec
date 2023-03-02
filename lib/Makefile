CC=/usr/bin/g++
JOIN_FLAGS= -I. -g
ENC_FLAGS = -ltfhe-spqlios-fma -fopenmp
DEPS = %.h
GPU_DEPS = GPU/%.cuh
ENC_OBJ = IntOps_enc.o BinOps_enc.o
GPU_OBJ = GPU/gates.o GPU/BinOps_gpu.o GPU/IntOps_gpu.o GPU/BinFunc_gpu.o GPU/IntFunc_gpu.o GPU/BinLayer.o GPU/IntLayer.o GPU/Layer.o
JOIN_OBJ = IntFunc.o BinFunc.o BinOps.o IntOps.o BinLayer.o IntLayer.o Layer.o

%.o: %.cpp $(DEPS)
	@mkdir -p wght
	$(CC) -c -o wght/$@ $< $(JOIN_FLAGS) -fopenmp -D_WEIGHT_CONVERT_
	@mkdir -p unenc
	$(CC) -c -o unenc/$@ $< $(JOIN_FLAGS) -fopenmp
	@mkdir -p enc
	$(CC) -c -o enc/$@ $< $(JOIN_FLAGS) -fopenmp -DENCRYPTED

GPU/%.o: GPU/%.cu $(GPU_DEPS)
	nvcc -c -o $@ $< $(JOIN_FLAGS) -I/usr/local/include -lredcufhe -Xcompiler -fopenmp -Xcompiler -DGPU_ENC

all: ptxt cpu-encrypt weight_convert gpu-encrypt ## Build all library variants

ptxt: $(JOIN_OBJ) ## Build library for ptxt inference (verification)
cpu-encrypt: $(JOIN_OBJ) $(ENC_OBJ) ## Build library for encrypted inference with CPUs
weight_convert: $(JOIN_OBJ) ## Build library for weight compression and optimization
gpu-encrypt: $(GPU_OBJ) ## Build library for encrypted inference with GPUs

clean: ## Delete object files
	@rm -f *.o enc/*.o unenc/*.o wght/*.o GPU/*.o
	@rmdir enc unenc wght

.PHONY: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help