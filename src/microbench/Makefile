# Step 2.1 micro-bench build.
# Targets land in ../../build/ so verify_sass.sh can find them at a stable path.

NVCC      ?= nvcc
ARCH      ?= sm_90a
BUILD_DIR  = $(abspath ../../build)
SRC_DIR    = $(abspath .)

NVCC_FLAGS = -arch=$(ARCH) -O3 -std=c++17 \
             -Xcompiler -O3 --use_fast_math \
             --ptxas-options=-v -lineinfo \
             --expt-relaxed-constexpr

KERNELS    = kernel_pure_cd kernel_pure_tc
CUBINS     = $(addprefix $(BUILD_DIR)/,$(addsuffix .cubin,$(KERNELS)))
BIN        = $(BUILD_DIR)/bench_isolated

.PHONY: all clean sass
all: $(BUILD_DIR) $(CUBINS) $(BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.cubin: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) -cubin -o $@ $<

$(BIN): $(SRC_DIR)/bench_isolated.cu $(SRC_DIR)/kernel_pure_cd.cu $(SRC_DIR)/kernel_pure_tc.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ \
	    $(SRC_DIR)/bench_isolated.cu \
	    $(SRC_DIR)/kernel_pure_cd.cu \
	    $(SRC_DIR)/kernel_pure_tc.cu

clean:
	rm -f $(CUBINS) $(BIN)
