CC      = gcc
CFLAGS  = -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -Wpedantic
LDFLAGS = -lm -lpthread
SRCS    = picolm.c model.c tensor.c quant.c tokenizer.c sampler.c grammar.c
TARGET  = picolm

PREFIX  ?= /usr/local
MODEL_DIR ?= /opt/picolm/models

# --- Native build (auto-detect CPU, enables SSE2/AVX/AVX2 on x86) ---
native: CFLAGS += -march=native
native: $(TARGET)

# --- x86-64 with AVX2 + FMA (explicit, for CPUs that support it) ---
native-avx2: CFLAGS += -O3 -mavx2 -mfma -mpopcnt
native-avx2: $(TARGET)

# --- Barrier pool mode (fused FFN gate+up, -DUSE_BARRIER_POOL) ---
barrier: CFLAGS += -march=native -DUSE_BARRIER_POOL
barrier: $(TARGET)

barrier-avx2: CFLAGS += -O3 -mavx2 -mfma -mpopcnt -DUSE_BARRIER_POOL
barrier-avx2: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# --- Vulkan Compute backend ---
# Runtime env vars:
#   PICOLM_GPU_FORCE=1   use integrated GPU (skipped by default)
#   PICOLM_GPU_LAYERS=N   offload only first N layers (default: all)
VULKAN_SDK_ROOT ?= E:/soft/VulkanSDK/1.4.350.0
VULKAN_SRCS = $(SRCS) gpu.c

vulkan: CFLAGS += -DUSE_VULKAN -I$(VULKAN_SDK_ROOT)/Include
vulkan: LDFLAGS += -L$(VULKAN_SDK_ROOT)/Lib -lvulkan-1
vulkan: shaders/spv/q4_k.spv shaders/spv/q6_k.spv shaders/spv/q8_0.spv
vulkan: $(VULKAN_SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(VULKAN_SRCS) $(LDFLAGS)

# Cross-platform: Linux (system-installed Vulkan)
vulkan-linux: CFLAGS += -DUSE_VULKAN
vulkan-linux: LDFLAGS += -lvulkan
vulkan-linux: shaders/spv/q4_k.spv shaders/spv/q6_k.spv shaders/spv/q8_0.spv
vulkan-linux: $(VULKAN_SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(VULKAN_SRCS) $(LDFLAGS)

vulkan-mac: CFLAGS += -DUSE_VULKAN
vulkan-mac: LDFLAGS += -lMoltenVK -framework Cocoa -framework IOKit
vulkan-mac: shaders/spv/q4_k.spv shaders/spv/q6_k.spv shaders/spv/q8_0.spv
vulkan-mac: $(VULKAN_SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(VULKAN_SRCS) $(LDFLAGS)

# --- Shader compilation ---
SHADER_COMPILER ?= "$(VULKAN_SDK_ROOT)/Bin/glslangValidator"
SHADER_SRCS = shaders/q4_k.comp shaders/q6_k.comp shaders/q8_0.comp
SHADER_SPV  = $(SHADER_SRCS:shaders/%.comp=shaders/spv/%.spv)

shaders/spv/%.spv: shaders/%.comp
	@mkdir -p shaders/spv
	$(SHADER_COMPILER) -V --target-env vulkan1.2 -o $@ $<

shaders: $(SHADER_SPV)

# --- Static build for single-binary deployment ---
static: CFLAGS += -march=native
static: LDFLAGS += -static
static: $(TARGET)

# --- Pi 3/4/5 (64-bit ARM with NEON SIMD) ---
pi: CFLAGS += -march=armv8-a+simd
pi: $(TARGET)

# --- Pi Zero / Pi 1 (32-bit ARM) ---
pi-arm32: CC = arm-linux-gnueabihf-gcc
pi-arm32: CFLAGS += -march=armv6 -mfpu=vfp
pi-arm32: $(TARGET)

# --- Cross-compile for Pi64 from platforms (e.g. x86) ---
cross-pi: CC = aarch64-linux-gnu-gcc
cross-pi: CFLAGS += -march=armv8-a+simd
cross-pi: LDFLAGS += -static
cross-pi: $(TARGET)

# --- RISC-V (Sipeed LicheeRV, etc.) ---
riscv: CC = riscv64-linux-gnu-gcc
riscv: CFLAGS += -march=rv64gc -mabi=lp64d
riscv: $(TARGET)

# --- Cross-compile for RISC-V from platforms (e.g. x86) ---
cross-riscv: CC = riscv64-linux-gnu-gcc
cross-riscv: CFLAGS += -march=rv64gc -mabi=lp64d
cross-riscv: LDFLAGS += -static
cross-riscv: $(TARGET)

# --- Debug build ---
debug: CFLAGS = -std=c11 -Wall -Wextra -Wpedantic -g -O0
debug: $(TARGET)

# --- Install ---
install: $(TARGET)
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(TARGET) $(DESTDIR)$(PREFIX)/bin/
	@echo "Installed $(TARGET) to $(DESTDIR)$(PREFIX)/bin/"

# --- Download default model ---
model:
	@mkdir -p $(MODEL_DIR)
	@if [ ! -f "$(MODEL_DIR)/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" ]; then \
		echo "Downloading TinyLlama 1.1B Q4_K_M (638 MB)..."; \
		curl -L -o "$(MODEL_DIR)/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
			"https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"; \
		echo "Done."; \
	else \
		echo "Model already exists at $(MODEL_DIR)/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"; \
	fi

clean:
	rm -f $(TARGET) $(TARGET).exe *.obj *.o

.PHONY: native native-avx2 static pi pi-arm32 cross-pi riscv cross-riscv debug install model clean
