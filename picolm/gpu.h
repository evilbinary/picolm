#ifndef GPU_H
#define GPU_H

#include "model.h"
#include <stddef.h>

/* ---- Lifecycle ---- */

/* Initialize Vulkan. Select first discrete GPU with compute support.
 * Checks VRAM against total model weight size.
 * Returns 0 on success; on failure gpu_available() returns 0. */
int gpu_init(const model_t *m);

/* Upload all quantized weight tensors to GPU VRAM.
 * Called once after model_load() + gpu_init().
 * Builds internal cpu_ptr->VkBuffer lookup table.
 * Returns 0 on success; -1 on failure (safe to ignore). */
int gpu_upload_model(const model_t *m);

/* Returns 1 if Vulkan is initialized and ready. */
int gpu_available(void);

/* Try to compute out[d] = W[d,n] @ x[n] on GPU.
 * Returns 0 on success; -1 if W not found (caller falls through to CPU). */
int gpu_try_matmul(float *out, const float *x, const void *W,
                   int n, int d, gguf_type_t qtype);

/* Drain pending GPU work and release all resources. */
void gpu_shutdown(void);

/* ---- Stubs when USE_VULKAN is not defined ---- */

#ifndef USE_VULKAN
static inline int  gpu_init(const model_t *m)      { (void)m; return -1; }
static inline int  gpu_upload_model(const model_t *m) { (void)m; return -1; }
static inline int  gpu_available(void)               { return 0; }
static inline int  gpu_try_matmul(float *o, const float *x, const void *W,
                                   int n, int d, gguf_type_t qt) {
    (void)o; (void)x; (void)W; (void)n; (void)d; (void)qt; return -1;
}
static inline void gpu_shutdown(void) {}
#endif

#endif /* GPU_H */
