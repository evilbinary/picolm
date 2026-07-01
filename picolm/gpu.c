#include "gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_VULKAN

#include <vulkan/vulkan.h>

/* ---- Convenience macros ---- */
#define VK_CHECK(x) do { \
    VkResult __vr = (x); \
    if (__vr != VK_SUCCESS) { \
        fprintf(stderr, "Vulkan error %d at %s:%d\n", __vr, __FILE__, __LINE__); \
        return -1; \
    } \
} while(0)

/* ---- Internal state ---- */

#define MAX_GPU_WEIGHTS 256

typedef struct {
    VkBuffer       buffer;
    VkDeviceMemory memory;
    const void    *cpu_ptr;      /* lookup key (mmap address) */
    int            n, d;         /* columns, rows */
    gguf_type_t    qtype;
    size_t         row_bytes;
    size_t         total_bytes;
} gpu_weight_t;

static struct {
    /* Core Vulkan */
    VkInstance                    instance;
    VkPhysicalDevice              phys_dev;
    VkDevice                      device;
    uint32_t                      queue_family;
    VkQueue                       queue;

    /* Memory properties */
    VkPhysicalDeviceMemoryProperties mem_props;

    /* Command */
    VkCommandPool                 cmd_pool;
    VkCommandBuffer               cmd_buf;
    VkFence                       fence;

    /* Descriptor */
    VkDescriptorSetLayout         desc_layout;
    VkDescriptorPool              desc_pool;
    VkDescriptorSet               desc_set;

    /* Pipeline */
    VkPipelineLayout              pipe_layout;
    VkPipeline                    pipe_q4k;
    VkPipeline                    pipe_q6k;
    VkPipeline                    pipe_q80;

    /* Staging buffers (host-visible, persistently mapped) */
    VkBuffer                      staging_x_buf;
    VkDeviceMemory                staging_x_mem;
    float                        *staging_x_ptr;
    size_t                        staging_x_cap;  /* in floats */

    VkBuffer                      staging_out_buf;
    VkDeviceMemory                staging_out_mem;
    float                        *staging_out_ptr;
    size_t                        staging_out_cap;

    VkBuffer                      staging_w_buf;
    VkDeviceMemory                staging_w_mem;
    void                         *staging_w_ptr;
    size_t                        staging_w_cap;  /* in bytes */

    /* Weights */
    gpu_weight_t                  weights[MAX_GPU_WEIGHTS];
    int                           n_weights;
    int                           gpu_layers;  /* how many layers on GPU (0 = all) */

    int                           initialized;
} gs;

/* ---- Helper: find memory type ---- */

static int gpu_find_mem_type(uint32_t type_filter, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < gs.mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1u << i)) &&
            (gs.mem_props.memoryTypes[i].propertyFlags & props) == props)
            return (int)i;
    }
    return -1;
}

/* ---- Helper: create a buffer ---- */

static int gpu_create_buffer(size_t size, VkBufferUsageFlags usage,
                             VkMemoryPropertyFlags props,
                             VkBuffer *buf, VkDeviceMemory *mem) {
    VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(gs.device, &bi, NULL, buf));

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(gs.device, *buf, &mr);

    int mtype = gpu_find_mem_type(mr.memoryTypeBits, props);
    if (mtype < 0) {
        fprintf(stderr, "Failed to find suitable memory type\n");
        vkDestroyBuffer(gs.device, *buf, NULL);
        return -1;
    }

    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = (uint32_t)mtype;
    VK_CHECK(vkAllocateMemory(gs.device, &ai, NULL, mem));

    VK_CHECK(vkBindBufferMemory(gs.device, *buf, *mem, 0));
    return 0;
}

/* ---- Helper: load SPIR-V shader ---- */

static VkShaderModule gpu_load_shader(const unsigned char *spv, size_t size) {
    VkShaderModuleCreateInfo ci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    ci.codeSize = size;
    ci.pCode = (const uint32_t *)spv;
    VkShaderModule mod;
    if (vkCreateShaderModule(gs.device, &ci, NULL, &mod) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return mod;
}

/* ---- Init ---- */

int gpu_init(const model_t *m) {
    memset(&gs, 0, sizeof(gs));

    /* Layer offloading: default = all layers. Set PICOLM_GPU_LAYERS=N to limit. */
    gs.gpu_layers = m->config.n_layers;
    const char *layers_env = getenv("PICOLM_GPU_LAYERS");
    if (layers_env) {
        int n = atoi(layers_env);
        if (n > 0 && n <= m->config.n_layers)
            gs.gpu_layers = n;
    }

    /* Estimate total weight size for VRAM check (only gpu_layers layers) */
    size_t total_weights = 0;
    int layers_to_upload = gs.gpu_layers;
    for (int l = 0; l < layers_to_upload && l < m->config.n_layers; l++) {
        const layer_weights_t *lw = &m->weights.layers[l];
        #define ADD_WEIGHT(ptr, qtype, dim_out, dim_in) do { \
            if (ptr && (qtype) != GGUF_TYPE_F32) { \
                total_weights += gguf_type_row_size(qtype, dim_in) * (size_t)(dim_out); \
            } \
        } while(0)
        ADD_WEIGHT(lw->attn_q,      lw->type_attn_q,      m->config.n_embd, m->config.n_embd);
        ADD_WEIGHT(lw->attn_k,      lw->type_attn_k,      m->config.n_kv_heads * m->config.head_dim, m->config.n_embd);
        ADD_WEIGHT(lw->attn_v,      lw->type_attn_v,      m->config.n_kv_heads * m->config.head_dim, m->config.n_embd);
        ADD_WEIGHT(lw->attn_output, lw->type_attn_output, m->config.n_embd, m->config.n_heads * m->config.head_dim);
        ADD_WEIGHT(lw->ffn_gate,    lw->type_ffn_gate,    m->config.n_ffn, m->config.n_embd);
        ADD_WEIGHT(lw->ffn_up,      lw->type_ffn_up,      m->config.n_ffn, m->config.n_embd);
        ADD_WEIGHT(lw->ffn_down,    lw->type_ffn_down,    m->config.n_embd, m->config.n_ffn);
        #undef ADD_WEIGHT
    }
    if (gs.gpu_layers > 0 &&
        m->weights.output && m->weights.type_output != GGUF_TYPE_F32)
        total_weights += gguf_type_row_size(m->weights.type_output, m->config.n_embd)
                         * (size_t)m->config.vocab_size;

    fprintf(stderr, "GPU: ~%.1f MB model weights to upload (%d/%d layers)\n",
            (double)total_weights / (1024.0 * 1024.0),
            gs.gpu_layers, m->config.n_layers);

    /* ---- Instance ---- */
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "picolm";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ic = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ic.pApplicationInfo = &app;
    /* No layers, no extensions — minimal instance */
    if (vkCreateInstance(&ic, NULL, &gs.instance) != VK_SUCCESS) {
        fprintf(stderr, "GPU: vkCreateInstance failed (no Vulkan driver?)\n");
        return -1;
    }

    /* ---- Physical device ---- */
    uint32_t nphys = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(gs.instance, &nphys, NULL));
    if (nphys == 0) {
        fprintf(stderr, "GPU: no Vulkan physical devices found\n");
        vkDestroyInstance(gs.instance, NULL);
        return -1;
    }

    VkPhysicalDevice *phys = (VkPhysicalDevice *)malloc(nphys * sizeof(VkPhysicalDevice));
    VK_CHECK(vkEnumeratePhysicalDevices(gs.instance, &nphys, phys));

    /* Prefer discrete GPU */
    int chosen = -1;
    for (uint32_t i = 0; i < nphys; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(phys[i], &props);
        uint32_t nqf = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys[i], &nqf, NULL);
        VkQueueFamilyProperties *qf = (VkQueueFamilyProperties *)malloc(nqf * sizeof(VkQueueFamilyProperties));
        vkGetPhysicalDeviceQueueFamilyProperties(phys[i], &nqf, qf);
        for (uint32_t q = 0; q < nqf; q++) {
            if (qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                if (chosen < 0 || props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                    chosen = (int)i;
                    gs.queue_family = q;
                }
                break;
            }
        }
        free(qf);
        if (chosen == (int)i && props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            break;  /* prefer discrete */
    }

    if (chosen < 0) {
        fprintf(stderr, "GPU: no device with compute queue found\n");
        free(phys);
        vkDestroyInstance(gs.instance, NULL);
        return -1;
    }

    gs.phys_dev = phys[chosen];
    VkPhysicalDeviceProperties dev_props;
    vkGetPhysicalDeviceProperties(gs.phys_dev, &dev_props);
    vkGetPhysicalDeviceMemoryProperties(gs.phys_dev, &gs.mem_props);

    /* Skip integrated GPUs by default — they're slower than CPU for this workload.
     * Set PICOLM_GPU_FORCE=1 to override. */
    if (dev_props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        const char *force = getenv("PICOLM_GPU_FORCE");
        if (!force || force[0] != '1') {
            fprintf(stderr, "GPU: %s (integrated), skipping — set PICOLM_GPU_FORCE=1 to use it\n",
                    dev_props.deviceName);
            free(phys);
            vkDestroyInstance(gs.instance, NULL);
            return -1;
        }
        fprintf(stderr, "GPU: %s (integrated, forced)\n", dev_props.deviceName);
    } else {
        fprintf(stderr, "GPU: %s\n", dev_props.deviceName);
    }

    /* VRAM check */
    VkDeviceSize vram_total = 0;
    for (uint32_t i = 0; i < gs.mem_props.memoryHeapCount; i++) {
        if (gs.mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            vram_total += gs.mem_props.memoryHeaps[i].size;
    }
    /* Staging: x (max n_embd/n_ffn) + out (max vocab_size) + weight upload (largest tensor) */
    size_t max_dim = m->config.n_embd > m->config.n_ffn ? m->config.n_embd : m->config.n_ffn;
    if (m->config.vocab_size > (int)max_dim) max_dim = m->config.vocab_size;
    size_t staging_needed = (max_dim * sizeof(float) * 2) + (total_weights / 4); /* rough */
    if (total_weights + staging_needed > (size_t)vram_total) {
        fprintf(stderr, "GPU: model %.1f MB + staging > VRAM %.0f MB\n",
                (double)total_weights / (1024.0 * 1024.0),
                (double)vram_total / (1024.0 * 1024.0));
        free(phys);
        vkDestroyInstance(gs.instance, NULL);
        return -1;
    }

    /* ---- Logical device ---- */
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = gs.queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;

    VK_CHECK(vkCreateDevice(gs.phys_dev, &dci, NULL, &gs.device));

    vkGetDeviceQueue(gs.device, gs.queue_family, 0, &gs.queue);

    /* ---- Command pool + buffer ---- */
    VkCommandPoolCreateInfo cpci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = gs.queue_family;
    VK_CHECK(vkCreateCommandPool(gs.device, &cpci, NULL, &gs.cmd_pool));

    VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool = gs.cmd_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(gs.device, &cbai, &gs.cmd_buf));

    /* ---- Fence (signaled initially) ---- */
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_CHECK(vkCreateFence(gs.device, &fci, NULL, &gs.fence));

    /* ---- Descriptor set layout: 3 bindings (x, W, out) ---- */
    VkDescriptorSetLayoutBinding bindings[3] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
    };
    VkDescriptorSetLayoutCreateInfo dslci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dslci.bindingCount = 3;
    dslci.pBindings = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(gs.device, &dslci, NULL, &gs.desc_layout));

    /* ---- Descriptor pool + set ---- */
    VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &pool_size;
    VK_CHECK(vkCreateDescriptorPool(gs.device, &dpci, NULL, &gs.desc_pool));

    VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = gs.desc_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &gs.desc_layout;
    VK_CHECK(vkAllocateDescriptorSets(gs.device, &dsai, &gs.desc_set));

    /* ---- Pipeline layout (3 bindings + push constants) ---- */
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 12 }; /* n, d, row_bytes */
    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &gs.desc_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(gs.device, &plci, NULL, &gs.pipe_layout));

    /* ---- Staging buffers (allocate on first use) ---- */
    gs.staging_x_cap = 0;
    gs.staging_out_cap = 0;
    gs.staging_w_cap = 0;

    free(phys);

    gs.initialized = 1;
    fprintf(stderr, "GPU: Vulkan init OK (%d/%d layers)\n", gs.gpu_layers, m->config.n_layers);
    return 0;
}

/* ---- Shader embed (compiled SPIR-V from separate files at build time) ---- */

/* These are declared in the .spv files linked at build time.
 * The Makefile compiles .comp -> .spv, then extracts as .c arrays. */

/* For now we load from external SPIR-V files at runtime.
 * The spv directory must be discoverable. */

/* We'll use a simple file-load approach instead of linking arrays. */

static int gpu_load_pipeline_from_file(const char *spv_path, VkPipeline *pipeline) {
    FILE *f = fopen(spv_path, "rb");
    if (!f) {
        fprintf(stderr, "GPU: cannot open %s\n", spv_path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || (sz & 3)) {
        fprintf(stderr, "GPU: invalid SPIR-V file %s\n", spv_path);
        fclose(f);
        return -1;
    }

    unsigned char *code = (unsigned char *)malloc((size_t)sz);
    fread(code, 1, (size_t)sz, f);
    fclose(f);

    VkShaderModule mod = gpu_load_shader(code, (size_t)sz);
    free(code);
    if (mod == VK_NULL_HANDLE) return -1;

    VkComputePipelineCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = mod;
    ci.stage.pName = "main";
    ci.layout = gs.pipe_layout;

    VK_CHECK(vkCreateComputePipelines(gs.device, VK_NULL_HANDLE, 1, &ci, NULL, pipeline));
    vkDestroyShaderModule(gs.device, mod, NULL);
    return 0;
}

static int gpu_load_pipelines(void) {
    /* Look for SPIR-V files relative to executable path.
     * For now, try "./shaders/spv/" relative to CWD. */
    if (gpu_load_pipeline_from_file("shaders/spv/q4_k.spv", &gs.pipe_q4k) != 0 &&
        gpu_load_pipeline_from_file("picolm/shaders/spv/q4_k.spv", &gs.pipe_q4k) != 0) {
        fprintf(stderr, "GPU: cannot load q4_k.spv shader\n");
        return -1;
    }
    if (gpu_load_pipeline_from_file("shaders/spv/q6_k.spv", &gs.pipe_q6k) != 0 &&
        gpu_load_pipeline_from_file("picolm/shaders/spv/q6_k.spv", &gs.pipe_q6k) != 0) {
        fprintf(stderr, "GPU: cannot load q6_k.spv shader\n");
        return -1;
    }
    if (gpu_load_pipeline_from_file("shaders/spv/q8_0.spv", &gs.pipe_q80) != 0 &&
        gpu_load_pipeline_from_file("picolm/shaders/spv/q8_0.spv", &gs.pipe_q80) != 0) {
        fprintf(stderr, "GPU: cannot load q8_0.spv shader\n");
        return -1;
    }
    return 0;
}

/* ---- Upload model weights ---- */

/* Forward declaration */
static int gpu_upload_tensor(const void *cpu_ptr, gguf_type_t qtype, int d, int n);

int gpu_upload_model(const model_t *m) {
    if (!gs.initialized) return -1;

    /* Load shaders first */
    if (gpu_load_pipelines() != 0) {
        fprintf(stderr, "GPU: shader loading failed, disabling GPU\n");
        gpu_shutdown();
        return -1;
    }

    /* We'll iterate through all weight tensors and upload each quantized one.
     * Helper macro to upload a single tensor. */
    #define UPLOAD(ptr, qtype, dim_out, dim_in) do { \
        if (ptr && (qtype) != GGUF_TYPE_F32 && (qtype) != GGUF_TYPE_F16) { \
            if (gpu_upload_tensor(ptr, qtype, dim_out, dim_in) != 0) { \
                fprintf(stderr, "GPU: warning: failed to upload tensor\n"); \
            } \
        } \
    } while(0)

    for (int l = 0; l < gs.gpu_layers && l < m->config.n_layers; l++) {
        const layer_weights_t *lw = &m->weights.layers[l];
        UPLOAD(lw->attn_q,      lw->type_attn_q,      m->config.n_embd, m->config.n_embd);
        UPLOAD(lw->attn_k,      lw->type_attn_k,      m->config.n_kv_heads * m->config.head_dim, m->config.n_embd);
        UPLOAD(lw->attn_v,      lw->type_attn_v,      m->config.n_kv_heads * m->config.head_dim, m->config.n_embd);
        UPLOAD(lw->attn_output, lw->type_attn_output, m->config.n_embd, m->config.n_heads * m->config.head_dim);
        UPLOAD(lw->ffn_gate,    lw->type_ffn_gate,    m->config.n_ffn, m->config.n_embd);
        UPLOAD(lw->ffn_up,      lw->type_ffn_up,      m->config.n_ffn, m->config.n_embd);
        UPLOAD(lw->ffn_down,    lw->type_ffn_down,    m->config.n_embd, m->config.n_ffn);
    }
    if (gs.gpu_layers > 0)
        UPLOAD(m->weights.output, m->weights.type_output, m->config.vocab_size, m->config.n_embd);

    #undef UPLOAD

    fprintf(stderr, "GPU: uploaded %d tensors\n", gs.n_weights);
    return 0;
}

/* Upload a single weight tensor to GPU VRAM */
static int gpu_upload_tensor(const void *cpu_ptr, gguf_type_t qtype,
                              int d, int n) {
    if (gs.n_weights >= MAX_GPU_WEIGHTS) return -1;

    size_t row_bytes = gguf_type_row_size(qtype, n);
    size_t total_bytes = row_bytes * (size_t)d;
    if (total_bytes == 0) return -1;

    /* Allocate device-local buffer */
    VkBuffer buf;
    VkDeviceMemory mem;
    if (gpu_create_buffer(total_bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &buf, &mem) != 0)
        return -1;

    /* Staging: host-visible, copy data, submit transfer */
    if (gs.staging_w_cap < total_bytes) {
        /* Free old staging if exists */
        if (gs.staging_w_buf) {
            vkDestroyBuffer(gs.device, gs.staging_w_buf, NULL);
            vkFreeMemory(gs.device, gs.staging_w_mem, NULL);
        }
        if (gpu_create_buffer(total_bytes,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &gs.staging_w_buf, &gs.staging_w_mem) != 0) {
            vkDestroyBuffer(gs.device, buf, NULL);
            vkFreeMemory(gs.device, mem, NULL);
            return -1;
        }
        VK_CHECK(vkMapMemory(gs.device, gs.staging_w_mem, 0, total_bytes, 0,
                              (void **)&gs.staging_w_ptr));
        gs.staging_w_cap = total_bytes;
    }

    /* Copy data to staging */
    memcpy(gs.staging_w_ptr, cpu_ptr, total_bytes);

    /* Submit copy command */
    VkCommandBufferBeginInfo begin = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(gs.cmd_buf, 0);
    vkBeginCommandBuffer(gs.cmd_buf, &begin);

    VkBufferCopy copy = { 0, 0, total_bytes };
    vkCmdCopyBuffer(gs.cmd_buf, gs.staging_w_buf, buf, 1, &copy);
    vkEndCommandBuffer(gs.cmd_buf);

    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &gs.cmd_buf;
    vkResetFences(gs.device, 1, &gs.fence);
    VK_CHECK(vkQueueSubmit(gs.queue, 1, &si, gs.fence));
    vkWaitForFences(gs.device, 1, &gs.fence, VK_TRUE, UINT64_MAX);

    /* Store in lookup table */
    gpu_weight_t *gw = &gs.weights[gs.n_weights++];
    gw->buffer = buf;
    gw->memory = mem;
    gw->cpu_ptr = cpu_ptr;
    gw->n = n;
    gw->d = d;
    gw->qtype = qtype;
    gw->row_bytes = row_bytes;
    gw->total_bytes = total_bytes;

    return 0;
}

/* ---- Matmul dispatch ---- */

int gpu_available(void) {
    return gs.initialized;
}

static int gpu_ensure_staging(size_t x_needed, size_t out_needed) {
    /* x staging */
    if (x_needed > gs.staging_x_cap) {
        if (gs.staging_x_buf) {
            vkDestroyBuffer(gs.device, gs.staging_x_buf, NULL);
            vkFreeMemory(gs.device, gs.staging_x_mem, NULL);
        }
        size_t bytes = x_needed * sizeof(float);
        if (gpu_create_buffer(bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &gs.staging_x_buf, &gs.staging_x_mem) != 0)
            return -1;
        VK_CHECK(vkMapMemory(gs.device, gs.staging_x_mem, 0, bytes, 0,
                              (void **)&gs.staging_x_ptr));
        gs.staging_x_cap = x_needed;
    }
    /* out staging */
    if (out_needed > gs.staging_out_cap) {
        if (gs.staging_out_buf) {
            vkDestroyBuffer(gs.device, gs.staging_out_buf, NULL);
            vkFreeMemory(gs.device, gs.staging_out_mem, NULL);
        }
        size_t bytes = out_needed * sizeof(float);
        if (gpu_create_buffer(bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &gs.staging_out_buf, &gs.staging_out_mem) != 0)
            return -1;
        VK_CHECK(vkMapMemory(gs.device, gs.staging_out_mem, 0, bytes, 0,
                              (void **)&gs.staging_out_ptr));
        gs.staging_out_cap = out_needed;
    }
    return 0;
}

int gpu_try_matmul(float *out, const float *x, const void *W,
                    int n, int d, gguf_type_t qtype) {
    if (!gs.initialized) return -1;

    /* Lookup weight by CPU pointer */
    int idx = -1;
    for (int i = 0; i < gs.n_weights; i++) {
        if (gs.weights[i].cpu_ptr == W) { idx = i; break; }
    }
    if (idx < 0) {
        fprintf(stderr, "GPU: weight ptr %p not found, falling back to CPU\n", W);
        return -1;
    }

    gpu_weight_t *gw = &gs.weights[idx];

    /* Select pipeline */
    VkPipeline pipeline;
    switch (qtype) {
        case GGUF_TYPE_Q4_K: pipeline = gs.pipe_q4k; break;
        case GGUF_TYPE_Q6_K: pipeline = gs.pipe_q6k; break;
        case GGUF_TYPE_Q8_0: pipeline = gs.pipe_q80; break;
        default:
            fprintf(stderr, "GPU: unsupported qtype %d\n", qtype);
            return -1;
    }

    /* Ensure staging buffers */
    if (gpu_ensure_staging((size_t)n, (size_t)d) != 0) return -1;

    /* Copy input x to staging */
    memcpy(gs.staging_x_ptr, x, (size_t)n * sizeof(float));

    /* Wait for previous work */
    vkWaitForFences(gs.device, 1, &gs.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(gs.device, 1, &gs.fence);

    fprintf(stderr, "GPU: matmul n=%d d=%d qtype=%d row_bytes=%zu groups=%d\n",
            n, d, (int)qtype, gw->row_bytes, (d + 255) / 256);

    /* Update descriptor set */
    VkDescriptorBufferInfo x_info   = { gs.staging_x_buf, 0, (size_t)n * sizeof(float) };
    VkDescriptorBufferInfo W_info   = { gw->buffer, 0, gw->total_bytes };
    VkDescriptorBufferInfo out_info = { gs.staging_out_buf, 0, (size_t)d * sizeof(float) };

    VkWriteDescriptorSet writes[3];
    memset(writes, 0, sizeof(writes));
    for (int i = 0; i < 3; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = gs.desc_set;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    writes[0].pBufferInfo = &x_info;
    writes[1].pBufferInfo = &W_info;
    writes[2].pBufferInfo = &out_info;
    vkUpdateDescriptorSets(gs.device, 3, writes, 0, NULL);

    /* Build command buffer */
    VkCommandBufferBeginInfo begin = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(gs.cmd_buf, 0);
    vkBeginCommandBuffer(gs.cmd_buf, &begin);

    /* Barrier: x staging host write -> shader read */
    VkBufferMemoryBarrier x_bar = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
    x_bar.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    x_bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    x_bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    x_bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    x_bar.buffer = gs.staging_x_buf;
    x_bar.offset = 0;
    x_bar.size = (size_t)n * sizeof(float);

    vkCmdPipelineBarrier(gs.cmd_buf,
        VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, NULL, 1, &x_bar, 0, NULL);

    vkCmdBindPipeline(gs.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(gs.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            gs.pipe_layout, 0, 1, &gs.desc_set, 0, NULL);

    /* Push constants: n, d, row_bytes */
    struct { int n, d, row_bytes; } pc = { n, d, (int)gw->row_bytes };
    vkCmdPushConstants(gs.cmd_buf, gs.pipe_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    /* Dispatch: one workgroup per 256 rows */
    int groups_x = (d + 255) / 256;
    vkCmdDispatch(gs.cmd_buf, groups_x, 1, 1);

    /* Barrier: out staging shader write -> host read */
    VkBufferMemoryBarrier out_bar = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
    out_bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    out_bar.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    out_bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    out_bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    out_bar.buffer = gs.staging_out_buf;
    out_bar.offset = 0;
    out_bar.size = (size_t)d * sizeof(float);

    vkCmdPipelineBarrier(gs.cmd_buf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 0, NULL, 1, &out_bar, 0, NULL);

    vkEndCommandBuffer(gs.cmd_buf);

    /* Submit */
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &gs.cmd_buf;
    VK_CHECK(vkQueueSubmit(gs.queue, 1, &si, gs.fence));

    /* Wait with 5s timeout */
    VkResult fr = vkWaitForFences(gs.device, 1, &gs.fence, VK_TRUE, 5000000000ULL);
    if (fr != VK_SUCCESS) {
        fprintf(stderr, "GPU: fence wait failed (result=%d), falling back to CPU\n", (int)fr);
        return -1;
    }
    memcpy(out, gs.staging_out_ptr, (size_t)d * sizeof(float));

    return 0;
}

/* ---- Shutdown ---- */

void gpu_shutdown(void) {
    if (!gs.initialized) return;
    vkDeviceWaitIdle(gs.device);

    /* Destroy weight buffers */
    for (int i = 0; i < gs.n_weights; i++) {
        vkDestroyBuffer(gs.device, gs.weights[i].buffer, NULL);
        vkFreeMemory(gs.device, gs.weights[i].memory, NULL);
    }
    gs.n_weights = 0;

    /* Destroy staging buffers */
    if (gs.staging_x_buf) {
        vkUnmapMemory(gs.device, gs.staging_x_mem);
        vkDestroyBuffer(gs.device, gs.staging_x_buf, NULL);
        vkFreeMemory(gs.device, gs.staging_x_mem, NULL);
    }
    if (gs.staging_out_buf) {
        vkUnmapMemory(gs.device, gs.staging_out_mem);
        vkDestroyBuffer(gs.device, gs.staging_out_buf, NULL);
        vkFreeMemory(gs.device, gs.staging_out_mem, NULL);
    }
    if (gs.staging_w_buf) {
        vkUnmapMemory(gs.device, gs.staging_w_mem);
        vkDestroyBuffer(gs.device, gs.staging_w_buf, NULL);
        vkFreeMemory(gs.device, gs.staging_w_mem, NULL);
    }

    /* Destroy pipelines */
    if (gs.pipe_q4k) vkDestroyPipeline(gs.device, gs.pipe_q4k, NULL);
    if (gs.pipe_q6k) vkDestroyPipeline(gs.device, gs.pipe_q6k, NULL);
    if (gs.pipe_q80) vkDestroyPipeline(gs.device, gs.pipe_q80, NULL);

    if (gs.pipe_layout) vkDestroyPipelineLayout(gs.device, gs.pipe_layout, NULL);
    if (gs.desc_pool) vkDestroyDescriptorPool(gs.device, gs.desc_pool, NULL);
    if (gs.desc_layout) vkDestroyDescriptorSetLayout(gs.device, gs.desc_layout, NULL);
    if (gs.fence) vkDestroyFence(gs.device, gs.fence, NULL);
    if (gs.cmd_buf) vkFreeCommandBuffers(gs.device, gs.cmd_pool, 1, &gs.cmd_buf);
    if (gs.cmd_pool) vkDestroyCommandPool(gs.device, gs.cmd_pool, NULL);
    if (gs.device) vkDestroyDevice(gs.device, NULL);
    if (gs.instance) vkDestroyInstance(gs.instance, NULL);

    memset(&gs, 0, sizeof(gs));
}

#else  /* !USE_VULKAN */
/* Empty translation unit — GPU stubs are in gpu.h */
typedef int gpu_make_iso_compilers_happy;
#endif /* USE_VULKAN */
