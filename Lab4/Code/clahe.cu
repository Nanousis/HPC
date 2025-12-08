#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "clahe.h"
#include <cuda_runtime.h>

#define RED     "\033[1;31m"
#define GREEN   "\033[1;32m"
#define YELLOW  "\033[1;33m"
#define BLUE    "\033[1;34m"
#define MAGENTA "\033[1;35m"
#define CYAN    "\033[1;36m"
#define RESET   "\033[0m"
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, RED "CUDA Error: %s (err_num=%d) at %s:%d\n" RESET, \
                cudaGetErrorString(err), err, __FILE__, __LINE__);    \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)



// Helper: Read PGM
PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    PGM_IMG result;
    int v_max;

    in_file = fopen(path, "rb");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip P5*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d",&v_max);
    fgetc(in_file); // Skip the single whitespace/newline after max_val

    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    fread(result.img, sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    
    return result;
}

// Helper: Write PGM
void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n", img.w, img.h);
    fwrite(img.img, sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

// Helper: Free PGM Memory
void free_pgm(PGM_IMG img) {
    if(img.img) free(img.img);
}

// Compute & Clip Histogram for a specific tile
void compute_histogram(unsigned char* data, 
                       int w, int h, int start_x, int start_y, 
                       int tile_w, int tile_h, 
                       int* lut) {
    int hist[256] = {0};
    int x, y, i, avg_inc, val;
    int excess = 0, cdf = 0, total_pixels = tile_w * tile_h; 

    // Build Histogram
    for (y = start_y; y < start_y + tile_h; ++y) {
        for (x = start_x; x < start_x + tile_w; ++x) {
            // Boundary check mostly for the right/bottom edge tiles
            if(x < w && y < h) {
                hist[data[y * w + x]]++;
            }
        }
    }

    // Clip Histogram
    for (i = 0; i < 256; ++i) {
        if (hist[i] > CLIP_LIMIT) {
            excess += (hist[i] - CLIP_LIMIT);
            hist[i] = CLIP_LIMIT;
        }
    }

    // Redistribute Excess (simplisticly)
    avg_inc = excess / 256;
    for (i = 0; i < 256; ++i) {
        hist[i] += avg_inc;
    }
    
    // Compute CDF & LUT
    for (i = 0; i < 256; ++i) {
        cdf += hist[i];
        // Calculate equalized value
        val = (int)((float)cdf * 255.0f / total_pixels + 0.5f);
        if (val > 255) 
            val = 255;
        lut[i] = val;
    }
}

// One block per tile, 16x16 for 256 values of brightess
__global__ void compute_all_luts_kernel(const unsigned char* d_img,
                                        int image_w, int image_h,
                                        int* d_all_luts,
                                        int grid_w, int grid_h)
{
    const int tx = blockIdx.x;
    const int ty = blockIdx.y;
    const int x_start = tx * TILE_SIZE;
    const int y_start = ty * TILE_SIZE;

    const int tile_w = min(TILE_SIZE, max(0, image_w - x_start));
    const int tile_h = min(TILE_SIZE, max(0, image_h - y_start));
    if (tile_w <= 0 || tile_h <= 0) {
        return;
    }

    __shared__ unsigned int hist[256];
    __shared__ unsigned int excess;
    unsigned int avg_inc;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < 256) hist[tid] = 0u;
    if (tid == 0) { excess = 0u; avg_inc = 0u; }
    __syncthreads();

    // Build histogram with striding over tile
    for (int yy = threadIdx.y; yy < tile_h; yy += blockDim.y) {
        const int gy = y_start + yy;
        for (int xx = threadIdx.x; xx < tile_w; xx += blockDim.x) {
            const int gx = x_start + xx;
            const unsigned char pix = d_img[gy * image_w + gx];
            atomicAdd(&hist[(int)pix], 1u);
        }
    }
    __syncthreads();
    // Clip Histogram
    unsigned int h = hist[tid];
    if ((int)h > CLIP_LIMIT) {
        atomicAdd(&excess, h - (unsigned int)CLIP_LIMIT);
        hist[tid] = (unsigned int)CLIP_LIMIT;
    }

    __syncthreads();
    // technically this can be done by one thread but then synchgronization is needed
    avg_inc = excess / 256u;

    hist[tid] += avg_inc;
    __syncthreads();

    #ifndef SERIAL_CDF
    // This is a parallel exclusive scan to do the same thing we did in the presentations
    __shared__ unsigned int cdf_s[256];

    cdf_s[tid] = hist[tid];
    __syncthreads();

    for (int stride = 1; stride < 256; stride <<= 1) {
        int idx = ((tid + 1) << 1) * stride - 1;
        if (idx < 256)
            cdf_s[idx] += cdf_s[idx - stride];
        __syncthreads();
    }

    if (tid == 255) cdf_s[255] = 0;
    __syncthreads();

    for (int stride = 128; stride >= 1; stride >>= 1) {
        int idx = ((tid + 1) << 1) * stride - 1;
        if (idx < 256) {
            unsigned int t = cdf_s[idx - stride];
            cdf_s[idx - stride] = cdf_s[idx];
            cdf_s[idx] += t;
        }
        __syncthreads();
    }

    // Inclusive CDF = exclusive prefix + current bin
    unsigned int cdf_inclusive = cdf_s[tid] + hist[tid];

    const int total_pixels = tile_w * tile_h;
    const int lut_base = (ty * grid_w + tx) * 256;

    int val = (int)((double)cdf_inclusive * 255.0 / (double)total_pixels + 0.5);
    if (val > 255) val = 255;
    d_all_luts[lut_base + tid] = val;
    #else
    // Serialk Compute CDF and LUT
    if (tid == 0) {
        const int total_pixels = tile_w * tile_h;
        int cdf = 0;
        const int lut_base = (ty * grid_w + tx) * 256;
        for (int i = 0; i < 256; ++i) {
            cdf += (int)hist[i];
            int val = (int)((float)cdf * 255.0f / (float)total_pixels + 0.5f);
            if (val > 255) val = 255;
            d_all_luts[lut_base + i] = val;
        }
    }
    #endif
}
// Each thread maps one output pixel using the four neighboring tile LUTs
__global__ void render_image(const unsigned char* __restrict__ d_img_in,
                             unsigned char* __restrict__ d_img_out,
                             int w, int h,
                             const int* __restrict__ d_all_luts,
                             int grid_w, int grid_h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    // Center-aligned tile coordinates (match the CPU code)
    const float ty_f = (float)y / (float)TILE_SIZE - 0.5f;
    const float tx_f = (float)x / (float)TILE_SIZE - 0.5f;

    int y1 = (int)floorf(ty_f);
    int x1 = (int)floorf(tx_f);
    int y2 = y1 + 1;
    int x2 = x1 + 1;

    float y_weight = ty_f - (float)y1;
    float x_weight = tx_f - (float)x1;

    // Clamp tile indices to grid
    if (x1 < 0)
        x1 = 0;
    if (y1 < 0)
        y1 = 0;
    if (x2 >= grid_w)
      x2 = grid_w - 1;
    if (y2 >= grid_h)
        y2 = grid_h - 1;

    const unsigned char val = d_img_in[y * w + x];

    const int tile_idx = y1 * grid_w;
    const int tile_idx2 = y2 * grid_w;

    // Fetch mapped values from the 4 nearest tile LUTs
    const int addr_tl = (tile_idx + x1) << 8;
    const int addr_tr = (tile_idx + x2) << 8;
    const int addr_bl = (tile_idx2 + x1) << 8;
    const int addr_br = (tile_idx2 + x2) << 8;

    const float tl = (float)d_all_luts[addr_tl + val];
    const float tr = (float)d_all_luts[addr_tr + val];
    const float bl = (float)d_all_luts[addr_bl + val];
    const float br = (float)d_all_luts[addr_br + val];

    const float top = tl * (1.0f - x_weight) + tr * x_weight;
    const float bot = bl * (1.0f - x_weight) + br * x_weight;
    float final_val = top * (1.0f - y_weight) + bot * y_weight;

    int out = (int)(final_val + 0.5f);
    d_img_out[y * w + x] = (unsigned char)out;
}

extern double get_time_sec();

// Core CLAHE
PGM_IMG apply_clahe(PGM_IMG h_img_in) {


    printf("-------------------------\n");
    // Query CUDA devices
    {
        int devCount = 0;
        cudaError_t cerr = cudaGetDeviceCount(&devCount);
        if (cerr != cudaSuccess) {
            printf("CUDA query error: %s\n", cudaGetErrorString(cerr));
        } else if (devCount == 0) {
            printf("No CUDA devices found\n");
        } else {
            printf("Found %d CUDA device(s)\n", devCount);
            for (int d = 0; d < devCount; ++d) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, d);
                printf("CUDA Device %d: %s\n", d, prop.name);
                printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
                printf("  Total Global Mem: %zu bytes\n", prop.totalGlobalMem);
                printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
                printf("  Max threads / block: %d\n", prop.maxThreadsPerBlock);
                printf("  Max grid dim: %d x %d x %d\n",
                        prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
                printf("  Shared mem / block: %zu bytes\n", prop.sharedMemPerBlock);
                printf("  L2 cache size: %d\n", prop.l2CacheSize);
            }
        }
    }
    printf("-------------------------\n");

    PGM_IMG d_img_out;
    PGM_IMG h_img_out;
    PGM_IMG d_img_in;
    int w = h_img_in.w;
    int h = h_img_in.h;
    int grid_w, grid_h;
    int *h_all_luts, *d_all_luts; // Big array to store LUTs for all tiles
    int* current_lut_ptr;
    int ty, tx, x, y, x1, x2, y1, y2, tl, tr, bl, br, val;
    int x_start, y_start, actual_tile_w, actual_tile_h;
    float tx_f, ty_f, x_weight, y_weight, top, bot, final_val;
    // Allocate output image
    h_img_out.w = w;
    h_img_out.h = h;
    h_img_out.img = (unsigned char *)malloc(w * h * sizeof(unsigned char));
    
    
    double start_time = get_time_sec();
    d_img_out.w = w;
    d_img_out.h = h;
    CUDA_CHECK(cudaMalloc((void**)&d_img_out.img, w * h * sizeof(unsigned char)));

    d_img_in.w = w;
    d_img_in.h = h;
    CUDA_CHECK(cudaMalloc((void**)&d_img_in.img, w * h * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_img_in.img, h_img_in.img, w * h * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Calculate grid dimensions
    grid_w = (w + TILE_SIZE - 1) / TILE_SIZE;
    grid_h = (h + TILE_SIZE - 1) / TILE_SIZE;
    
    printf("Width: %d, Height: %d, grid_w: %d, grid_h: %d\n", w, h, grid_w, grid_h);
    // Allocate memory for all LUTs: [grid_h][grid_w][256],
    // as an 1D array
    h_all_luts = (int *)malloc(grid_w * grid_h * 256 * sizeof(int));
    // allocate device LUTs (already in your code)
    CUDA_CHECK(cudaMalloc((void**)&d_all_luts, grid_w * grid_h * 256 * sizeof(int)));

    // for 256 brightness levels
    dim3 lutBlock(16, 16);
    dim3 lutGrid(grid_w, grid_h);

    start_time = get_time_sec();
    compute_all_luts_kernel<<<lutGrid, lutBlock>>>(d_img_in.img, w, h, d_all_luts, grid_w, grid_h);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    double precomp_time = get_time_sec() - start_time;
    
    // CUDA_CHECK(cudaMemcpy(h_all_luts,
    //                 d_all_luts,
    //                 grid_w * grid_h * 256 * sizeof(int),
    //                 cudaMemcpyDeviceToHost));
    // for(int i=0; i<10; i++) {
    //     printf("LUT Sample %d: %d\n", i, h_all_luts[i]);
    // }

    // now render
    start_time = get_time_sec();
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    render_image<<<grid, block>>>(d_img_in.img, d_img_out.img, w, h, d_all_luts, grid_w, grid_h);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    double render_time = get_time_sec() - start_time;

    // copy out
    CUDA_CHECK(cudaMemcpy(h_img_out.img, d_img_out.img, w * h, cudaMemcpyDeviceToHost));
    printf("Precomputation Time: %f seconds\n", precomp_time);
    printf("Rendering Time: %f seconds\n", render_time);

    // cleanup
    CUDA_CHECK(cudaFree(d_all_luts));
    CUDA_CHECK(cudaFree(d_img_in.img));
    CUDA_CHECK(cudaFree(d_img_out.img));
    free(h_all_luts);
    return h_img_out;
}
