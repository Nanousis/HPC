#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "timer.h"
#include "omp.h"

#define SOFTENING 0.01f

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


#define ACCURACY 1e-2

typedef struct {
    float x, y, z, vx, vy, vz;
} Body;

/* Update a single galaxy. Parameters:
    - array of bodies
    - time step
    - number of bodies
*/
#define CUDA_BLOCK_SIZE 1024



__global__ void bodyForceKernel_many_systems(Body * p, float dt, int n) {
    // splits the tile since we have at most 1024 threads per block

    int inner_tile_id = blockIdx.y;
    int inside_tile_idx = threadIdx.x;
    int i = blockIdx.x;
    int j = inside_tile_idx + inner_tile_id * blockDim.x;

    int system_id = blockIdx.z;
    Body *system_ptr = &p[system_id * n];
    // printf("System ID: %d\n", system_id);

    __shared__ float total_Fx;
    __shared__ float total_Fy;
    __shared__ float total_Fz;
    // Initiating the Force accumulators
    __shared__ float Fx[CUDA_BLOCK_SIZE];
    __shared__ float Fy[CUDA_BLOCK_SIZE];
    __shared__ float Fz[CUDA_BLOCK_SIZE];
    float dx, dy, dz, distSqr, invDist, invDist3;

    dx = system_ptr[j].x - system_ptr[i].x;
    dy = system_ptr[j].y - system_ptr[i].y;
    dz = system_ptr[j].z - system_ptr[i].z;
    distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
    // inverse srt goes brrrrrrrr here
    // invDist = 1.0f / sqrtf(distSqr);
    invDist = rsqrtf(distSqr);;
    invDist3 = invDist * invDist * invDist;
    Fx[inside_tile_idx] = dx * invDist3;
    Fy[inside_tile_idx] = dy * invDist3;
    Fz[inside_tile_idx] = dz * invDist3;
    __syncthreads();

    // add the stuff inside each subtile
    for (unsigned int s = CUDA_BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (inside_tile_idx < s) {
            Fx[inside_tile_idx] += Fx[inside_tile_idx + s];
            Fy[inside_tile_idx] += Fy[inside_tile_idx + s];
            Fz[inside_tile_idx] += Fz[inside_tile_idx + s];
        }
        __syncthreads(); 
    }
    if(inside_tile_idx == 0){
        atomicAdd(&p[i].vx, dt * Fx[0]);
    }
    if(inside_tile_idx == 1){
        atomicAdd(&p[i].vy, dt * Fy[0]);
    }
    if(inside_tile_idx == 2){
        atomicAdd(&p[i].vz, dt * Fz[0]);
    }

}
__global__ void integrateKernel_many_systems(Body * p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int system_id = blockIdx.z;
    Body *system_ptr = &p[system_id * n];
    if (i < n) {
        system_ptr[i].x += system_ptr[i].vx * dt;
        system_ptr[i].y += system_ptr[i].vy * dt;
        system_ptr[i].z += system_ptr[i].vz * dt;
    }
}

__global__ void bodyForceKernel(Body * p, float dt, int n) {
    // splits the tile since we have at most 1024 threads per block
    int inner_tile_id = blockIdx.y;
    int inside_tile_idx = threadIdx.x;
    int i = blockIdx.x;
    int j = inside_tile_idx + inner_tile_id * blockDim.x;
    Body pi = p[i];
    Body pj = p[j];

    __shared__ float total_Fx;
    __shared__ float total_Fy;
    __shared__ float total_Fz;
    // Initiating the Force accumulators
    __shared__ float Fx[CUDA_BLOCK_SIZE];
    __shared__ float Fy[CUDA_BLOCK_SIZE];
    __shared__ float Fz[CUDA_BLOCK_SIZE];
    float dx, dy, dz, distSqr, invDist, invDist3;
    dx = pj.x - pi.x;
    dy = pj.y - pi.y;
    dz = pj.z - pi.z;
    distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
    // inverse srt goes brrrrrrrr here
    // invDist = 1.0f / sqrtf(distSqr);
    invDist = rsqrtf(distSqr);;
    invDist3 = invDist * invDist * invDist;
    Fx[inside_tile_idx] = dx * invDist3;
    Fy[inside_tile_idx] = dy * invDist3;
    Fz[inside_tile_idx] = dz * invDist3;
    __syncthreads();

    // add the stuff inside each subtile
    for (unsigned int s = CUDA_BLOCK_SIZE/2; s > 0; s >>= 1) {
        if(i==0){
            printf("s=%d, inside_tile_idx=%d, Fx=%f\n",s,inside_tile_idx,Fx[inside_tile_idx]);
        }
        if (inside_tile_idx < s) {
            Fx[inside_tile_idx] += Fx[inside_tile_idx + s];
            Fy[inside_tile_idx] += Fy[inside_tile_idx + s];
            Fz[inside_tile_idx] += Fz[inside_tile_idx + s];
        }
        // __syncthreads(); 
    }
    // if(inside_tile_idx == 0){
    //     atomicAdd(&p[i].vx, dt * Fx[0]);
    // }
    // if(inside_tile_idx == 1){
    //     atomicAdd(&p[i].vy, dt * Fy[0]);
    // }
    // if(inside_tile_idx == 2){
    //     atomicAdd(&p[i].vz, dt * Fz[0]);
    // }

}
__global__ void integrateKernel(Body * p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

void bodyForce(Body * p, float dt, int n) {
    int i, j;
    float Fx, Fy, Fz, dx, dy, dz, distSqr, invDist, invDist3;
    #pragma omp parallel for private(i,j,Fx,Fy,Fz,dx,dy,dz,distSqr,invDist,invDist3)
    for (i = 0; i < n; i++) {
	    Fx = 0.0f;
    	Fy = 0.0f;
    	Fz = 0.0f;

    	for (j = 0; j < n; j++) {
	        dx = p[j].x - p[i].x;
            dy = p[j].y - p[i].y;
            dz = p[j].z - p[i].z;
            distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            // inverse srt goes brrrrrrrr here
            invDist = 1.0f / sqrtf(distSqr);
            invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
}
/* Integrate positions.
    - array of bodies
    - time step
    - number of bodies
*/
            
void integrate(Body * p, float dt, int n) {
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < n; i++) {
	    p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}
float calc_max_err(Body *a, Body *b, int n) {
    float err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err_x = fabsf(a[i].x - b[i].x);
        float err_y = fabsf(a[i].y - b[i].y);
        float err_z = fabsf(a[i].z - b[i].z);
        if (err_x > err) err = err_x;
        if (err_y > err) err = err_y;
        if (err_z > err) err = err_z;
        if(i < 100 && err > ACCURACY)
        printf("Body %d: CPU(%f,%f,%f) GPU(%f,%f,%f)\n", i, a[i].x, a[i].y, a[i].z, b[i].x, b[i].y, b[i].z);
    }
    return err;
}
int main(const int argc, const char *argv[]) {

    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    printf("GPU name: %s\n", prop.name);
    printf("Shared memory per block: %zu bytes\n",
           prop.sharedMemPerBlock);
    printf("Shared memory per SM: %zu bytes\n",
           prop.sharedMemPerMultiprocessor);


    /* Default Configuration */
    int num_systems = 32;       	/* Number of independent galaxies */
    int bodies_per_system = 8192;	/* Number of bodies per galaxy */
    int nIters = 20;            	/* Simulation steps */ 

    const float dt = 0.01f;
    FILE *fp;
    int total_bodies, bytes, sys, iter;
    Body *dataGPU, *dataCPU, *h_data, *system_ptr;
    float *buf;
    double totalTimeGPU, totalTimeCPU, interactions_per_system, total_interactions;

    {
        /* Attempt to load dataset */
        fp = fopen("galaxy_data.bin", "rb");
        if (fp) {
            fread(&num_systems, sizeof(int), 1, fp);
            fread(&bodies_per_system, sizeof(int), 1, fp);
            printf("Found dataset: %d systems of %d bodies.\n", num_systems,
                    bodies_per_system);
        } else {
            printf("No dataset found. Using random initialization.\n");
        }

        /* Allocate memory for ALL systems */
        total_bodies = num_systems * bodies_per_system;
        bytes = total_bodies * sizeof(Body);
        dataCPU = (Body *) malloc(bytes);
        CUDA_CHECK(cudaMalloc((void **) &dataGPU, bytes));
        h_data = (Body *) malloc(bytes);

        /* Initialize dataGPU */
        if (fp) {
            fread(dataCPU, sizeof(Body), total_bodies, fp);
            fclose(fp);
        } else {
        /* Random initialization if file missing */
            buf = (float *) dataCPU;
            for (int i = 0; i < 6 * total_bodies; i++) {
                buf[i] = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
            }
        }
        CUDA_CHECK(cudaMemcpy(dataGPU, dataCPU, bytes,
                    cudaMemcpyHostToDevice));

        printf("Running GPU simulation for %d systems...\n",
            num_systems);

        totalTimeGPU = 0.0;
        StartTimer();
        double tempTime = 0.0;
        double  bodyforceTime = 0.0f;
        double integrateTime = 0.0f;
        dim3 Block(bodies_per_system, 8);

        cudaStream_t *streams = (cudaStream_t *) malloc(num_systems * sizeof(cudaStream_t));
        for (int i = 0; i < num_systems; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }

        /* Time-steps */
        for (iter = 1; iter <= nIters; iter++) {
            /* Galaxies */
            for (sys = 0; sys < num_systems; sys++) {
                /* Calculate offset for the galaxy */
                system_ptr = &dataGPU[sys * bodies_per_system];
                // bodyForceKernel<<<Block, CUDA_BLOCK_SIZE,0,steams[sys]>>>(system_ptr, dt, bodies_per_system);
                bodyForceKernel<<<Block, CUDA_BLOCK_SIZE>>>(system_ptr, dt, bodies_per_system);
                cudaDeviceSynchronize();
                // integrateKernel<<<8, CUDA_BLOCK_SIZE,0,steams[sys]>>>(system_ptr, dt, bodies_per_system);
                integrateKernel<<<8, CUDA_BLOCK_SIZE>>>(system_ptr, dt, bodies_per_system);
                cudaDeviceSynchronize();
            }
        }
        outGPU:
        totalTimeGPU = GetTimer() / 1000.0;
        /* Metrics calculation */
        interactions_per_system = (double) bodies_per_system * bodies_per_system;
        total_interactions = interactions_per_system * num_systems * nIters;
        printf(CYAN"Total GPU Time: %.3f seconds\n" RESET, totalTimeGPU);
        printf("Average  GPU Throughput: %0.3f Billion Interactions / second\n\n\n",
            1e-9 * total_interactions / totalTimeGPU);
        CUDA_CHECK(cudaMemcpy(h_data, dataGPU, bytes,
                    cudaMemcpyDeviceToHost));
        free(dataCPU);
        cudaFree(dataGPU);
    }

    // CPU computation for validation
    // {
    //     /* Attempt to load dataset */
    //     fp = fopen("galaxy_data.bin", "rb");
    //     if (fp) {
    //         fread(&num_systems, sizeof(int), 1, fp);
    //         fread(&bodies_per_system, sizeof(int), 1, fp);
    //         printf("Found dataset: %d systems of %d bodies.\n", num_systems,
    //                 bodies_per_system);
    //     } else {
    //         printf("No dataset found. Using random initialization.\n");
    //     }

    //     /* Allocate memory for ALL systems */
    //     total_bodies = num_systems * bodies_per_system;
    //     bytes = total_bodies * sizeof(Body);
    //     dataCPU = (Body *) malloc(bytes);

    //     /* Initialize dataCPU */
    //     if (fp) {
    //         fread(dataCPU, sizeof(Body), total_bodies, fp);
    //         fclose(fp);
    //     } else {
    //     /* Random initialization if file missing */
    //         buf = (float *) dataCPU;
    //         for (int i = 0; i < 6 * total_bodies; i++) {
    //             buf[i] = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
    //         }
    //     }

    //     printf("Running parallel CPU simulation for %d systems...\n",
    //         num_systems);

    //     totalTimeCPU = 0.0;

    //     StartTimer();

    //     /* Time-steps */
    //     for (iter = 1; iter <= nIters; iter++) {
    //         /* Galaxies */
    //         for (sys = 0; sys < num_systems; sys++) {
    //             /* Calculate offset for the galaxy */
    //             system_ptr = &dataCPU[sys * bodies_per_system];
    //             /* Compute forces & integrate for the galaxy */
    //             bodyForce(system_ptr, dt, bodies_per_system);
    //             integrate(system_ptr, dt, bodies_per_system);
                
    //             // if(sys==0)
    //             //     goto outCPU;
    //         }
    //     }
    //     outCPU:
    //     totalTimeCPU = GetTimer() / 1000.0;

    //     /* Metrics calculation */
    //     interactions_per_system = (double) bodies_per_system * bodies_per_system;
    //     total_interactions = interactions_per_system * num_systems * nIters;

    //     printf(BLUE "Total CPU Time: %.3f seconds\n" RESET, totalTimeCPU);
    //     printf("Average CPU Throughput: %0.3f Billion Interactions / second\n",
    //         1e-9 * total_interactions / totalTimeCPU);
    // }
    // float err = calc_max_err(dataCPU, h_data, total_bodies);
    // if (err < ACCURACY){
    //     printf(GREEN "Final Error between CPU and GPU results: %f<%f\n" RESET "\n", err, ACCURACY);
    //     printf(MAGENTA "SPEEDUP: %0.2fx\n" RESET, totalTimeCPU/totalTimeGPU);
    // }
    // else
    //     printf(RED "Final Error between CPU and GPU results: %f>%f\n" RESET "\n", err, ACCURACY);
    // free(dataCPU);
    free(h_data);
    return 0;
}