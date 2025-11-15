/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <time.h>
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



double now() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);   // monotonic = won't jump if system time changes
    return t.tv_sec + t.tv_nsec * 1e-9;
}


unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0
typedef float f_data;

 

__global__ void convolutionRowGPU(f_data *d_Dst, f_data *d_Src, f_data *d_Filter, 
                            int imageW, int imageH, int filterR) {

  // 2D Thread ID
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < imageW && y < imageH) {
    f_data sum = 0;

    for (int k = -filterR; k <= filterR; k++) {
      int d = x + k;

      if (d >= 0 && d < imageW) {
        sum += d_Src[y * imageW + d] * d_Filter[filterR - k];
      }     
    }

    d_Dst[y * imageW + x] = sum;
  }
}


__global__ void convolutionColumnGPU(f_data *d_Dst, f_data *d_Src, f_data *d_Filter,
                             int imageW, int imageH, int filterR) {

  // 2D Thread ID
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < imageW && y < imageH) {
    f_data sum = 0;

    for (int k = -filterR; k <= filterR; k++) {
      int d = y + k;

      if (d >= 0 && d < imageH) {
        sum += d_Src[d * imageW + x] * d_Filter[filterR - k];
      }   
    }

    d_Dst[y * imageW + x] = sum;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(f_data *h_Dst, f_data *h_Src, f_data *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      f_data sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }     

        h_Dst[y * imageW + x] = sum;
      }
    }
  }
        
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(f_data *h_Dst, f_data *h_Src, f_data *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      f_data sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }   
 
        h_Dst[y * imageW + x] = sum;
      }
    }
  }
    
}


bool checkResults(f_data *hostRef, f_data *gpuRef, double epsilon, const int SIZE) {
  bool match = true;

  for (int i = 0; i < SIZE; i++) {
    if (ABS(hostRef[i] - gpuRef[i]) > epsilon) {
      match = false;
      printf("Arrays do not match!\n");
      printf("host %f gpu %f at current %d, delta = %f, epsilon = %f\n", hostRef[i], gpuRef[i], i, ABS(hostRef[i] - gpuRef[i]), epsilon);
      break;
    }
  }

  return match;
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    f_data
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *h_OutputGPU;
  
    f_data
    *d_Filter,
    *d_Input,
    *d_Buffer,
    *d_OutputGPU;


    int imageW;
    int imageH;
    unsigned int i;

    int count = 0;
    cudaGetDeviceCount(&count);

    if(count==0){
        printf("No devices supporting CUDA.\n");
        return 0;
    }

    for (int i = 0; i < count; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);

      printf("Device %d: %s\n", i, prop.name);
      printf("  Compute capability:   %d.%d\n", prop.major, prop.minor);
      printf("  Total global memory:  %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
      printf("  Shared memory/block:  %zu KB\n", prop.sharedMemPerBlock / 1024);
      printf("  Registers/block:      %d\n", prop.regsPerBlock);
      printf("  Warp size:            %d\n", prop.warpSize);
      printf("  Max threads/block:    %d\n", prop.maxThreadsPerBlock);
      printf("  Multiprocessors:      %d\n", prop.multiProcessorCount);
      printf("  Max threads dim:      (%d, %d, %d)\n",
              prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
      printf("  Max grid size:        (%d, %d, %d)\n",
              prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

      printf("--------------------------------------\n");
    }


	printf("Enter filter radius : ");
	scanf("%d", &filter_radius);

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (f_data *)malloc(FILTER_LENGTH * sizeof(f_data));
    h_Input     = (f_data *)malloc(imageW * imageH * sizeof(f_data));
    h_Buffer    = (f_data *)malloc(imageW * imageH * sizeof(f_data));
    h_OutputCPU = (f_data *)malloc(imageW * imageH * sizeof(f_data));
    h_OutputGPU = (f_data *)malloc(imageW * imageH * sizeof(f_data));
    
    //  Cuda malloc, same items as the host arrays
    CUDA_CHECK(cudaMalloc((void **)&d_Filter,    FILTER_LENGTH * sizeof(f_data)));
    CUDA_CHECK(cudaMalloc((void **)&d_Input,     imageW * imageH * sizeof(f_data)));
    CUDA_CHECK(cudaMalloc((void **)&d_Buffer,    imageW * imageH * sizeof(f_data)));
    CUDA_CHECK(cudaMalloc((void **)&d_OutputGPU, imageW * imageH * sizeof(f_data)));


    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (f_data)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (f_data)rand() / ((f_data)RAND_MAX / 255) + (f_data)rand() / (f_data)RAND_MAX;
    }

    double startGPU = now();
    // Copy the fiilter and image to the gpu
    CUDA_CHECK(cudaMemcpy(d_Filter, h_Filter, 
                FILTER_LENGTH * sizeof(f_data),
                 cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Input, h_Input, 
                imageW * imageH * sizeof(f_data),
                 cudaMemcpyHostToDevice));
      

    //  GPU CALCULATIONS
    printf("GPU computation...\n");
    // dim3 dimBlock(imageW, imageH);
    dim3 dimBlock(32, 32);
    dim3 dimGrid((imageW + dimBlock.x - 1) / dimBlock.x,
                 (imageH + dimBlock.y - 1) / dimBlock.y);
    convolutionRowGPU<<<dimGrid, dimBlock>>>(d_Buffer, d_Input, d_Filter, 
                                    imageW, imageH, filter_radius);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    convolutionColumnGPU<<<dimGrid, dimBlock>>>(d_OutputGPU, d_Buffer, d_Filter,
                                     imageW, imageH, filter_radius);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    double endGPU = now();
    printf("GPU computation time: %f sec\n", endGPU - startGPU);

    // Transfering back to host the result of GPU computation
    CUDA_CHECK(cudaMemcpy(h_OutputGPU, d_OutputGPU, 
                imageW * imageH * sizeof(f_data),
                 cudaMemcpyDeviceToHost));


    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");
    double start = now();
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
    double end = now();
    printf("CPU computation time: %f sec\n", end - start);


    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  
    printf("Checking the results...\n");
    bool res = checkResults(h_OutputCPU, h_OutputGPU, accuracy, imageW * imageH);
    if (res) {
        printf(GREEN "Results match.\n" RESET);
    } else {
        printf(RED "Results do not match.\n" RESET);
    }

    cudaFree(d_Filter);
    cudaFree(d_Input);
    cudaFree(d_Buffer);
    cudaFree(d_OutputGPU);

    // free all the allocated memory
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);

    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    cudaDeviceReset();


    return 0;
}
