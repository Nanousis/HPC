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

#define PADDED_WIDTH(image_W, filter_R)  (image_W + 2*filter_R)
#define PADDED_HEIGHT(image_H, filter_R) (image_H + 2*filter_R)

// we want to start from (0,filter/2) in the padded image
/* For example  filter = 2 image = 4,4
p,p,p, p, p, p, p, p
p,p,p, p, p, |p|, p, p
p,p,         ------
p,p,i, i, i, |i|, p, p
p,p,        -------
p,p,i, i, i, i, p, p
p,p,i, i, i, i, p, p
p,p,i, i, i, i, p, p
p,p,p, p, p, p, p, p
p,p,p, p, p, p, p, p

The filters are the || and -- vertical and horizontal lines respectively
*/
#define WRITE_TO_PADDED_IMAGE(x,y,padded_W, filter_R) \
    ((x + filter_R) + ((y + (filter_R)) * padded_W))

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
#define accuracy  	10
typedef float f_data;

 

__global__ void convolutionRowGPU(f_data *d_Dst, f_data *d_Src, f_data *d_Filter, 
                            int imageW, int imageH, int filterR) {

  // 2D Thread ID
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int paddedImageW = PADDED_WIDTH(imageW, filterR);

  if (x < imageW && y < imageH) {
    f_data sum = 0;

    for (int k = -filterR; k <= filterR; k++) {
      int d = x + k;
      // if (d >= 0 && d < imageW) {
        sum += d_Src[WRITE_TO_PADDED_IMAGE(d, y, paddedImageW, filterR)] * d_Filter[filterR - k];
      // }     
    }

    d_Dst[WRITE_TO_PADDED_IMAGE(x, y, paddedImageW, filterR)] = sum;
  }
}


__global__ void convolutionColumnGPU(f_data *d_Dst, f_data *d_Src, f_data *d_Filter,
                             int imageW, int imageH, int filterR) {

  // 2D Thread ID
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int paddedImageW = PADDED_WIDTH(imageW, filterR);
  if (x < imageW && y < imageH) {
    f_data sum = 0;

    for (int k = -filterR; k <= filterR; k++) {
      int d = y + k;
      // if (d >= 0 && d < imageH) {
        sum += d_Src[WRITE_TO_PADDED_IMAGE(x, d, paddedImageW, filterR)] * d_Filter[filterR - k];
      // }   
    }

    d_Dst[WRITE_TO_PADDED_IMAGE(x, y, paddedImageW, filterR)] = sum;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU_Padded(f_data *h_Dst, f_data *h_Src, f_data *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
  int paddedImageW = PADDED_WIDTH(imageW, filterR);
  int prints =0;
  prints = 0;
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      f_data sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[WRITE_TO_PADDED_IMAGE(d, y, paddedImageW, filterR)] * h_Filter[filterR - k];
        }     
        h_Dst[WRITE_TO_PADDED_IMAGE(x, y, paddedImageW, filterR)] = sum;
      }
    }
  }
        
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU_Padded(f_data *h_Dst, f_data *h_Src, f_data *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  int paddedImageW = PADDED_WIDTH(imageW, filterR);
  printf("Starting convolutionColumnCPU_Padded\n");
  int prints =0;
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      f_data sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          // if(prints++ <10){
          //   printf("Accessing h_Src[%d] = %f \n", WRITE_TO_PADDED_IMAGE(x, d, paddedImageW, filterR), h_Src[WRITE_TO_PADDED_IMAGE(x, d, paddedImageW, filterR)]);
          // }
          sum += h_Src[WRITE_TO_PADDED_IMAGE(x, d, paddedImageW, filterR)] * h_Filter[filterR - k];
        }   
 
        h_Dst[WRITE_TO_PADDED_IMAGE(x, y, paddedImageW, filterR)] = sum;
      }
    }
    printf("current progress: x=%d y=%d\r", x, y);
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
    printf("current progress: x=%d y=%d\r", x, y);
  }
    
}



bool checkResults_both_padded(f_data *hostRef, f_data *gpuRef, double epsilon, const int width, const int height, const unsigned int filter_radius) {
  bool match = true;
  float max_delta = 0.0f;
  for (int i = 0; i < height; i++) {
    for(int j = 0; j < width; j++) {
      int idx = WRITE_TO_PADDED_IMAGE(j, i, PADDED_WIDTH(width, filter_radius), filter_radius);
      float delta = ABS(hostRef[idx] - gpuRef[idx]);
      if (delta > epsilon) {
        match = false;
        if (delta > max_delta) {
          max_delta = delta;
        }
        printf("Arrays do not match!\n");
        printf("host %f gpu %f at current %d, delta = %f, epsilon = %f\n", hostRef[idx], gpuRef[idx], i, ABS(hostRef[idx] - gpuRef[idx]), epsilon);
        goto exit_error;
      }
    }
  }
  exit_error:
  printf("Max delta: %f\n", max_delta);
  return match;
}
bool checkResults(f_data *hostRef, f_data *gpuRef, double epsilon, const int width, const int height, const unsigned int filter_radius) {
  printf("Checking results with normal CPU and GPU Padding...\n");
  bool match = true;
  float max_delta = 0.0f;
  for (int i = 0; i < height; i++) {
    for(int j = 0; j < width; j++) {
      int idx = WRITE_TO_PADDED_IMAGE(j, i, PADDED_WIDTH(width, filter_radius), filter_radius);
      float delta = ABS(hostRef[i*height + j] - gpuRef[idx]);
      if (delta > epsilon) {
        match = false;
        if (delta > max_delta) {
          max_delta = delta;
        }
        printf("Arrays do not match!\n");
        printf("host %f [%d] gpu %f [%d] at current (%d,%d), delta = %f, epsilon = %f\n", hostRef[i*height + j], i*height + j, gpuRef[idx], idx, j, i, ABS(hostRef[i*height + j] - gpuRef[idx]), epsilon);
        goto exit_error;
      }
    }
  }
  exit_error:
  printf("Max delta: %f\n", max_delta);
  return match;
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    f_data
    *h_Filter,
    *h_Input,
    *h_Input_test,
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
    
    int paddedImageW = PADDED_WIDTH(imageW, filter_radius);
    int paddedImageH = PADDED_HEIGHT(imageH, filter_radius);

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (f_data *)malloc(FILTER_LENGTH * sizeof(f_data));
    h_Input     = (f_data *)malloc(paddedImageW * paddedImageH * sizeof(f_data));
    h_Input_test     = (f_data *)malloc(imageW * imageH * sizeof(f_data));
    h_Buffer    = (f_data *)malloc(paddedImageW * paddedImageH * sizeof(f_data));
    h_OutputCPU = (f_data *)malloc(paddedImageW * paddedImageH * sizeof(f_data));
    h_OutputGPU = (f_data *)malloc(paddedImageW * paddedImageH * sizeof(f_data));    
    //  Cuda malloc, same items as the host arrays
    CUDA_CHECK(cudaMalloc((void **)&d_Filter,    FILTER_LENGTH * sizeof(f_data)));
    CUDA_CHECK(cudaMalloc((void **)&d_Input,     paddedImageW * paddedImageH * sizeof(f_data)));
    CUDA_CHECK(cudaMalloc((void **)&d_Buffer,    paddedImageW * paddedImageH * sizeof(f_data)));
    CUDA_CHECK(cudaMalloc((void **)&d_OutputGPU, paddedImageW * paddedImageH * sizeof(f_data)));


    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.
    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (f_data)(rand() % 16);
    }
    // Initialize the input pad with 0. Can be more efficient but oh well...
    for( i = 0 ; i < paddedImageW * paddedImageH; i++) {
        h_Input[i] = 0;
    }
    printf("Padded size = %d x %d = %d vs %d x %d = %d \n", paddedImageW, paddedImageH, paddedImageW * paddedImageH, imageW, imageH, imageW * imageH);
    for (i = 0; i < imageW * imageH; i++) {
        h_Input_test[i] = (f_data)rand() / ((f_data)RAND_MAX / 255) + (f_data)rand() / (f_data)RAND_MAX;
        // printf("Input test[%d] = %f\n", i, h_Input_test[i]);
    }
    int prints =0;
    for (int i = 0; i < imageH; i++) {
      for(int j = 0; j < imageW; j++) {
        int idx = WRITE_TO_PADDED_IMAGE(j, i, paddedImageW, filter_radius);
        if(prints++ <10){
          printf("|%d,%d = %d -> %f|\n", i, j, idx, h_Input_test[j + imageW* i]);
          fflush(stdout);
        }
        h_Input[idx] = h_Input_test[j + imageW* i];
      }
    }

    double startGPU = now();
    // Copy the fiilter and image to the gpu
    CUDA_CHECK(cudaMemcpy(d_Filter, h_Filter, 
                FILTER_LENGTH * sizeof(f_data),
                 cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Input, h_Input, 
                paddedImageW * paddedImageH * sizeof(f_data),
                 cudaMemcpyHostToDevice));
      

    //  GPU CALCULATIONS
    printf("GPU computation...\n");
    // dim3 dimBlock(imageW, imageH);
    dim3 dimBlock(32, 32);
    dim3 dimGrid((imageW + dimBlock.x - 1) / dimBlock.x,
                 (imageH + dimBlock.y - 1) / dimBlock.y);
    convolutionRowGPU<<<dimGrid, dimBlock>>>(d_Buffer, d_Input, d_Filter, 
                                    imageW, imageH, filter_radius);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    convolutionColumnGPU<<<dimGrid, dimBlock>>>(d_OutputGPU, d_Buffer, d_Filter,
                                     imageW, imageH, filter_radius);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    double endGPU = now();
    printf("GPU computation time: %f sec\n", endGPU - startGPU);

    // Transfering back to host the result of GPU computation
    CUDA_CHECK(cudaMemcpy(h_OutputGPU, d_OutputGPU, 
                paddedImageW * paddedImageH * sizeof(f_data),
                 cudaMemcpyDeviceToHost));


    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");
    double start = now();
    #ifndef DEBUG
    convolutionRowCPU(h_Buffer, h_Input_test, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
    #else
    convolutionRowCPU_Padded(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU_Padded(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
    #endif
    double end = now();
    printf("CPU computation time: %f sec\n", end - start);


    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  
    printf("Checking the results...\n");
    printf("Output GPU (padded):\n");
    #ifdef DEBUG
    for(int y = 0; y < 10; y++) {
      for(int x = 0; x < 10; x++) {
        // printf("%.2f, ", h_OutputCPU[y * imageW + x]);
        printf("%.2f vs %.2f, ", h_OutputCPU[WRITE_TO_PADDED_IMAGE(x, y, paddedImageW, filter_radius)], h_OutputGPU[WRITE_TO_PADDED_IMAGE(x, y, paddedImageW, filter_radius)]);
        // printf("|%d,%d = %d |", x, y, WRITE_TO_PADDED_IMAGE(x, y, paddedImageW,filter_radius));
        // fflush(stdout);
      }
      printf("\n");
    }
    #endif
    #ifdef DEBUG
    bool res = checkResults_both_padded(h_OutputCPU, h_OutputGPU, accuracy, imageW, imageH, filter_radius);
    #else
    bool res = checkResults(h_OutputCPU, h_OutputGPU, accuracy, imageW, imageH, filter_radius);
    #endif
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
