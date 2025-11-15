# Lab 3 notes

0) The device query gives the result of 
Device 0: NVIDIA GeForce RTX 4070
  Compute capability:   8.9
  Total global memory:  11850 MB
  Shared memory/block:  48 KB
  Registers/block:      65536
  Warp size:            32
  Max threads/block:    1024
  Multiprocessors:      46
  Max threads dim:      (1024, 1024, 64)
  Max grid size:        (2147483647, 65535, 65535)

We need to do that as well with the HPC PC.

1) Cool

2) 
    - Done
    - Done with the `checkResults` Function
3) 
    - The maximum size of a single block is 32x32 -> 1024 threads. After that we get 
    `CUDA Error: invalid argument (err_num=1) at Convolution2D.cu:271` 
    This is because the maximum number of a Warp is 32 and the maximum number of threads/block as we can see from the device query is 1024.
    - Apparently 0 is good enough....
4)  
    - Done with 
    ```
    dim3 dimBlock(32, 32);
    dim3 dimGrid((imageW + dimBlock.x - 1) / dimBlock.x,
                 (imageH + dimBlock.y - 1) / dimBlock.y);
    convolutionRowGPU<<<dimGrid, dimBlock>>>(d_Buffer, d_Input, d_Filter, 
                imageW, imageH, filter_radius);
    ```
    Instead of 
    ```
    dim3 dimBlock(imageW, imageH);
    convolutionRowGPU<<<1, dimBlock>>>(d_Buffer, d_Input, d_Filter, 
                                imageW, imageH, filter_radius);
    ```
   The maximum image size **30.000x30.000** (ON MY PC NEED TO CHECK ON THE HPC CLUSTER). This is because exceeding that size means the memory for the GPU is not enough. Since the GPU has 12GB of ram and we have 3 buffers of image size, whith each being 4GB (a bit less) big, the memory does not suffice.
5) 
    - For 1024x1024 pictures, I dont know man.... its 100x times faster. That is not a problem...., PROBABLY NEED SOMETHING DIFFERENT.
    -  ON MY PC the CPU time is 258s in comparisson with 0.32s on the GPU.

6)  - ON MY PC THE GPU took 0.4s in comparisson with 286s.... In general the time increase is in the orders of magnitude of +30%.

7) TO BE DONE

8) TO BE DONE
