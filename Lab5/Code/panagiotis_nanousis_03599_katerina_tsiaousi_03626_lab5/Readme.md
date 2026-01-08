To compile the program, use the provided makefile.

`make`

to run the nbody program either on CPU or GPU you can use 

`make run_cpu` `make run_gpu`

To validate the correctness of the gpu code you can use 

`make run_gpuV` or `./nbody_gpu v`

where the verification runs both the cpu and the GPU version and finds the maximum difference between the position of the bodies.

You can also cache the cpu data by uncommenting the line 514 and commenting the line 513 like this:
```
        //if(1){
        if(!correct_dump){
```

Make sure to delete the `corect_dump` file when changing the number of systems/bodies or the initial galaxy data.