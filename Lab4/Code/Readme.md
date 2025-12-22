## To compile
To compile you can just use the makefile. It will produce two executables, one is the serial cpu implementation and the other one uses cuda.

You can also quickly test the program by running `make run_cpu` or `make run_gpu`.

To check for the correctness of the picture you can run 
`python psnr.py python psnr.py output_cpu.pgm output_gpu.pgm`

You can also measure the times with std deviation with `measure.sh`