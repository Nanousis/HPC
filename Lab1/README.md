## Running the program

You can make the program with make
`make CFLAGS="O0/O3 -DYOURFLAG"`
and you can run it with
`./sobel_orig`

you can also see the output picture by converting it to jpeg with `make image`. The picture is called on ./output_sobel.jpg

## Running the tests

To run the tests simply run 
`python run_timings.py`

## Making new tests

To make new tests simply add a new definition flag on the .c file. The idea is to have each change not affecting the other changes as much as possible. After that add the definition flag as a new test on the python script here:
```
    opt_flags = {
        "-O0": "-O0",
        "-O3": "-O3 -march=znver4 -mtune=znver4", <-- Change this to -fast if you have an intel cpu
    }
    def_flags = ["", "-DLOOP_SWAP", "-DLOOP_UNROLL", "-DLOOP_UNROLL2"] <--- HERE
    tests = []
```
After that just run the program and see the results.