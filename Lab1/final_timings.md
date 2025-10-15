### Final timings from the optimization combos :

```
=== Top combinations by speedup vs O0 (descending) ===
132.57x  O3_ZNVER4_C3[LOOP_SWAP+LOOP_UNROLL+COMPILER_ASSIST] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DCOMPILER_ASSIST   (avg 0.008223 s)
131.90x  O3_ZNVER4_C4[LOOP_SWAP+LOOP_UNROLL+FUNC_INLINE+COMPILER_ASSIST] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DFUNC_INLINE -DCOMPILER_ASSIST   (avg 0.008265 s)
129.48x  O3_ZNVER4_C4[LOOP_SWAP+LOOP_UNROLL+LOOP_UNROLL2+COMPILER_ASSIST] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DCOMPILER_ASSIST   (avg 0.008420 s)
127.53x  O3_ZNVER4_C5[LOOP_SWAP+LOOP_UNROLL+LOOP_UNROLL2+COMPILER_ASSIST+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DCOMPILER_ASSIST -DFUSION   (avg 0.008548 s)
126.23x  O3_ZNVER4_C5[LOOP_SWAP+LOOP_UNROLL+LOOP_UNROLL2+FUNC_INLINE+COMPILER_ASSIST] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST   (avg 0.008636 s)
119.84x  O3_ZNVER4_C4[LOOP_SWAP+LOOP_UNROLL+COMPILER_ASSIST+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DCOMPILER_ASSIST -DFUSION   (avg 0.009097 s)
118.69x  O3_ZNVER4_C6[LOOP_SWAP+LOOP_UNROLL+LOOP_UNROLL2+FUNC_INLINE+COMPILER_ASSIST+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST -DFUSION   (avg 0.009185 s)
117.65x  O3_ZNVER4_C5[LOOP_SWAP+LOOP_UNROLL+FUNC_INLINE+COMPILER_ASSIST+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DFUNC_INLINE -DCOMPILER_ASSIST -DFUSION   (avg 0.009266 s)
117.18x  O3_ZNVER4_ALLDEFS :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST -DFUSION   (avg 0.009303 s)
 99.40x  O3_ZNVER4_C3[LOOP_SWAP+LOOP_UNROLL+FUNC_INLINE] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DFUNC_INLINE   (avg 0.010967 s)
```

```
O3_ZNVER4_C2[LOOP_SWAP+LOOP_UNROLL] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL:
	 Average time = 0.011059 s, Stddev = 0.000132 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 98.58x, speedup vs O3: 18.82x

O3_ZNVER4_C2[LOOP_SWAP+LOOP_UNROLL2] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL2:
	 Average time = 0.085690 s, Stddev = 0.001446 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 12.72x, speedup vs O3: 2.43x

O3_ZNVER4_C2[LOOP_SWAP+FUNC_INLINE] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DFUNC_INLINE:
	 Average time = 0.083877 s, Stddev = 0.001630 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 13.00x, speedup vs O3: 2.48x

O3_ZNVER4_C2[LOOP_SWAP+COMPILER_ASSIST] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DCOMPILER_ASSIST:
	 Average time = 0.031912 s, Stddev = 0.000478 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 34.16x, speedup vs O3: 6.52x

O3_ZNVER4_C2[LOOP_SWAP+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DFUSION:
	 Average time = 0.094922 s, Stddev = 0.000785 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 11.48x, speedup vs O3: 2.19x

O3_ZNVER4_C2[LOOP_UNROLL+LOOP_UNROLL2] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL -DLOOP_UNROLL2:
	 Average time = 0.261356 s, Stddev = 0.004424 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 4.17x, speedup vs O3: 0.80x

O3_ZNVER4_C2[LOOP_UNROLL+FUNC_INLINE] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL -DFUNC_INLINE:
	 Average time = 0.232322 s, Stddev = 0.003293 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 4.69x, speedup vs O3: 0.90x

O3_ZNVER4_C2[LOOP_UNROLL+COMPILER_ASSIST] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL -DCOMPILER_ASSIST:
	 Average time = 0.301575 s, Stddev = 0.002092 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 3.61x, speedup vs O3: 0.69x

O3_ZNVER4_C2[LOOP_UNROLL+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL -DFUSION:
	 Average time = 0.294049 s, Stddev = 0.004724 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 3.71x, speedup vs O3: 0.71x

O3_ZNVER4_C2[LOOP_UNROLL2+FUNC_INLINE] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL2 -DFUNC_INLINE:
	 Average time = 0.248345 s, Stddev = 0.003497 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 4.39x, speedup vs O3: 0.84x

O3_ZNVER4_C2[LOOP_UNROLL2+COMPILER_ASSIST] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL2 -DCOMPILER_ASSIST:
	 Average time = 0.320924 s, Stddev = 0.002490 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 3.40x, speedup vs O3: 0.65x

O3_ZNVER4_C2[LOOP_UNROLL2+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL2 -DFUSION:
	 Average time = 0.386506 s, Stddev = 0.009820 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 2.82x, speedup vs O3: 0.54x

O3_ZNVER4_C2[FUNC_INLINE+COMPILER_ASSIST] :: -O3 -march=znver4 -mtune=znver4 -DFUNC_INLINE -DCOMPILER_ASSIST:
	 Average time = 0.295989 s, Stddev = 0.003204 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 3.68x, speedup vs O3: 0.70x

O3_ZNVER4_C2[FUNC_INLINE+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DFUNC_INLINE -DFUSION:
	 Average time = 0.326001 s, Stddev = 0.004065 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 3.34x, speedup vs O3: 0.64x

O3_ZNVER4_C2[COMPILER_ASSIST+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DCOMPILER_ASSIST -DFUSION:
	 Average time = 0.386797 s, Stddev = 0.007475 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 2.82x, speedup vs O3: 0.54x

O3_ZNVER4_C3[LOOP_SWAP+LOOP_UNROLL+LOOP_UNROLL2] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2:
	 Average time = 0.012089 s, Stddev = 0.000731 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 90.18x, speedup vs O3: 17.22x

O3_ZNVER4_C3[LOOP_SWAP+LOOP_UNROLL+FUNC_INLINE] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DFUNC_INLINE:
	 Average time = 0.010967 s, Stddev = 0.000131 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 99.40x, speedup vs O3: 18.98x

O3_ZNVER4_C3[LOOP_SWAP+LOOP_UNROLL+COMPILER_ASSIST] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DCOMPILER_ASSIST:
	 Average time = 0.008223 s, Stddev = 0.000065 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 132.57x, speedup vs O3: 25.32x

O3_ZNVER4_C3[LOOP_SWAP+LOOP_UNROLL+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DFUSION:
	 Average time = 0.011372 s, Stddev = 0.000097 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 95.86x, speedup vs O3: 18.31x

O3_ZNVER4_C3[LOOP_SWAP+LOOP_UNROLL2+FUNC_INLINE] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL2 -DFUNC_INLINE:
	 Average time = 0.085913 s, Stddev = 0.000987 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 12.69x, speedup vs O3: 2.42x

O3_ZNVER4_C3[LOOP_SWAP+LOOP_UNROLL2+COMPILER_ASSIST] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL2 -DCOMPILER_ASSIST:
	 Average time = 0.031990 s, Stddev = 0.000662 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 34.08x, speedup vs O3: 6.51x

O3_ZNVER4_C3[LOOP_SWAP+LOOP_UNROLL2+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL2 -DFUSION:
	 Average time = 0.094783 s, Stddev = 0.001157 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 11.50x, speedup vs O3: 2.20x

O3_ZNVER4_C3[LOOP_SWAP+FUNC_INLINE+COMPILER_ASSIST] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DFUNC_INLINE -DCOMPILER_ASSIST:
	 Average time = 0.032168 s, Stddev = 0.000526 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 33.89x, speedup vs O3: 6.47x

O3_ZNVER4_C3[LOOP_SWAP+FUNC_INLINE+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DFUNC_INLINE -DFUSION:
	 Average time = 0.095240 s, Stddev = 0.001761 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 11.45x, speedup vs O3: 2.19x

O3_ZNVER4_C3[LOOP_SWAP+COMPILER_ASSIST+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DCOMPILER_ASSIST -DFUSION:
	 Average time = 0.034209 s, Stddev = 0.000516 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 31.87x, speedup vs O3: 6.09x

O3_ZNVER4_C3[LOOP_UNROLL+LOOP_UNROLL2+FUNC_INLINE] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE:
	 Average time = 0.232791 s, Stddev = 0.002723 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 4.68x, speedup vs O3: 0.89x

O3_ZNVER4_C3[LOOP_UNROLL+LOOP_UNROLL2+COMPILER_ASSIST] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL -DLOOP_UNROLL2 -DCOMPILER_ASSIST:
	 Average time = 0.310572 s, Stddev = 0.004862 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 3.51x, speedup vs O3: 0.67x

O3_ZNVER4_C3[LOOP_UNROLL+LOOP_UNROLL2+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUSION:
	 Average time = 0.371591 s, Stddev = 0.007911 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 2.93x, speedup vs O3: 0.56x

O3_ZNVER4_C3[LOOP_UNROLL+FUNC_INLINE+COMPILER_ASSIST] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL -DFUNC_INLINE -DCOMPILER_ASSIST:
	 Average time = 0.298751 s, Stddev = 0.006752 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 3.65x, speedup vs O3: 0.70x

O3_ZNVER4_C3[LOOP_UNROLL+FUNC_INLINE+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL -DFUNC_INLINE -DFUSION:
	 Average time = 0.332297 s, Stddev = 0.003760 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 3.28x, speedup vs O3: 0.63x

O3_ZNVER4_C3[LOOP_UNROLL+COMPILER_ASSIST+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL -DCOMPILER_ASSIST -DFUSION:
	 Average time = 0.400342 s, Stddev = 0.003033 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 2.72x, speedup vs O3: 0.52x

O3_ZNVER4_C3[LOOP_UNROLL2+FUNC_INLINE+COMPILER_ASSIST] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST:
	 Average time = 0.322886 s, Stddev = 0.004898 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 3.38x, speedup vs O3: 0.64x

O3_ZNVER4_C3[LOOP_UNROLL2+FUNC_INLINE+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL2 -DFUNC_INLINE -DFUSION:
	 Average time = 0.392242 s, Stddev = 0.011524 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 2.78x, speedup vs O3: 0.53x

O3_ZNVER4_C3[LOOP_UNROLL2+COMPILER_ASSIST+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL2 -DCOMPILER_ASSIST -DFUSION:
	 Average time = 0.433852 s, Stddev = 0.001687 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 2.51x, speedup vs O3: 0.48x

O3_ZNVER4_C3[FUNC_INLINE+COMPILER_ASSIST+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DFUNC_INLINE -DCOMPILER_ASSIST -DFUSION:
	 Average time = 0.390802 s, Stddev = 0.003777 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 2.79x, speedup vs O3: 0.53x

O3_ZNVER4_C4[LOOP_SWAP+LOOP_UNROLL+LOOP_UNROLL2+FUNC_INLINE] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE:
	 Average time = 0.012192 s, Stddev = 0.000640 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 89.42x, speedup vs O3: 17.07x

O3_ZNVER4_C4[LOOP_SWAP+LOOP_UNROLL+LOOP_UNROLL2+COMPILER_ASSIST] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DCOMPILER_ASSIST:
	 Average time = 0.008420 s, Stddev = 0.000211 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 129.48x, speedup vs O3: 24.73x

O3_ZNVER4_C4[LOOP_SWAP+LOOP_UNROLL+LOOP_UNROLL2+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUSION:
	 Average time = 0.012507 s, Stddev = 0.000842 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 87.16x, speedup vs O3: 16.64x

O3_ZNVER4_C4[LOOP_SWAP+LOOP_UNROLL+FUNC_INLINE+COMPILER_ASSIST] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DFUNC_INLINE -DCOMPILER_ASSIST:
	 Average time = 0.008265 s, Stddev = 0.000217 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 131.90x, speedup vs O3: 25.19x

O3_ZNVER4_C4[LOOP_SWAP+LOOP_UNROLL+FUNC_INLINE+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DFUNC_INLINE -DFUSION:
	 Average time = 0.011333 s, Stddev = 0.000081 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 96.19x, speedup vs O3: 18.37x

O3_ZNVER4_C4[LOOP_SWAP+LOOP_UNROLL+COMPILER_ASSIST+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DCOMPILER_ASSIST -DFUSION:
	 Average time = 0.009097 s, Stddev = 0.000348 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 119.84x, speedup vs O3: 22.88x

O3_ZNVER4_C4[LOOP_SWAP+LOOP_UNROLL2+FUNC_INLINE+COMPILER_ASSIST] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST:
	 Average time = 0.031526 s, Stddev = 0.000647 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 34.58x, speedup vs O3: 6.60x

O3_ZNVER4_C4[LOOP_SWAP+LOOP_UNROLL2+FUNC_INLINE+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL2 -DFUNC_INLINE -DFUSION:
	 Average time = 0.091795 s, Stddev = 0.000712 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 11.88x, speedup vs O3: 2.27x

O3_ZNVER4_C4[LOOP_SWAP+LOOP_UNROLL2+COMPILER_ASSIST+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL2 -DCOMPILER_ASSIST -DFUSION:
	 Average time = 0.031726 s, Stddev = 0.000264 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 34.36x, speedup vs O3: 6.56x

O3_ZNVER4_C4[LOOP_SWAP+FUNC_INLINE+COMPILER_ASSIST+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DFUNC_INLINE -DCOMPILER_ASSIST -DFUSION:
	 Average time = 0.033376 s, Stddev = 0.000182 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 32.66x, speedup vs O3: 6.24x

O3_ZNVER4_C4[LOOP_UNROLL+LOOP_UNROLL2+FUNC_INLINE+COMPILER_ASSIST] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST:
	 Average time = 0.314999 s, Stddev = 0.001197 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 3.46x, speedup vs O3: 0.66x

O3_ZNVER4_C4[LOOP_UNROLL+LOOP_UNROLL2+FUNC_INLINE+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DFUSION:
	 Average time = 0.353184 s, Stddev = 0.010240 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 3.09x, speedup vs O3: 0.59x

O3_ZNVER4_C4[LOOP_UNROLL+LOOP_UNROLL2+COMPILER_ASSIST+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL -DLOOP_UNROLL2 -DCOMPILER_ASSIST -DFUSION:
	 Average time = 0.444422 s, Stddev = 0.009628 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 2.45x, speedup vs O3: 0.47x

O3_ZNVER4_C4[LOOP_UNROLL+FUNC_INLINE+COMPILER_ASSIST+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL -DFUNC_INLINE -DCOMPILER_ASSIST -DFUSION:
	 Average time = 0.404932 s, Stddev = 0.002084 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 2.69x, speedup vs O3: 0.51x

O3_ZNVER4_C4[LOOP_UNROLL2+FUNC_INLINE+COMPILER_ASSIST+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST -DFUSION:
	 Average time = 0.428909 s, Stddev = 0.006334 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 2.54x, speedup vs O3: 0.49x

O3_ZNVER4_C5[LOOP_SWAP+LOOP_UNROLL+LOOP_UNROLL2+FUNC_INLINE+COMPILER_ASSIST] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST:
	 Average time = 0.008636 s, Stddev = 0.000255 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 126.23x, speedup vs O3: 24.10x

O3_ZNVER4_C5[LOOP_SWAP+LOOP_UNROLL+LOOP_UNROLL2+FUNC_INLINE+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DFUSION:
	 Average time = 0.012061 s, Stddev = 0.000756 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 90.39x, speedup vs O3: 17.26x

O3_ZNVER4_C5[LOOP_SWAP+LOOP_UNROLL+LOOP_UNROLL2+COMPILER_ASSIST+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DCOMPILER_ASSIST -DFUSION:
	 Average time = 0.008548 s, Stddev = 0.000242 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 127.53x, speedup vs O3: 24.35x

O3_ZNVER4_C5[LOOP_SWAP+LOOP_UNROLL+FUNC_INLINE+COMPILER_ASSIST+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DFUNC_INLINE -DCOMPILER_ASSIST -DFUSION:
	 Average time = 0.009266 s, Stddev = 0.000262 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 117.65x, speedup vs O3: 22.47x

O3_ZNVER4_C5[LOOP_SWAP+LOOP_UNROLL2+FUNC_INLINE+COMPILER_ASSIST+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST -DFUSION:
	 Average time = 0.033059 s, Stddev = 0.000364 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 32.98x, speedup vs O3: 6.30x

O3_ZNVER4_C5[LOOP_UNROLL+LOOP_UNROLL2+FUNC_INLINE+COMPILER_ASSIST+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST -DFUSION:
	 Average time = 0.429153 s, Stddev = 0.009399 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 2.54x, speedup vs O3: 0.49x

O3_ZNVER4_C6[LOOP_SWAP+LOOP_UNROLL+LOOP_UNROLL2+FUNC_INLINE+COMPILER_ASSIST+FUSION] :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST -DFUSION:
	 Average time = 0.009185 s, Stddev = 0.000231 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 118.69x, speedup vs O3: 22.66x

O0_ALLDEFS :: -O0 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST -DFUSION:
	 Average time = 0.864861 s, Stddev = 0.006925 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 1.26x, speedup vs O3: 0.24x

FAST_ALLDEFS :: -fast -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST -DFUSION:
	 (insufficient data)

O3_ZNVER4_ALLDEFS :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST -DFUSION:
	 Average time = 0.009303 s, Stddev = 0.000162 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 117.18x, speedup vs O3: 22.38x
     ```