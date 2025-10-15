### Loop invariant code / common sub expression elimination it seems to not change the performance of the code much and seems to minorly hinder performance.
It does increase the O0 timings though from 1.27 to 1.31X

```
====== WITHOUT STRENGTH REDUCTION
O0_ALLDEFS :: -O0 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST:
	 Average time = 0.826792 s, Stddev = 0.008028 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 1.31x, speedup vs O3: 0.25x

FAST_ALLDEFS :: -fast -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST:
	 (insufficient data)

O3_ZNVER4_ALLDEFS :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST:
	 Average time = 0.009150 s, Stddev = 0.000264 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 118.69x, speedup vs O3: 23.01x
```



vs 


```
O0_ALLDEFS :: -O0 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST:
	 Average time = 0.853979 s, Stddev = 0.000000 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 1.27x, speedup vs O3: 0.25x

FAST_ALLDEFS :: -fast -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST:
	 (insufficient data)

O3_ZNVER4_ALLDEFS :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST:
	 Average time = 0.008324 s, Stddev = 0.000000 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 130.30x, speedup vs O3: 25.16x
```