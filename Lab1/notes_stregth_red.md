### Loop strength reduction seems to be increasing performance minorly from the invariance and subexpression but is still slower than not including any of them

total speedup on O0 with strength reduction, invariant code and subexpression elimination is 1.27x to 1.33X

```
=====STRENGTH REDUCTION=======

O0_ALLDEFS :: -O0 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST:
	 Average time = 0.818604 s, Stddev = 0.010051 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 1.33x, speedup vs O3: 0.26x

FAST_ALLDEFS :: -fast -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST:
	 (insufficient data)

O3_ZNVER4_ALLDEFS :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST:
	 Average time = 0.008857 s, Stddev = 0.000319 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 122.76x, speedup vs O3: 23.68x
```



vs 


```
====== WITHOUT STRENGTH REDUCTION
O0_ALLDEFS :: -O0 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST:
	 Average time = 0.826792 s, Stddev = 0.008028 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 1.31x, speedup vs O3: 0.25x

FAST_ALLDEFS :: -fast -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST:
	 (insufficient data)

O3_ZNVER4_ALLDEFS :: -O3 -march=znver4 -mtune=znver4 -DLOOP_SWAP -DLOOP_UNROLL -DLOOP_UNROLL2 -DFUNC_INLINE -DCOMPILER_ASSIST:
	 Average time = 0.009150 s, Stddev = 0.000264 s, PSNR(avg±std) = inf ± N/A, speedup vs O0: 118.69x, speedup vs O3: 23.01x
```