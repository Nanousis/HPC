Good to know that we do not need to store both position and velocity of the items


According to the nvidia nsight compute we got that we had shit tons of uncoalesced memory accesses.
to solve this we tried padding the body array since it was 24 byte long and each memory transaction is 32byte wide. This did not work at all... Instead we lost performance.

So we made a structure of arrays instead to leverage the power of coalesced memory


talk about floating point approximations 

        distSqr = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, SOFTENING)));
        invDist = rsqrtf(distSqr);
        invDist3 = invDist * invDist * invDist;
        fx = fmaf(dx, invDist3, fx);
        fy = fmaf(dy, invDist3, fy);
        fz = fmaf(dz, invDist3, fz);

Apparently after converting the warps from 1024 threads to 512 and even 256, we get a substantial performance increase

Unrolling seemeed to not offer any substantial help and instead even made the performance worse


One further optimization we tried that was very fruitfull was using multiple GPUs. With multiple GPUs we actually got the time down as many times as the gpus we were using.

In the end we got stuck to 0.38s / 120BI/s.
we slept and we tried something new. Instead of doing this fine grained parallelization we tried to do one thread working on a single body instead of multiple threads working on multiple bodies...... We got the same performance with even a tiny bit of speedup. Now the performance was 130Bi/s

After this the code was WAYYYYYYYYY simpler....
We tried unrolling. That pushed the performance even further going from 130B to 160Bi/s


After that we revesited tiling. Doing CUDA_BLOCK_SIZE tiles every time. This increased the throughtput of the program by a lot. It sped up a further 3x going from 160Bi/s to 424Bi/s 
We also tweaked the CUDA_BLOCK_SIZE to find the best time possible. This ended up being 256 threads per block.

[01/12] Total=0.115000s  Throughput=372.950 BIPS
[02/12] Total=0.101000s  Throughput=427.241 BIPS
[03/12] Total=0.100000s  Throughput=427.398 BIPS
[04/12] Total=0.101000s  Throughput=426.935 BIPS
[05/12] Total=0.101000s  Throughput=425.843 BIPS
[06/12] Total=0.102000s  Throughput=421.956 BIPS
[07/12] Total=0.101000s  Throughput=424.765 BIPS
[08/12] Total=0.101000s  Throughput=424.992 BIPS
[09/12] Total=0.102000s  Throughput=422.214 BIPS
[10/12] Total=0.102000s  Throughput=423.049 BIPS
[11/12] Total=0.102000s  Throughput=422.924 BIPS
[12/12] Total=0.102000s  Throughput=421.774 BIPS

=== Trimmed results (discard lowest & highest throughput run) ===
Discarded (lowest throughput): run #01 -> 372.950 BIPS, Total=0.115000s
Discarded (highest throughput): run #03 -> 427.398 BIPS, Total=0.100000s

Kept runs: 10 / 12

Throughput (Billion Interactions / second):
  mean = 424.169 BIPS
  std  = 2.060 BIPS  (sample)

Total GPU Time (seconds) on kept runs:
  mean = 0.101500 s
  std  = 0.000527 s  (sample)

128 block size 
Throughput (Billion Interactions / second):
  mean = 421.841 BIPS
  std  = 2.353 BIPS  (sample)

Total GPU Time (seconds) on kept runs:
  mean = 0.101800 s
  std  = 0.000789 s  (sample)

  512 block size
  Throughput (Billion Interactions / second):
  mean = 405.492 BIPS
  std  = 13.489 BIPS  (sample)

Total GPU Time (seconds) on kept runs:
  mean = 0.106000 s
  std  = 0.003801 s  (sample)