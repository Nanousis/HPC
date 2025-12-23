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


