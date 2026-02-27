// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Tim Fischer <fischeti@iis.ee.ethz.ch>
//         Luca Colagrande <colluca@iis.ee.ethz.ch>
//         Viviane Potocnik <vivianep@iis.ee.ethz.ch>

#include <math.h>
#include <stdint.h>

#include "blas.h"

#include "data.h"
#include "snrt.h"

#ifdef BF16
#define DTYPE uint16_t
#else
#define DTYPE uint32_t
#endif

int main() {

#ifdef DUAL_SSR
    write_csr(3, (1 << 12)); // Enable dual SSRs for scales
#endif

    uint32_t fmode = 0;

#ifdef FP8ALT
    fmode |= 2;  // FP8ALT = 2
#endif
#ifdef FP6
    fmode |= 4;  // FP6 = 4
#endif
#ifdef FP6ALT
    fmode |= 6;  // FP6ALT = 6
#endif
#ifdef FP4
    fmode |= 8;  // FP4 = 8
#endif
#ifdef INT8
    fmode |= 10; // INT8 = 10
#endif

#ifdef BF16
    fmode |= 1;  // Bit 0 = 1 for BF16 destination
#endif

#ifdef BASE_ALT
    fmode |= 1; // Required for FP8ALT
#endif

    write_csr(2048, fmode);

    int retcode = mxgemm(&args);

    snrt_cluster_hw_barrier();

#ifdef BIST
    // Aliases
    uint32_t M = args.M;
    uint32_t N = args.N;

    int errors = M * N;

    if (snrt_cluster_core_idx() == 0) {
        for (uint32_t m = 0; m < M; m++) {
            for (uint32_t n = 0; n < N; n++) {
                uint32_t idx = m * N + n;

                DTYPE c_val = *((DTYPE*)&c[idx]);
                DTYPE r_val = *((DTYPE*)&result[idx]);

                if (c_val == r_val) {
                    errors--;
                } else {
                    printf("Mismatch at %d: C = 0x%04x, R = 0x%04x\n", idx, c_val, r_val);
                }
            }
        }
        printf("%d/%d Errors\n", errors, M * N);

        return errors;
    }
#endif

    return retcode;
}
