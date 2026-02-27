// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Tim Fischer <fischeti@iis.ee.ethz.ch>
//         Luca Bertaccini <lbertaccini@iis.ee.ethz.ch>
//         Luca Colagrande <colluca@iis.ee.ethz.ch>
//         Viviane Potocnik <vivianep@iis.ee.ethz.ch>

#ifdef MXGEMM_APP
#include "config.h"
#endif

#ifdef BF16
#define STORE_INSTR \
    "fsh %[c0], 0(%[C]) \n" \
    "fsh %[c1], 2(%[C]) \n" \
    "fsh %[c2], 4(%[C]) \n" \
    "fsh %[c3], 6(%[C]) \n" \
    "fsh %[c4], 8(%[C]) \n" \
    "fsh %[c5], 10(%[C]) \n" \
    "fsh %[c6], 12(%[C]) \n" \
    "fsh %[c7], 14(%[C]) \n"
#else
#define STORE_INSTR \
    "fsw %[c0], 0(%[C]) \n" \
    "fsw %[c1], 4(%[C]) \n" \
    "fsw %[c2], 8(%[C]) \n" \
    "fsw %[c3], 12(%[C]) \n" \
    "fsw %[c4], 16(%[C]) \n" \
    "fsw %[c5], 20(%[C]) \n" \
    "fsw %[c6], 24(%[C]) \n" \
    "fsw %[c7], 28(%[C]) \n"
#endif

void mxgemm_fp8_opt_ex(uint32_t M, uint32_t N, uint32_t K, void* A_p,
                     uint32_t ldA, uint32_t ta, void* B_p, uint32_t ldB,
                     uint32_t tb, void* C_p, uint32_t ldC, uint32_t BETA,
                     uint32_t setup_SSR, uint32_t dual_ssr,
                     void* S_p, uint32_t ldS, uint32_t block_size,
                     void* S_pa, void* S_pb, uint32_t ldSab, uint32_t sub_byte) {
    char* A = (char*)A_p;
    char* B = (char*)B_p;
    uint8_t* S = (uint8_t*)S_p;
    uint8_t* Sa = (uint8_t*)S_pa;
    uint8_t* Sb = (uint8_t*)S_pb;
    // Unrolling factor of most inner loop.
    // Should be at least as high as the FMA delay
    // for maximum utilization
    const uint32_t unroll = 8;

    // SSR strides and bounds only have to be configured
    // once in the beginning
    if (setup_SSR) {
        // Operand A
        uint32_t ssr0_b[4] = {unroll, K * 6 / 8 / sub_byte, N / unroll, M};
        uint32_t ssr0_i[4] = {0, sizeof(char) * 8, 0, sizeof(char) * ldA};

        // Operand B
        uint32_t ssr1_b[4] = {unroll, K * 6 / 8 / sub_byte, N / unroll, M};
        uint32_t ssr1_i[4] = {sizeof(char) * ldB, sizeof(char) * 8,
                              sizeof(char) * unroll * ldB, 0};

        // Configure SSR for A
        snrt_ssr_loop_3d(SNRT_SSR_DM0, ssr0_b[1], ssr0_b[2], ssr0_b[3],
                         ssr0_i[1], ssr0_i[2], ssr0_i[3]);
        snrt_ssr_repeat(SNRT_SSR_DM0, unroll);

        // Configure SSR for B
        snrt_ssr_loop_4d(SNRT_SSR_DM1, ssr1_b[0], ssr1_b[1], ssr1_b[2],
                         ssr1_b[3], ssr1_i[0], ssr1_i[1], ssr1_i[2], ssr1_i[3]);

        // Scales
        if (dual_ssr) {
          // Scale A
          uint32_t ssr2_b[3] = {K / block_size, N / unroll, M};
          uint32_t ssr2_i[3] = {sizeof(uint8_t), 0, sizeof(uint8_t) * ldSab};
          // Scale B
          uint32_t ssr3_b[4] = {unroll, K / block_size, N / unroll, M};
          uint32_t ssr3_i[4] = {sizeof(uint8_t), sizeof(uint8_t) * N, sizeof(uint8_t) * unroll, 0};

          // Configure SSR for Scale A
          snrt_ssr_loop_3d(SNRT_SSR_DM2, ssr2_b[0], ssr2_b[1], ssr2_b[2],
                          ssr2_i[0], ssr2_i[1], ssr2_i[2]);
          snrt_ssr_repeat(SNRT_SSR_DM2, unroll * block_size / 8 * 6 / sub_byte);

          // Configure SSR for Scale B
          snrt_ssr_loop_3d(SNRT_SSR_DM3, ssr3_b[1], ssr3_b[2], ssr3_b[3],
                          ssr3_i[1], ssr3_i[2], ssr3_i[3]);
          snrt_ssr_repeat(SNRT_SSR_DM3, unroll * block_size / 8 * 6 / sub_byte); // Max block size is 64 (ssr rep bit width=6)
        } else {
          uint32_t ssr2_b[5] = {unroll / 4, block_size * 6 / 8 / sub_byte, K / block_size, N / unroll, M};
          uint32_t ssr2_i[5] = {2 * sizeof(uint8_t) * 4, 0,
                                2 * sizeof(uint8_t) * N, 2 * sizeof(uint8_t) * unroll,
                                2 * sizeof(uint8_t) * ldS};
          // Configure SSR for Scales
          snrt_ssr_loop_4d(SNRT_SSR_DM2, ssr2_b[0], ssr2_b[1], ssr2_b[2],
                          ssr2_b[3], ssr2_i[0], ssr2_i[1], ssr2_i[2], ssr2_i[3]);
          snrt_ssr_repeat(SNRT_SSR_DM2, 4);
        }
    }

    // SSR start address need to be configured each time
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, A);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, B);
    if (dual_ssr) {
      snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_3D, Sa);
      snrt_ssr_read(SNRT_SSR_DM3, SNRT_SSR_3D, Sb);
    }
    
    snrt_ssr_enable();

    // Kernel progresses by 8 values each step
    const uint32_t n_frep = K * 6 / 8 / sub_byte - 1;

    for (uint32_t m = 0; m < M; m++) {
        uint32_t n = 0;
        if (!dual_ssr) {
          snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_4D, (S+m*2*ldS));
        }
        for (uint32_t n0 = 0; n0 < N / unroll; n0++) {
#ifdef BF16
            __fp16* _C = &((__fp16*)C_p)[m * ldC + n];
            v4f16 c[unroll];
#else
            float* _C = ((float*)C_p) + m * ldC + n;
            float c[unroll];
#endif
            const register float zero = 0.0;
            asm volatile(
                // Initialize SIMD vector with zeros
                "vfcpka.s.s %[c0], %[zero], %[zero]\n"
                "vfcpka.s.s %[c1], %[zero], %[zero]\n"
                "vfcpka.s.s %[c2], %[zero], %[zero]\n"
                "vfcpka.s.s %[c3], %[zero], %[zero]\n"
                "vfcpka.s.s %[c4], %[zero], %[zero]\n"
                "vfcpka.s.s %[c5], %[zero], %[zero]\n"
                "vfcpka.s.s %[c6], %[zero], %[zero]\n"
                "vfcpka.s.s %[c7], %[zero], %[zero]\n"
                // Perform expanding sum-dotproducts
                "frep.o  %[n_frep], %[unroll], 0, 0 \n"
                "mxdotp.h.b0 %[c0], ft0, ft1, ft2 \n"
                "mxdotp.h.b1 %[c1], ft0, ft1, ft2 \n"
                "mxdotp.h.b2 %[c2], ft0, ft1, ft2 \n"
                "mxdotp.h.b3 %[c3], ft0, ft1, ft2 \n"
                "mxdotp.h.b0 %[c4], ft0, ft1, ft2 \n"
                "mxdotp.h.b1 %[c5], ft0, ft1, ft2 \n"
                "mxdotp.h.b2 %[c6], ft0, ft1, ft2 \n"
                "mxdotp.h.b3 %[c7], ft0, ft1, ft2 \n"
                // Store results back
                STORE_INSTR
                : [ c0 ] "+f"(c[0]), [ c1 ] "+f"(c[1]), [ c2 ] "+f"(c[2]),
                  [ c3 ] "+f"(c[3]), [ c4 ] "+f"(c[4]), [ c5 ] "+f"(c[5]),
                  [ c6 ] "+f"(c[6]), [ c7 ] "+f"(c[7])
                : [ C ] "r"(_C), [ n_frep ] "r"(n_frep), [ beta ] "r"(BETA),
                  [ unroll ] "i"(unroll), [ zero ] "f"(zero)
                : "ft0", "ft1", "ft2", "ft3");
            n += unroll;
        }
    }

    snrt_ssr_disable();
}

void mxgemm_fp8_fp32_baseline(uint32_t M, uint32_t N, uint32_t K, void* A_p,
                        uint32_t ldA, uint32_t ta, void* B_p, uint32_t ldB,
                        uint32_t tb, void* C_p, uint32_t ldC, uint32_t BETA,
                        uint32_t setup_SSR, uint32_t dual_ssr,
                        void* S_p, uint32_t ldS, uint32_t block_size, 
                        void* S_pa, void* S_pb, uint32_t ldSab, uint32_t sub_byte) {
    char* A = (char*)A_p;
    char* B = (char*)B_p;
    float* C = (float*)C_p;
    uint8_t* Sa = (uint8_t*)S_pa;
    uint8_t* Sb = (uint8_t*)S_pb;

    volatile register uint8_t *scale_ptr_a, *scale_ptr_b;

    for (uint32_t m = 0; m < M; m++) {
        scale_ptr_b = Sb; // go back to the beginning of the scale array
        for (uint32_t n = 0; n < N; n++) {
            volatile register char *a_ptr, *b_ptr;
            volatile float* c_ptr;
            
            const register float zero = 0.0;
            uint32_t repeat = K / block_size;

            volatile float acc;

            a_ptr = (char*)(&A[m * ldA]);
            b_ptr = (char*)(&B[n * ldB]);
            c_ptr = &C[m * ldC + n];

            // Load scales
            scale_ptr_a = (uint8_t*)(&Sa[m * ldSab]);

            // Initialize accumulator
            acc = 0.0;
            
            asm volatile(
                // Initialize outer repeat counter
                "mv t4, %[repeat]\n"
                // Initialize accumulator
                "fmv.s fa1, %[zero]\n"
                "2:\n" // Outer repeat loop
                // Initialize loop counter
                "li     t0, 0 \n"
                // Initialize accumulators
                "fmv.s ft8, %[zero] \n"
                "fmv.s ft9, %[zero] \n"
                "fmv.s ft10, %[zero] \n"
                "fmv.s ft11, %[zero] \n"

                // Load scales and compute scale factor
                "lbu t1, 0(%[scale_ptr_a])\n"   // Load scale_a
                "lbu t2, 0(%[scale_ptr_b])\n"   // Load scale_b
                "add t3, t1, t2\n"            // Add scales
                "addi t3, t3, -127\n"         // Subtract bias (127)

                "add %[scale_ptr_a], %[scale_ptr_a], 1\n" // Advance scale pointer
                "add %[scale_ptr_b], %[scale_ptr_b], 1\n" // Advance scale pointer

                // Main MAC loop
                "3: \n"
                "flb ft0, 0(%[a_ptr]) \n"    // Load FP8 from A
                "flb ft1, 0(%[b_ptr]) \n"    // Load FP8 from B
                "flb ft2, 1(%[a_ptr]) \n"
                "flb ft3, 1(%[b_ptr]) \n"
                "flb ft4, 2(%[a_ptr]) \n"
                "flb ft5, 2(%[b_ptr]) \n"
                "flb ft6, 3(%[a_ptr]) \n"
                "flb ft7, 3(%[b_ptr]) \n"
                
                "fcvt.s.b ft0, ft0\n"       // Convert FP8 -> FP32 (A)
                "fcvt.s.b ft1, ft1\n"       // Convert FP8 -> FP32 (B)
                "fcvt.s.b ft2, ft2\n"       // Convert FP8 -> FP32 (A)
                "fcvt.s.b ft3, ft3\n"       // Convert FP8 -> FP32 (B)
                "fcvt.s.b ft4, ft4\n"       // Convert FP8 -> FP32 (A)
                "fcvt.s.b ft5, ft5\n"       // Convert FP8 -> FP32 (B)
                "fcvt.s.b ft6, ft6\n"       // Convert FP8 -> FP32 (A)
                "fcvt.s.b ft7, ft7\n"       // Convert FP8 -> FP32 (B)

                "fmadd.s ft8, ft0, ft1, ft8 \n"   // Accumulate: ft3 += ft0 * ft1
                "fmadd.s ft9, ft2, ft3, ft9 \n"   // Accumulate: ft3 += ft0 * ft1
                "fmadd.s ft10, ft4, ft5, ft10 \n"   // Accumulate: ft3 += ft0 * ft1
                "fmadd.s ft11, ft6, ft7, ft11 \n"   // Accumulate: ft3 += ft0 * ft1

                "add %[a_ptr], %[a_ptr], 4 \n"   // Advance A pointer
                "add %[b_ptr], %[b_ptr], 4 \n"   // Advance B pointer
                "addi  t0, t0, 4 \n"             // Increment loop counter by 8
                "blt   t0, %[block_size], 3b \n" // Continue if t0 < BLOCK_SIZE

                // Compute scaling factor 2^(sa + sb)
                "slli t3, t3, 23 \n"      // Shift scale to exponent position
                "fmv.w.x ft0, t3 \n"      // Move scale (sa + sb) into ft0

                // Reduce result
                "fadd.s ft8, ft8, ft9 \n"      // Reduce vector to scalar
                "fadd.s ft10, ft10, ft11 \n"    // Reduce vector to scalar
                "fadd.s ft8, ft8, ft10 \n"            // Reduce vector to scalar
                "fmadd.s fa1, ft8, ft0, fa1 \n"    // Scale result

                // Decrement outer repeat counter
                "addi t4, t4, -1\n"
                "bnez t4, 2b\n"              // Repeat if t4 != 0

                // Store result
                "fsw fa1, 0(%[C]) \n"        // Store result into C

                : [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr),
                  [ scale_ptr_a ] "+r"(scale_ptr_a), [ scale_ptr_b ] "+r"(scale_ptr_b)
                : [ C ] "r"(c_ptr), [ K ] "r"(K), [ zero ] "f"(zero),
                  [block_size] "r"(block_size), [repeat] "r"(repeat)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8", "ft9", "ft10", "ft11", "fa1", "fa2",
                  "t0", "t1", "t2", "t3", "t4");
        }
    }
}

void mxgemm_fp8_bf16_baseline(uint32_t M, uint32_t N, uint32_t K, void* A_p,
                        uint32_t ldA, uint32_t ta, void* B_p, uint32_t ldB,
                        uint32_t tb, void* C_p, uint32_t ldC, uint32_t BETA,
                        uint32_t setup_SSR, uint32_t dual_ssr,
                        void* S_p, uint32_t ldS, uint32_t block_size, 
                        void* S_pa, void* S_pb, uint32_t ldSab, uint32_t sub_byte) {
    char* A = (char*)A_p;
    char* B = (char*)B_p;
    // float* C = (float*)C_p;
    __fp16* C = (__fp16*)C_p;
    uint8_t* Sa = (uint8_t*)S_pa;
    uint8_t* Sb = (uint8_t*)S_pb;

    write_csr(2048, 3); // Enable BF16 -> This also forces FP8ALT

    volatile register uint8_t *scale_ptr_a, *scale_ptr_b;

    for (uint32_t m = 0; m < M; m++) {
        scale_ptr_b = Sb; // go back to the beginning of the scale array
        for (uint32_t n = 0; n < N; n++) {
            volatile register char *a_ptr, *b_ptr;
            volatile v4f16* c_ptr;
            
            const register float zero = 0.0;
            const register __fp16 zero_h = 0.0;
            uint32_t repeat = K / block_size;

            volatile float acc;

            a_ptr = (char*)(&A[m * ldA]);
            b_ptr = (char*)(&B[n * ldB]);
            c_ptr = &C[m * ldC + n];

            // Load scales
            scale_ptr_a = (uint8_t*)(&Sa[m * ldSab]);

            // Initialize accumulator
            acc = 0.0;
            
            asm volatile(
                // Initialize outer repeat counter
                "mv t4, %[repeat]\n"
                // Initialize accumulator with zero
                "fmv.h.x fa1, x0\n"
                "2:\n" // Outer repeat loop
                // Initialize loop counter
                "li     t0, 0 \n"
                // Initialize accumulators
                "fmv.h.x ft8, x0\n"
                "fmv.h.x ft9, x0\n"
                "fmv.h.x ft10, x0\n"
                "fmv.h.x ft11, x0\n"

                // Load scales and compute scale factor
                "lbu t1, 0(%[scale_ptr_a])\n"   // Load scale_a
                "lbu t2, 0(%[scale_ptr_b])\n"   // Load scale_b
                "add t3, t1, t2\n"            // Add scales
                "addi t3, t3, -127\n"         // Subtract bias (127)

                "add %[scale_ptr_a], %[scale_ptr_a], 1\n" // Advance scale pointer
                "add %[scale_ptr_b], %[scale_ptr_b], 1\n" // Advance scale pointer

                // Main MAC loop
                "3: \n"
                "flb ft0, 0(%[a_ptr]) \n"    // Load FP8 from A
                "flb ft1, 0(%[b_ptr]) \n"    // Load FP8 from B
                "flb ft2, 1(%[a_ptr]) \n"
                "flb ft3, 1(%[b_ptr]) \n"
                "flb ft4, 2(%[a_ptr]) \n"
                "flb ft5, 2(%[b_ptr]) \n"
                "flb ft6, 3(%[a_ptr]) \n"
                "flb ft7, 3(%[b_ptr]) \n"
                
                "fcvt.ah.b ft0, ft0\n"       // Convert FP8 -> BF16 (A)
                "fcvt.ah.b ft1, ft1\n"       // Convert FP8 -> BF16 (B)
                "fcvt.ah.b ft2, ft2\n"       // Convert FP8 -> BF16 (A)
                "fcvt.ah.b ft3, ft3\n"       // Convert FP8 -> BF16 (B)
                "fcvt.ah.b ft4, ft4\n"       // Convert FP8 -> BF16 (A)
                "fcvt.ah.b ft5, ft5\n"       // Convert FP8 -> BF16 (B)
                "fcvt.ah.b ft6, ft6\n"       // Convert FP8 -> BF16 (A)
                "fcvt.ah.b ft7, ft7\n"       // Convert FP8 -> BF16 (B)

                "fmadd.ah ft8, ft0, ft1, ft8 \n"   // Accumulate: ft3 += ft0 * ft1
                "fmadd.ah ft9, ft2, ft3, ft9 \n"   // Accumulate: ft3 += ft0 * ft1
                "fmadd.ah ft10, ft4, ft5, ft10 \n"   // Accumulate: ft3 += ft0 * ft1
                "fmadd.ah ft11, ft6, ft7, ft11 \n"   // Accumulate: ft3 += ft0 * ft1

                "add %[a_ptr], %[a_ptr], 4 \n"   // Advance A pointer
                "add %[b_ptr], %[b_ptr], 4 \n"   // Advance B pointer
                "addi  t0, t0, 4 \n"             // Increment loop counter by 8
                "blt   t0, %[block_size], 3b \n" // Continue if t0 < BLOCK_SIZE

                // Compute scaling factor 2^(sa + sb)
                "slli t3, t3, 7 \n"      // Shift scale to exponent position
                "fmv.h.x ft0, t3 \n"      // Move scale (sa + sb) into ft0

                // Reduce result
                "fadd.ah ft8, ft8, ft9 \n"      // Reduce
                "fadd.ah ft10, ft10, ft11 \n"    // Reduce
                "fadd.ah ft8, ft8, ft10 \n"
                "fmadd.ah fa1, ft8, ft0, fa1 \n"    // Scale result

                // Decrement outer repeat counter
                "addi t4, t4, -1\n"
                "bnez t4, 2b\n"              // Repeat if t4 != 0

                // Store result
                "fsh fa1, 0(%[C]) \n"        // Store result into C

                : [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr),
                  [ scale_ptr_a ] "+r"(scale_ptr_a), [ scale_ptr_b ] "+r"(scale_ptr_b)
                : [ C ] "r"(c_ptr), [ K ] "r"(K), [ zero ] "f"(zero),
                  [block_size] "r"(block_size), [repeat] "r"(repeat)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8", "ft9", "ft10", "ft11", "fa1", "fa2", "fa3",
                  "t0", "t1", "t2", "t3", "t4");
        }
    }
}

void mxgemm_fp8_bf16_exsdotp(uint32_t M, uint32_t N, uint32_t K, void* A_p,
                        uint32_t ldA, uint32_t ta, void* B_p, uint32_t ldB,
                        uint32_t tb, void* C_p, uint32_t ldC, uint32_t BETA,
                        uint32_t setup_SSR, uint32_t dual_ssr,
                        void* S_p, uint32_t ldS, uint32_t block_size, 
                        void* S_pa, void* S_pb, uint32_t ldSab, uint32_t sub_byte) {
    char* A = (char*)A_p;
    char* B = (char*)B_p;
    __fp16* C = (__fp16*)C_p;
    uint8_t* Sa = (uint8_t*)S_pa;
    uint8_t* Sb = (uint8_t*)S_pb;

    const uint32_t unroll = 4;
    const uint32_t n_frep = block_size / 32 - 1;
    const uint32_t repeat = K / block_size;

    write_csr(2048, 3); // Enable BF16 -> This also forces FP8ALT

    volatile register uint8_t *scale_ptr_a, *scale_ptr_b;

    snrt_ssr_loop_4d(SNRT_SSR_DM0, block_size/8, repeat, N, M, 8, block_size, 0, ldA);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, A);

    snrt_ssr_loop_4d(SNRT_SSR_DM1, block_size/8, repeat, N, M, 8, block_size, ldB, 0);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, B);
    snrt_ssr_enable();

    for (uint32_t m = 0; m < M; m++) {
        scale_ptr_b = Sb; // go back to the beginning of the scale array
        for (uint32_t n = 0; n < N; n++) {
            volatile v4f16* c_ptr;
            
            const register float zero = 0.0;
            volatile float acc;

            c_ptr = &C[m * ldC + n];

            // Load scales
            scale_ptr_a = (uint8_t*)(&Sa[m * ldSab]);

            // Initialize accumulator
            acc = 0.0;

            asm volatile(
                // Initialize outer repeat counter
                "mv t4, %[repeat]\n"
                // Initialize accumulator
                "fmv.s fa1, %[zero]\n"
                "fcvt.h.s fa1, fa1\n"
                "2:\n" // Outer repeat loop
                // Initialize accumulators
                "vfcpka.s.s ft8, %[zero], %[zero]\n"
                "vfcpka.s.s ft9, %[zero], %[zero]\n"
                "vfcpka.s.s ft10, %[zero], %[zero]\n"
                "vfcpka.s.s ft11, %[zero], %[zero]\n"

                // Load scales and compute scale factor
                "lbu t1, 0(%[scale_ptr_a])\n"   // Load scale_a
                "lbu t2, 0(%[scale_ptr_b])\n"   // Load scale_b
                "add t3, t1, t2\n"            // Add scales
                "addi t3, t3, -127\n"         // Subtract bias (127)

                "add %[scale_ptr_a], %[scale_ptr_a], 1\n" // Advance scale pointer
                "add %[scale_ptr_b], %[scale_ptr_b], 1\n" // Advance scale pointer

                // Main MAC loop
                "frep.o  %[n_frep], %[unroll], 0, 0 \n"
                "vfdotpex.h.b ft8, ft0, ft1 \n"   // Accumulate: ft3 += ft0 * ft1
                "vfdotpex.h.b ft9, ft0, ft1 \n"   // Accumulate: ft3 += ft0 * ft1
                "vfdotpex.h.b ft10, ft0, ft1 \n"   // Accumulate: ft3 += ft0 * ft1
                "vfdotpex.h.b ft11, ft0, ft1 \n"   // Accumulate: ft3 += ft0 * ft1

                // Compute scaling factor 2^(sa + sb)
                "slli t3, t3, 7 \n"      // Shift scale to exponent position
                "fmv.h.x ft3, t3 \n"      // Move scale (sa + sb) into ft3

                // Reduce result
                "vfadd.ah ft8, ft8, ft9 \n"      // Vector add
                "vfadd.ah ft10, ft10, ft11 \n"   // Vector add
                "vfcpka.s.s fa2, %[zero], %[zero]\n"
                "vfadd.ah ft8, ft8, ft10 \n"     // Vector add
                "fmv.s fa3, %[zero]\n"
                "vfsumex.s.h fa2, ft8 \n"
                "vfsum.s fa3, fa2 \n"           // Reduce vector to scalar
                "fcvt.h.s fa2, fa3 \n"           // Convert to BF16
                "fmadd.ah fa1, fa2, ft3, fa1 \n"    // Scale result

                // Decrement outer repeat counter
                "addi t4, t4, -1\n"
                "bnez t4, 2b\n"              // Repeat if t4 != 0

                // Store result
                "fsh fa1, 0(%[C]) \n"        // Store result into C

                : [ scale_ptr_a ] "+r"(scale_ptr_a), [ scale_ptr_b ] "+r"(scale_ptr_b)
                : [ C ] "r"(c_ptr), [ K ] "r"(K), [ zero ] "f"(zero),
                  [block_size] "r"(block_size), [repeat] "r"(repeat),
                  [ n_frep ] "r"(n_frep), [ unroll ] "i"(unroll)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8", "ft9", "ft10", "ft11", "fa1", "fa2", "fa3",
                  "t1", "t2", "t3", "t4");
        }
    }
}

void mxgemm_fp8_fp32_exsdotp(uint32_t M, uint32_t N, uint32_t K, void* A_p,
                        uint32_t ldA, uint32_t ta, void* B_p, uint32_t ldB,
                        uint32_t tb, void* C_p, uint32_t ldC, uint32_t BETA,
                        uint32_t setup_SSR, uint32_t dual_ssr,
                        void* S_p, uint32_t ldS, uint32_t block_size, 
                        void* S_pa, void* S_pb, uint32_t ldSab, uint32_t sub_byte) {
    char* A = (char*)A_p;
    char* B = (char*)B_p;
    float* C = (float*)C_p;
    uint8_t* Sa = (uint8_t*)S_pa;
    uint8_t* Sb = (uint8_t*)S_pb;

    const uint32_t unroll = 12;
    const uint32_t n_frep = block_size / 16 - 1;
    const uint32_t repeat = K / block_size;

    volatile register uint8_t *scale_ptr_a, *scale_ptr_b;

    snrt_ssr_loop_4d(SNRT_SSR_DM0, block_size/8, repeat, N, M, 8, block_size, 0, ldA);
    snrt_ssr_repeat(SNRT_SSR_DM0, 2);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, A);

    snrt_ssr_loop_4d(SNRT_SSR_DM1, block_size/8, repeat, N, M, 8, block_size, ldB, 0);
    snrt_ssr_repeat(SNRT_SSR_DM1, 2);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, B);
    snrt_ssr_enable();

    for (uint32_t m = 0; m < M; m++) {
        scale_ptr_b = Sb; // go back to the beginning of the scale array
        for (uint32_t n = 0; n < N; n++) {
            volatile v2f32* c_ptr;
            
            const register float zero = 0.0;
            volatile float acc;

            c_ptr = &C[m * ldC + n];

            // Load scales
            scale_ptr_a = (uint8_t*)(&Sa[m * ldSab]);

            // Initialize accumulator
            acc = 0.0;

            asm volatile(
                // Initialize outer repeat counter
                "mv t4, %[repeat]\n"
                // Initialize accumulator
                "fmv.s fa1, %[zero]\n"
                "2:\n" // Outer repeat loop
                // Initialize accumulators
                "vfcpka.s.s ft8, %[zero], %[zero]\n"
                "vfcpka.s.s ft9, %[zero], %[zero]\n"
                "vfcpka.s.s ft10, %[zero], %[zero]\n"
                "vfcpka.s.s ft11, %[zero], %[zero]\n"

                // Load scales and compute scale factor
                "lbu t1, 0(%[scale_ptr_a])\n"   // Load scale_a
                "lbu t2, 0(%[scale_ptr_b])\n"   // Load scale_b
                "add t3, t1, t2\n"            // Add scales
                "addi t3, t3, -127\n"         // Subtract bias (127)

                "add %[scale_ptr_a], %[scale_ptr_a], 1\n" // Advance scale pointer
                "add %[scale_ptr_b], %[scale_ptr_b], 1\n" // Advance scale pointer

                // Main MAC loop
                "frep.o  %[n_frep], %[unroll], 0, 0 \n"
                "vfcvt.h.b fa4, ft0\n"       // Convert FP8 -> FP16 (A)
                "vfcvt.h.b ft3, ft1\n" 
                "vfcvtu.h.b ft4, ft0\n"       // Convert FP8 -> FP16 (A)
                "vfcvtu.h.b ft5, ft1\n"
                "vfcvt.h.b ft6, ft0\n"       // Convert FP8 -> FP16 (A)
                "vfcvt.h.b ft7, ft1\n"
                "vfcvtu.h.b fa2, ft0\n"       // Convert FP8 -> FP16 (A)
                "vfcvtu.h.b fa3, ft1\n"
                "vfdotpex.s.h ft8, fa4, ft3 \n"   // Accumulate: ft3 += ft0 * ft1
                "vfdotpex.s.h ft9, ft4, ft5 \n"   // Accumulate: ft3 += ft0 * ft1
                "vfdotpex.s.h ft10, ft6, ft7 \n"   // Accumulate: ft3 += ft0 * ft1
                "vfdotpex.s.h ft11, fa2, fa3 \n"   // Accumulate: ft3 += ft0 * ft1

                // Compute scaling factor 2^(sa + sb)
                "slli t3, t3, 23 \n"      // Shift scale to exponent position
                "fmv.s.x ft3, t3 \n"      // Move scale (sa + sb) into ft3

                // Reduce result
                "vfadd.s ft8, ft8, ft9 \n"      // Vector add
                "vfadd.s ft10, ft10, ft11 \n"    // Vector add
                "fmv.s fa2, %[zero]\n"
                "vfsum.s fa2, ft8 \n"            // Reduce vector to scalar
                "vfsum.s fa2, ft10 \n"           // Reduce vector to scalar
                "fmadd.s fa1, fa2, ft3, fa1 \n"    // Scale result

                // Decrement outer repeat counter
                "addi t4, t4, -1\n"
                "bnez t4, 2b\n"              // Repeat if t4 != 0

                // Store result
                "fsw fa1, 0(%[C]) \n"        // Store result into C

                : [ scale_ptr_a ] "+r"(scale_ptr_a), [ scale_ptr_b ] "+r"(scale_ptr_b)
                : [ C ] "r"(c_ptr), [ K ] "r"(K), [ zero ] "f"(zero),
                  [block_size] "r"(block_size), [repeat] "r"(repeat),
                  [ n_frep ] "r"(n_frep), [ unroll ] "i"(unroll)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8", "ft9", "ft10", "ft11", "fa1", "fa2", "fa3", "fa4",
                  "t1", "t2", "t3", "t4");
        }
    }
}
