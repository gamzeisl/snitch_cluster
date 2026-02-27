#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Authors: Tim Fischer     <fischeti@iis.ee.ethz.ch>
#          Luca Bertaccini <lbertaccini@iis.ee.ethz.ch>
#          Viviane Potocnik <vivianep@iis.ee.ethz.ch>
#          Luca Colagrande <colluca@iis.ee.ethz.ch>

import numpy as np
import re
import sys

import os

import snitch.util.sim.data_utils as du

# Add path to MX golden model library (set MX_GOLDEN_PATH to the mx_golden directory)
sys.path.append(os.environ['MX_GOLDEN_PATH'])
import golden.random_tests as mx_tests

np.random.seed(42)

unmerged_fp6 = False

def sixbit_chunks_to_64bit_words(chunks):
    """
    Args:
      chunks: iterable of strings, each exactly 6 characters long,
              containing '0' or '1' (msb-first).

    Returns:
      A list of 64-character strings, each '0'/'1', where each word
      is formed little-endian at the chunk granularity:
        - chunk[0]..chunk[5] → bits 0..5 of word 0
        - next chunk → bits 6..11, etc.
      If the total bit-length isn’t a multiple of 64, the final word
      is zero-padded at the *most* significant end.
    """
    WORD_MASK = (1 << 64) - 1
    words = []

    acc = 0         # accumulator for bits
    acc_bits = 0    # how many valid bits in acc

    for chunk in chunks:
        if len(chunk) != 6 or any(c not in '01' for c in chunk):
            raise ValueError(f"Invalid chunk: {chunk!r}")
        # interpret chunk as an integer (msb-first)
        val = int(chunk, 2)
        # OR it in at the next free bit-position:
        acc |= (val << acc_bits)
        acc_bits += 6

        # while we have at least 64 bits, peel off a word:
        while acc_bits >= 64:
            word = acc & WORD_MASK
            # format to 64-bit binary string (msb-first for display)
            words.append(f"{word:064b}")
            # shift out the 64 bits we just consumed
            acc >>= 64
            acc_bits -= 64

    # any leftover bits? flush as a final word (pad MSBs with zeros)
    if acc_bits > 0:
        words.append(f"{acc:0{acc_bits}b}".ljust(64, '0'))

    return words

def unpack_64b_chunks(chunk_list):
    return [
        (int(w, 2) >> shift) & 0xFF
        for w in chunk_list
        for shift in (0, 8, 16, 24, 32, 40, 48, 56)
    ]

class MXGemmDataGen(du.DataGen):

    # AXI splits bursts crossing 4KB address boundaries. To minimize
    # the occurrence of these splits the data should be aligned to 4KB
    BURST_ALIGNMENT = 4096

    def golden_model(self, alpha, a, b, beta, c):
        return alpha * np.matmul(a, b) + beta * c

    def exact_golden_model(self, alpha, a, b, beta, c, block_size, src_type, prec_dst, mx):
        M, N, K = a.shape[0], b.shape[1], b.shape[0]

        if src_type == 0:
            data_type = "FP8"
        elif src_type == 1:
            data_type = "FP8ALT"
        elif src_type == 2:
            data_type = "FP6"
        elif src_type == 3:
            data_type = "FP6ALT"
        elif src_type == 4:
            data_type = "FP4"
        elif src_type == 5:
            data_type = "INT8"

        if prec_dst == 4:
            acc_data_type = "FP32"
        elif prec_dst == 2:
            acc_data_type = "BF16"

        # if base implementation is used, increase vector size to block size to prevent wrong signs in case of inf
        vector_size = block_size if mx == 0 else 16 if data_type == 'FP4' else 8

        a, b, scales, result_hex, result_value, sa, sb, sb_t = mx_tests.test_gemm_fp9_sop_fp32_with_random_values(seed=42,
                            data_type=data_type,
                            acc_data_type=acc_data_type,
                            vector_size=vector_size,
                            scale_range=[-63, 63],
                            rows=M,
                            cols=N,
                            inner_dim=K,
                            block_size=block_size,
                            use_external_data=False)
        
        return np.array(result_value, dtype=np.float32)

    def infer_implementation(self, gemm_fp):
        # gemm_fp: "gemm_fp64_opt"
        # create a regex with fp_<type>_<implementation>
        prec, impl = re.search(r'gemm_fp(\d+)_(\w+)', gemm_fp).group(1, 2)
        return (int(prec) / 8), impl

    def validate(self, gemm_fp, parallelize_m,
                 parallelize_k, m_tiles, n_tiles, k_tiles, transa,
                 transb, M, N, K, beta, **kwargs):
        frac_m = M / m_tiles
        frac_n = N / n_tiles
        frac_k = K / k_tiles

        dtype, impl = self.infer_implementation(gemm_fp)

        # Calculate total TCDM occupation
        # Note: doesn't account for double buffering
        src_type = kwargs['src_type']
        if src_type == 'FP8' or src_type == 'FP8ALT' or src_type == 'INT8':
            src_bits = 8
        elif src_type == 'FP6' or src_type == 'FP6ALT':
            if unmerged_fp6:
                src_bits = 8
            else:
                src_bits = 6
        elif src_type == 'FP4':
            src_bits = 4

        dst_type = kwargs['dst_type']
        if dst_type == 'FP32':
            dst_bits = 32
        elif dst_type == 'BF16':
            dst_bits = 16

        mx = 1 if gemm_fp == ("mxgemm_fp8_opt_ex") else 0
        dual_ssr = kwargs['dual_ssr'] if 'dual_ssr' in kwargs else mx

        a_size = frac_m * frac_k * src_bits / 8
        b_size = frac_k * frac_n * src_bits / 8
        c_size = frac_m * frac_n * dst_bits / 8

        total_size = a_size
        total_size += b_size
        total_size += c_size

        if dual_ssr or not mx:
            scale_a_size = frac_m * frac_k / kwargs['block_size']
            scale_b_size = frac_k * frac_n / kwargs['block_size']
            total_size += scale_a_size
            total_size += scale_b_size
        else:
            scale_size = frac_m * frac_k / kwargs['block_size'] * frac_n * 2
            total_size += scale_size

        du.validate_tcdm_footprint(total_size)

        assert (M % m_tiles) == 0, 'M is not an integer multiple of tile size'
        assert (N % n_tiles) == 0, 'N is not an integer multiple of tile size'
        assert (K % k_tiles) == 0, 'K is not an integer multiple of tile size'
        assert not (parallelize_m and parallelize_k), 'Cannot parallelize K and M simultaneously'
        assert not transa, 'SIMD kernels don\'t support transposed A matrix'
        assert not transb or n_tiles == 1, 'Tiling in the N dimension not supported' \
            ' if B is transposed'
        assert not transb or k_tiles == 1, 'Tiling in the K dimension not supported' \
            ' if B is transposed'
        assert (impl == 'baseline') or (impl == 'naive') or frac_n >= 8, \
            'N dimension of tile size must be greater or equal to the unrolling factor (8) ' \
            'when using optimized kernels'
        assert beta == 0 or beta == 1, 'Only values of 0 or 1 supported for beta'
        assert not (src_bits != 8 and impl == "baseline"), 'No baseline implemented' \
            ' for data types different from FP8'
        assert not (dual_ssr and mx == 0), 'Dual SSR can be used only with mxgemm_fp8_opt_ex implementation'

    def emit_header(self, **kwargs):
        header = [super().emit_header()]

        # Validate parameters
        self.validate(**kwargs)

        M, N, K, block_size = kwargs['M'], kwargs['N'], kwargs['K'], kwargs['block_size']

        gemm_fp = kwargs['gemm_fp']
        prec, _ = self.infer_implementation(gemm_fp)

        src_type = kwargs['src_type']
        dst_type = kwargs['dst_type']

        if dst_type == 'FP32':
            dst_ctype = 'float'
        elif dst_type == 'BF16':
            dst_ctype = 'uint16_t'

        # if base implementation is used, increase vector size to block size to prevent wrong signs in case of inf
        mx = 1 if gemm_fp == ("mxgemm_fp8_opt_ex") else 0
        vector_size = block_size if mx == 0 else 16 if src_type == 'FP4' else 8

        # if base implementation is used, dual ssr must be disabled to avoid issues with scales
        # if optimized mx implementation is used, dual ssr is enabled by default
        dual_ssr = kwargs['dual_ssr'] if 'dual_ssr' in kwargs else mx

        a, b, scales, result_hex, result_value, sa, sb, sb_t = mx_tests.test_gemm_fp9_sop_fp32_with_random_values(seed=42,
                            data_type=src_type,
                            acc_data_type=dst_type,
                            vector_size=vector_size,
                            scale_range=[-63, 63],
                            rows=M,
                            cols=N,
                            inner_dim=K,
                            block_size=block_size,
                            use_external_data=False)

        # b is transposed in the golden model

        a_uid = 'a'
        b_uid = 'b'
        c_uid = 'c'

        cfg = {
            'prec': prec,
            **{'prec_dst': 4 for k, v in kwargs.items() if k == 'dst_type' and v == 'FP32'},
            **{'prec_dst': 2 for k, v in kwargs.items() if k == 'dst_type' and v == 'BF16'},
            **{k: 0 for k, v in kwargs.items() if k == 'src_type' and v == 'FP8'},
            **{k: 1 for k, v in kwargs.items() if k == 'src_type' and v == 'FP8ALT'},
            **{k: 2 for k, v in kwargs.items() if k == 'src_type' and v == 'FP6'},
            **{k: 3 for k, v in kwargs.items() if k == 'src_type' and v == 'FP6ALT'},
            **{k: 4 for k, v in kwargs.items() if k == 'src_type' and v == 'FP4'},
            **{k: 5 for k, v in kwargs.items() if k == 'src_type' and v == 'INT8'},
            **{k: v for k, v in kwargs.items() if k != 'src_type' and k != 'dst_type'},
            'a': a_uid,
            'b': b_uid,
            'c': c_uid,
            **(
                {'scales_a': 'scales_a', 'scales_b': 'scales_b', 'dual_ssr': mx} if (mx and dual_ssr) or not mx else
                {'scales': 'scales', 'dual_ssr': 0}
            ),
            'mx': mx
        }

        # Convert binary strings to integers, take 8 MSBs only
        if src_type == 'FP8':
            a_int_list = [int(x[:8], 2) for x in a]
            b_int_list = [int(x[:8], 2) for x in b]
        elif src_type == 'FP8ALT':
            # Take MSB and 7 LSBs
            a_int_list = [int(x[0] + x[2:9], 2) for x in a]
            b_int_list = [int(x[0] + x[2:9], 2) for x in b]
        elif src_type in ['FP6', 'FP6ALT']:
            extract_6bit = (
                (lambda x: x[0] + x[3:8]) if src_type == 'FP6' else
                (lambda x: x[0] + x[4:9])  # FP6ALT
            )
            a_list = [extract_6bit(x) for x in a]
            b_list = [extract_6bit(x) for x in b]
            if unmerged_fp6 == True:
                # Convert 6-bit chunks to integers directly to process as 8-bit values
                a_int_list = [int(x, 2) for x in a_list]
                b_int_list = [int(x, 2) for x in b_list]
            else:
                # Merge 6-bit chunks into 64-bit words for blocks > 192b
                a_64b_list = sixbit_chunks_to_64bit_words(a_list)
                b_64b_list = sixbit_chunks_to_64bit_words(b_list)

                a_int_list = unpack_64b_chunks(a_64b_list)
                b_int_list = unpack_64b_chunks(b_64b_list)
        elif src_type == 'FP4':
            # e2m1, pack two FP4 numbers in one byte
            a_int_list = []
            b_int_list = []
            for i in range(0, len(a), 2):
                a_int_list.append(int(a[i][0] + a[i][4:7] + a[i + 1][0] + a[i + 1][4:7], 2))
            for i in range(0, len(b), 2):
                b_int_list.append(int(b[i][0] + b[i][4:7] + b[i + 1][0] + b[i + 1][4:7], 2))
        elif src_type == 'INT8':
            a_int_list = [int(x, 2) for x in a]
            b_int_list = [int(x, 2) for x in b]

        # print(int_list)
        # Generate C array content
        arrays = f"char {a_uid}[{len(a)}] = {{\n    "
        arrays += ", ".join(f"0x{num:02X}" for num in a_int_list)
        arrays += "\n};\n\n"

        arrays += f"char {b_uid}[{len(b)}] = {{\n    "
        arrays += ", ".join(f"0x{num:02X}" for num in b_int_list)
        arrays += "\n};\n\n"

        if (mx and dual_ssr) or not mx:
            int_list = [int(s, 2) for s in sa]
            arrays += f"uint8_t scales_a[{len(sa)}] = {{\n    "
            arrays += ", ".join(f"0x{num:02X}" for num in int_list)
            arrays += "\n};\n\n"

            if mx == 1:
                # transpose scales_b for mx implementation
                int_list = [int(s, 2) for s in sb_t]
            else:
                int_list = [int(s, 2) for s in sb]
            arrays += f"uint8_t scales_b[{len(sb)}] = {{\n    "
            arrays += ", ".join(f"0x{num:02X}" for num in int_list)
            arrays += "\n};\n\n"
        else:
            int_list = [int(s, 2) for s in scales]
            arrays += f"uint8_t scales[{len(scales)}] = {{\n    "
            arrays += ", ".join(f"0x{num:02X}" for num in int_list)
            arrays += "\n};\n\n"

        arrays += f"{dst_ctype} {c_uid}[{len(result_hex)}] = {{\n    "
        arrays += ", ".join(f"0" for num in result_hex)
        arrays += "\n};\n\n"

        arrays += f"{dst_ctype} result[{len(result_hex)}] = {{\n    "
        if dst_type == "FP32":
            arrays += ", ".join(
                "INFINITY" if num == float('inf') else
                "-INFINITY" if num == float('-inf') else
                "NAN" if num != num else
                f"{num}" for num in result_value
            )
        elif dst_type == "BF16":
            arrays += ", ".join(f"{num}" for num in result_hex)
        arrays += "\n};\n\n"

        # Output the result
        # print(arrays)

        if dual_ssr:
            header += ['#define DUAL_SSR\n']

        if src_type == 'FP8ALT':
            header += ['#define FP8ALT\n']
        elif src_type == 'FP6':
            header += ['#define FP6\n']
        elif src_type == 'FP6ALT':
            header += ['#define FP6ALT\n']
        elif src_type == 'FP4':
            header += ['#define FP4\n']
        elif src_type == 'INT8':
            header += ['#define INT8\n']

        if dst_type == 'BF16':
            header += ['#define BF16\n']

        # if baseline and FP8ALT needed, fmode must be set to 1
        if mx != 1 and src_type == 'FP8ALT':
            header += ['#define BASE_ALT\n']

        # header += ['#define BIST\n']

        header += [arrays]
        header += [du.format_struct_definition('mxgemm_args_t', 'args', cfg)]
        header = '\n\n'.join(header)

        return header


if __name__ == "__main__":
    sys.exit(MXGemmDataGen().main())
