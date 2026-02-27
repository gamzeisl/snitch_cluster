#!/usr/bin/env python3
# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

import numpy as np
import sys
from datagen import MXGemmDataGen

from snitch.util.sim.verif_utils import Verifier
from snitch.util.sim.data_utils import ctype_from_precision_t

import os

sys.path.append(os.environ['MX_GOLDEN_PATH'])
import golden.fp_accumulator as mx_fp

class MXGemmVerifier(Verifier):

    OUTPUT_UIDS = ['c']
    ERR_THRESHOLD = {
        1: 1e-4,
        2: 5e-1,
        4: 1e-3,
        8: 1e-3
    }

    def __init__(self):
        super().__init__()
        self.func_args = {
            'alpha': 'd',
            'mx': 'I',
            'prec': 'I',
            'prec_dst': 'I',
            'setup_ssr': 'I',
            'parallelize_m': 'I',
            'parallelize_k': 'I',
            'm_tiles': 'I',
            'n_tiles': 'I',
            'k_tiles': 'I',
            'load_a': 'I',
            'load_b': 'I',
            'load_c': 'I',
            'transa': 'I',
            'transb': 'I',
            'M': 'I',
            'N': 'I',
            'K': 'I',
            'block_size': 'I',
            'src_type': 'I',
            'a': 'I',
            'b': 'I',
            'beta': 'I',
            'c': 'I',
            'scales': 'I',
            'scales_a': 'I',
            'scales_b': 'I',
            'd1': 'I',
            'd2': 'I',
        }
        self.func_args = self.get_input_from_symbol('args', self.func_args)

    def get_actual_results(self):
        prec_dst = self.func_args['prec_dst']
        if prec_dst == 2:
            # bf16
            fp_16 = self.get_output_from_symbol(self.OUTPUT_UIDS[0], 'uint16_t') # np.float16 is returned
            fp_16_binary = [np.binary_repr(fp, width=16) for fp in fp_16]
            result = [mx_fp.FPAccumulator.binary_to_value(fp, data_type="BF16") for fp in fp_16_binary]
        elif prec_dst == 4:
            # fp32
            result = self.get_output_from_symbol(self.OUTPUT_UIDS[0], ctype_from_precision_t(prec_dst))
        return result

    def get_expected_results(self):
        prec = self.func_args['prec']
        prec_dst = self.func_args['prec_dst']
        a = self.get_input_from_symbol('a', ctype_from_precision_t(prec))
        b = self.get_input_from_symbol('b', ctype_from_precision_t(prec))
        c = self.get_input_from_symbol('c', ctype_from_precision_t(prec_dst))
        beta = self.func_args['beta']
        m = self.func_args['M']
        n = self.func_args['N']
        k = self.func_args['K']
        block_size = self.func_args['block_size']
        tb = self.func_args['transb']
        src_type = self.func_args['src_type']
        mx = self.func_args['mx']

        a = np.reshape(a, (m, k))
        if tb:
            b = np.reshape(b, (n, k))
            b = b.transpose()
        else:
            b = np.reshape(b, (k, n))
        c = np.reshape(c, (m, n))

        return MXGemmDataGen().exact_golden_model(1, a, b, beta, c, block_size, src_type, prec_dst, mx).flatten()

    def check_results(self, *args):
        prec_dst = self.func_args['prec_dst']
        return super().check_results(*args, rtol=self.ERR_THRESHOLD[prec_dst])


if __name__ == "__main__":
    sys.exit(MXGemmVerifier().main())
