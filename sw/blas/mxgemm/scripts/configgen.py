#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json5
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cfg", required=True, type=Path, help="Path to params.json")
parser.add_argument("-o", "--out", required=True, type=Path, help="Output path for config.h")

args = parser.parse_args()

with open(args.cfg) as f:
    data = json5.loads(f.read())

dst_type = data["dst_type"]

lines = ["// Auto-generated config.h\n"]

if dst_type == "FP32":
    lines.append("#define FP32\n")
elif dst_type == "BF16":
    lines.append("#define BF16\n")

with open(args.out, "w") as f:
    f.writelines(lines)