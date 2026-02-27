// Copyright 2020 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Fabian Schuiki <fschuiki@iis.ee.ethz.ch>
// Author: Florian Zaruba <zarubaf@iis.ee.ethz.ch>

`include "common_cells/registers.svh"

module snitch_ssr_switch #(
  parameter bit          XFMXDOTP   = 0,
  parameter int unsigned DataWidth  = 0,
  parameter int unsigned NumSsrs    = 0,
  parameter int unsigned RPorts     = 0,
  parameter int unsigned WPorts     = 0,
  parameter logic [NumSsrs-1:0][4:0] SsrRegs = '0,
  /// Derived parameter *Do not override*
  parameter int unsigned Ports = RPorts + WPorts,
  parameter type data_t = logic [DataWidth-1:0]
) (
  input  logic                    clk_i,
  input  logic                    rst_ni,
  // Read and write streams coming from the processor.
  input  logic  [RPorts-1:0][4:0] ssr_raddr_i,
  output data_t [RPorts-1:0]      ssr_rdata_o,
  input  logic  [RPorts-1:0]      ssr_rvalid_i,
  output logic  [RPorts-1:0]      ssr_rready_o,
  input  logic  [RPorts-1:0]      ssr_rdone_i,

  input  logic  [WPorts-1:0][4:0] ssr_waddr_i,
  input  data_t [WPorts-1:0]      ssr_wdata_i,
  input  logic  [WPorts-1:0]      ssr_wvalid_i,
  output logic  [WPorts-1:0]      ssr_wready_o,
  input  logic  [WPorts-1:0]      ssr_wdone_i,
  // Ports into memory.
  input  data_t [NumSsrs-1:0]     lane_rdata_i,
  output data_t [NumSsrs-1:0]     lane_wdata_o,
  output logic  [NumSsrs-1:0]     lane_write_o,
  input  logic  [NumSsrs-1:0]     lane_valid_i,
  output logic  [NumSsrs-1:0]     lane_ready_o,
  // MXDOTP dual SSR mode
  input  logic                    dual_ssr_en_i
);

  logic   [Ports-1:0][4:0] ssr_addr;
  data_t  [Ports-1:0]      ssr_rdata;
  data_t  [Ports-1:0]      ssr_wdata;
  logic   [Ports-1:0]      ssr_valid;
  logic   [Ports-1:0]      ssr_ready;
  logic   [Ports-1:0]      ssr_done;
  logic   [Ports-1:0]      ssr_write;

  // Unify the read and write ports into one structure that we can easily
  // switch.
  always_comb begin
    for (int i = 0; i < RPorts; i++) begin
      ssr_addr[i] = ssr_raddr_i[i];
      ssr_rdata_o[i] = ssr_rdata[i];
      ssr_rready_o[i] = ssr_ready[i];
      ssr_wdata[i] = '0;
      ssr_valid[i] = ssr_rvalid_i[i];
      ssr_done[i] = ssr_rdone_i[i];
      ssr_write[i] = 0;
    end
    for (int i = 0; i < WPorts; i++) begin
      ssr_addr[i+RPorts]  = ssr_waddr_i[i];
      ssr_wdata[i+RPorts] = ssr_wdata_i[i];
      ssr_valid[i+RPorts] = ssr_wvalid_i[i];
      ssr_done[i+RPorts]  = ssr_wdone_i[i];
      ssr_write[i+RPorts] = 1;
      ssr_wready_o[i] = ssr_ready[i+RPorts];
    end
  end

  logic [NumSsrs-1:0] lane_ready_int;
  data_t [Ports-1:0] ssr_rdata_int;
  logic  [Ports-1:0] ssr_ready_int;

  always_comb begin
    lane_ready_int = '0;
    lane_wdata_o = '0;
    lane_write_o = '0;
    ssr_rdata_int = '0;
    ssr_ready_int = '0;

    for (int o = 0; o < NumSsrs; o++) begin
      for (int i = 0; i < Ports; i++) begin
        if (ssr_valid[i] && ssr_addr[i] == SsrRegs[o]) begin
          lane_wdata_o[o] = ssr_wdata[i];
          lane_ready_int[o] = ssr_done[i];
          lane_wdata_o[o] = ssr_wdata[i];
          lane_write_o[o] = ssr_write[i];
          ssr_rdata_int[i] = lane_rdata_i[o];
          ssr_ready_int[i] = lane_valid_i[o];
        end
      end
    end
  end

  if (XFMXDOTP && NumSsrs == 4) begin : gen_dual_ssr
    // Internal signals
    logic [1:0] scale_counter_d, scale_counter_q;
    logic scale_toggle_d, scale_toggle_q;
    logic [31:0] scale_a, scale_b;

    // Registers
    `FF(scale_counter_q, scale_counter_d, '0, clk_i, rst_ni)
    `FF(scale_toggle_q, scale_toggle_d, 0, clk_i, rst_ni)

    always_comb begin
      lane_ready_o = lane_ready_int;
      ssr_rdata = ssr_rdata_int;
      ssr_ready = ssr_ready_int;

      scale_counter_d = scale_counter_q;
      scale_toggle_d = scale_toggle_q;

      if (dual_ssr_en_i) begin
        // In dual SSR mode, the first two SSRs are used to read scale_a and scale_b
        // for the MXDOTP instruction from the 3rd and 4th SSRs.
        scale_a = {24'b0, lane_rdata_i[2][7:0]};
        if (scale_toggle_q == 0) begin
          scale_b = lane_rdata_i[3][31:0];
        end else begin
          scale_b = lane_rdata_i[3][63:32];
        end

        if (ssr_valid[2] && ssr_addr[2] == 2) begin // read
          lane_ready_o[2] = ssr_done[2];
          lane_ready_o[3] = ssr_done[2];
          ssr_rdata[2] = {scale_b, scale_a};
          ssr_ready[2] = lane_valid_i[2] && lane_valid_i[3];
          if (ssr_done[2]) begin
            scale_counter_d = scale_counter_q + 1;
            if (scale_counter_q[1:0] == 3) begin
              scale_toggle_d = ~scale_toggle_q;
            end
          end
        end
      end
    end
  end else begin: gen_no_dual_ssr
    assign lane_ready_o = lane_ready_int;
    assign ssr_rdata = ssr_rdata_int;
    assign ssr_ready = ssr_ready_int;
  end
endmodule
