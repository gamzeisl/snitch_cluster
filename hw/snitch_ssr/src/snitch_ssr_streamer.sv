// Copyright 2020 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Fabian Schuiki <fschuiki@iis.ee.ethz.ch>
// Author: Paul Scheffler <paulsc@iis.ee.ethz.ch>

`include "common_cells/assertions.svh"
`include "common_cells/registers.svh"
`include "snitch_ssr/typedef.svh"

module snitch_ssr_streamer import snitch_ssr_pkg::*; #(
  parameter bit          XFMXDOTP   = 0,
  parameter int unsigned NumSsrs    = 0,
  parameter int unsigned NumMemSsrs = 0, // Number of memory ports for SSRs (different from NumSsrs only for MXDOTP)
  parameter int unsigned RPorts     = 0,
  parameter int unsigned WPorts     = 0,
  parameter int unsigned AddrWidth  = 0,
  parameter int unsigned DataWidth  = 0,
  parameter ssr_cfg_t [NumSsrs-1:0]  SsrCfgs = '0,
  parameter logic [NumSsrs-1:0][4:0] SsrRegs = '0,
  parameter type tcdm_user_t  = logic,
  parameter type tcdm_req_t   = logic,
  parameter type tcdm_rsp_t   = logic,
  /// Derived parameter *Do not override*
  parameter type addr_t = logic [AddrWidth-1:0],
  parameter type data_t = logic [DataWidth-1:0]
) (
  input  logic             clk_i,
  input  logic             rst_ni,
  // Access to configuration registers (REG_BUS).
  input  logic [11:0]      cfg_word_i,
  input  logic             cfg_write_i, // 0 = read, 1 = write
  output logic [31:0]      cfg_rdata_o,
  input  logic [31:0]      cfg_wdata_i,
  output logic             cfg_wready_o,
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
  output tcdm_req_t [NumMemSsrs-1:0] mem_req_o,
  input  tcdm_rsp_t [NumMemSsrs-1:0] mem_rsp_i,
  // From intersector to stream controller
  output logic             streamctl_done_o,
  output logic             streamctl_valid_o,
  input  logic             streamctl_ready_i,
  // MXDOTP dual SSR mode
  input  logic             dual_ssr_en_i
);

  // Derive intersection-related configuration from SSR configurations.
  // This will *not* validate the configuration (see assertions below).
  function automatic isect_cfg_t derive_isect_cfg();
    // Ensure nonzero width parameters to keep derived types sane.
    automatic isect_cfg_t ret = '{IndexWidth: 1, default: '0};
    for (int i = 0; i < NumSsrs; i++) begin
      if (SsrCfgs[i].IsectMaster) begin
        automatic int unsigned DataBufDepth =
            SsrCfgs[i].DataCredits + 2*unsigned'(SsrCfgs[i].IndirOutSpill);
        if (DataBufDepth > ret.StreamctlDepth)
          ret.StreamctlDepth = DataBufDepth;
        if (SsrCfgs[i].IndexWidth > ret.IndexWidth)
          ret.IndexWidth = SsrCfgs[i].IndexWidth;
        if (SsrCfgs[i].IsectMasterIdx) begin
          ret.NumMaster1++;
          ret.IdxMaster1 = i;
        end else begin
          ret.NumMaster0++;
          ret.IdxMaster0 = i;
        end
      end if (SsrCfgs[i].IsectSlave) begin
        ret.NumSlave++;
        ret.IdxSlave = i;
      end
    end
    return ret;
  endfunction

  // Intersection configuration
  localparam isect_cfg_t IsectCfg = derive_isect_cfg();

  // Intersection configuration assertions
  `ASSERT_INIT(isect_max_one_master0, IsectCfg.NumMaster0 <= 1)
  `ASSERT_INIT(isect_max_one_slave, IsectCfg.NumSlave <= 1)
  `ASSERT_INIT(isect_num_masters_equal, IsectCfg.NumMaster0 == IsectCfg.NumMaster1)
  `ASSERT_INIT(isect_slave_only_if_master, IsectCfg.NumSlave <= IsectCfg.NumMaster0)

  // Intersection types
  `SSR_ISECT_TYPEDEF_ALL(isect, logic [IsectCfg.IndexWidth-1:0])

  // Intersector IO
  isect_mst_req_t [NumSsrs-1:0] isect_mst_req;
  isect_slv_req_t [NumSsrs-1:0] isect_slv_req;
  isect_mst_rsp_t [1:0]         isect_mst_rsp;
  isect_slv_rsp_t               isect_slv_rsp;

  data_t [NumSsrs-1:0] lane_rdata;
  data_t [NumSsrs-1:0] lane_wdata;
  logic  [NumSsrs-1:0] lane_write;
  logic  [NumSsrs-1:0] lane_valid;
  logic  [NumSsrs-1:0] lane_ready;

  logic [4:0]               dmcfg_word;
  logic [4:0]               dmcfg_upper_addr;
  logic [NumSsrs-1:0][31:0] dmcfg_rdata;
  logic [NumSsrs-1:0]       dmcfg_strobe; // which data mover is currently addressed
  logic [NumSsrs-1:0]       dmcfg_wready;
  snitch_ssr_switch #(
    .XFMXDOTP  ( XFMXDOTP   ),
    .DataWidth ( DataWidth  ),
    .NumSsrs   ( NumSsrs    ),
    .RPorts    ( RPorts     ),
    .WPorts    ( WPorts     ),
    .SsrRegs   ( SsrRegs    )
  ) i_switch (
    .clk_i,
    .rst_ni,
    .ssr_raddr_i,
    .ssr_rdata_o,
    .ssr_rvalid_i,
    .ssr_rready_o,
    .ssr_rdone_i,
    .ssr_waddr_i,
    .ssr_wdata_i,
    .ssr_wvalid_i,
    .ssr_wready_o,
    .ssr_wdone_i,
    .lane_rdata_i ( lane_rdata ),
    .lane_wdata_o ( lane_wdata ),
    .lane_write_o ( lane_write ),
    .lane_valid_i ( lane_valid ),
    .lane_ready_o ( lane_ready ),
    .dual_ssr_en_i
  );

  tcdm_req_t [NumSsrs-1:0] mem_req;
  tcdm_rsp_t [NumSsrs-1:0] mem_rsp;
  for (genvar i = 0; i < NumSsrs; i++) begin : gen_ssrs
    snitch_ssr #(
      .Cfg          ( SsrCfgs [i] ),
      .AddrWidth    ( AddrWidth   ),
      .DataWidth    ( DataWidth   ),
      .tcdm_user_t  ( tcdm_user_t ),
      .tcdm_req_t   ( tcdm_req_t  ),
      .tcdm_rsp_t   ( tcdm_rsp_t  ),
      .isect_slv_req_t  ( isect_slv_req_t ),
      .isect_slv_rsp_t  ( isect_slv_rsp_t ),
      .isect_mst_req_t  ( isect_mst_req_t ),
      .isect_mst_rsp_t  ( isect_mst_rsp_t )
    ) i_ssr (
      .clk_i,
      .rst_ni,
      .cfg_wdata_i,
      .cfg_word_i     ( dmcfg_word        ),
      .cfg_write_i    ( cfg_write_i & dmcfg_strobe[i] ),
      .cfg_rdata_o    ( dmcfg_rdata  [i]  ),
      .cfg_wready_o   ( dmcfg_wready [i]  ),
      .lane_rdata_o   ( lane_rdata   [i]  ),
      .lane_wdata_i   ( lane_wdata   [i]  ),
      .lane_valid_o   ( lane_valid   [i]  ),
      .lane_ready_i   ( lane_ready   [i]  ),
      .mem_req_o      ( mem_req      [i]  ),
      .mem_rsp_i      ( mem_rsp      [i]  ),
      .isect_mst_req_o  ( isect_mst_req [i] ),
      .isect_slv_req_o  ( isect_slv_req [i] ),
      .isect_mst_rsp_i  ( isect_mst_rsp [SsrCfgs[i].IsectMasterIdx] ),
      .isect_slv_rsp_i  ( isect_slv_rsp     )
    );
  end

  if (IsectCfg.NumMaster0 != 0) begin : gen_intersector
    snitch_ssr_intersector #(
      .StreamctlDepth  ( IsectCfg.StreamctlDepth ),
      .isect_slv_req_t ( isect_slv_req_t ),
      .isect_slv_rsp_t ( isect_slv_rsp_t ),
      .isect_mst_req_t ( isect_mst_req_t ),
      .isect_mst_rsp_t ( isect_mst_rsp_t )
    ) i_snitch_ssr_intersector (
      .clk_i,
      .rst_ni,
      .mst_req_i ( {isect_mst_req[IsectCfg.IdxMaster1], isect_mst_req[IsectCfg.IdxMaster0]} ),
      .slv_req_i ( isect_slv_req[IsectCfg.IdxSlave] ),
      .mst_rsp_o ( isect_mst_rsp ),
      .slv_rsp_o ( isect_slv_rsp ),
      .streamctl_done_o,
      .streamctl_valid_o,
      .streamctl_ready_i
    );
  end else begin : gen_no_intersector
    assign isect_mst_rsp      = '0;
    assign isect_slv_rsp      = '0;
    assign streamctl_done_o   = '0;
    assign streamctl_valid_o  = '0;
  end

  // Determine which data movers are addressed via the config interface. We
  // use the upper address bits to select one of the data movers, or select
  // all if the bits are all 1.
  always_comb begin
    dmcfg_word = cfg_word_i[4:0];
    dmcfg_upper_addr = cfg_word_i[11:7];
    dmcfg_strobe = (dmcfg_upper_addr == '1 ? '1 : (1 << dmcfg_upper_addr));
    cfg_rdata_o = dmcfg_rdata[dmcfg_upper_addr];
  end

  // cfg_wready_o indicates whether the SSR(s) can service a write without
  // overriding a shadowed job; writes will not take effect until high.
  always_comb begin
    cfg_wready_o = 1'b1;
    if (dmcfg_upper_addr < NumSsrs) begin
      cfg_wready_o = dmcfg_wready[dmcfg_upper_addr];
    end else if (dmcfg_upper_addr == '1) begin
      cfg_wready_o = &dmcfg_wready;
    end
  end


  if (XFMXDOTP && NumSsrs == 4) begin : gen_dual_ssr_mux
    // Internal indices for port 2 and 3 (dual SSR handling)
    localparam int SSR2_IDX = 0;
    localparam int SSR3_IDX = 1;

    // Internal signals
    tcdm_rsp_t [1:0] mem_rsp_int;
    logic      [1:0][2:0] stride_q, stride_d;

    // Registers
    `FF(stride_q, stride_d, '0, clk_i, rst_ni)

    // TCDM MUX (for ports 2 and 3)
    tcdm_mux #(
      .NrPorts    ( 2             ),
      .AddrWidth  ( AddrWidth     ),
      .DataWidth  ( DataWidth     ),
      .user_t     ( tcdm_user_t   ),
      .RespDepth  ( 4             ),
      .tcdm_req_t ( tcdm_req_t    ),
      .tcdm_rsp_t ( tcdm_rsp_t    )
    ) i_tcdm_mux (
      .clk_i,
      .rst_ni,
      .slv_req_i  ( {mem_req[2], mem_req[3]} ),
      .slv_rsp_o  ( {mem_rsp_int[SSR2_IDX], mem_rsp_int[SSR3_IDX]} ),
      .mst_req_o  ( mem_req_o[2] ),
      .mst_rsp_i  ( mem_rsp_i[2] )
    );

    // Since scales are 8-bit, we keep track of the 3 LSBs of the address
    // to determine the byte offset within a 64-bit word.
    always_comb begin
      stride_d = stride_q;
      mem_rsp[2]  = mem_rsp_int[SSR2_IDX];
      mem_rsp[3]  = mem_rsp_int[SSR3_IDX];

      if (dual_ssr_en_i) begin
        // Shift the data according to the stride of the request
        // Always one cycle latency after the request
        mem_rsp[2].p.data = mem_rsp_int[SSR2_IDX].p.data >> (stride_q[SSR2_IDX] * 8);
        mem_rsp[3].p.data = mem_rsp_int[SSR3_IDX].p.data >> (stride_q[SSR3_IDX] * 8);

        // Update stride values on valid transaction
        if (mem_req[2].q_valid && mem_rsp_int[SSR2_IDX].q_ready)
          stride_d[SSR2_IDX] = mem_req[2].q.addr[2:0];

        if (mem_req[3].q_valid && mem_rsp_int[SSR3_IDX].q_ready)
          stride_d[SSR3_IDX] = mem_req[3].q.addr[2:0];
      end
    end

    // These ports are not affected by dual SSR mode
    assign mem_req_o[0] = mem_req[0];
    assign mem_req_o[1] = mem_req[1];
    assign mem_rsp[0]   = mem_rsp_i[0];
    assign mem_rsp[1]   = mem_rsp_i[1];
  end else begin : gen_no_dual_ssr_mux
    assign mem_req_o = mem_req;
    assign mem_rsp   = mem_rsp_i;
  end
endmodule
