`ifndef NNUE_AXI_LITE_WRAPPER_SV
`define NNUE_AXI_LITE_WRAPPER_SV

import nnue_pkg::*;

module nnue_axi_lite_wrapper #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 6,
    parameter BIASES_ACC_MEM_FILE        = "mem/biases_acc.mem",
    parameter WEIGHTS_ACC_MEM_FILE       = "mem/weights_acc.mem",
    parameter BIASES_FC1_MEM_FILE        = "mem/biases_fc1.mem",
    parameter WEIGHTS_FC1_MEM_FILE       = "mem/weights_fc1.mem",
    parameter BIASES_FC2_MEM_FILE        = "mem/biases_fc2.mem",
    parameter WEIGHTS_FC2_MEM_FILE       = "mem/weights_fc2.mem"
)(
    // AXI4-Lite Clock & Active-Low Reset
    input  logic                                  s_axi_aclk,
    input  logic                                  s_axi_aresetn,

    // AXI4-Lite Write Address Channel
    input  logic [C_S_AXI_ADDR_WIDTH-1:0]         s_axi_awaddr,
    input  logic [2:0]                            s_axi_awprot,
    input  logic                                  s_axi_awvalid,
    output logic                                  s_axi_awready,

    // AXI4-Lite Write Data Channel
    input  logic [C_S_AXI_DATA_WIDTH-1:0]         s_axi_wdata,
    input  logic [(C_S_AXI_DATA_WIDTH/8)-1:0]     s_axi_wstrb,
    input  logic                                  s_axi_wvalid,
    output logic                                  s_axi_wready,

    // AXI4-Lite Write Response Channel
    output logic [1:0]                            s_axi_bresp,
    output logic                                  s_axi_bvalid,
    input  logic                                  s_axi_bready,

    // AXI4-Lite Read Address Channel
    input  logic [C_S_AXI_ADDR_WIDTH-1:0]         s_axi_araddr,
    input  logic [2:0]                            s_axi_arprot,
    input  logic                                  s_axi_arvalid,
    output logic                                  s_axi_arready,

    // AXI4-Lite Read Data Channel
    output logic [C_S_AXI_DATA_WIDTH-1:0]         s_axi_rdata,
    output logic [1:0]                            s_axi_rresp,
    output logic                                  s_axi_rvalid,
    input  logic                                  s_axi_rready,

    // Optional Interrupt Output
    output logic                                  irq
);

    // Register Map Offsets
    // 0x00 (ADDR 0): Control Register   [bit 0: start_strobe, bit 1: irq_en]
    // 0x04 (ADDR 1): Status Register    [bit 0: ready, bit 1: done, bit 2: piece_ready]
    // 0x08 (ADDR 2): Config Register    [bit 5:0: num_pieces]
    // 0x0C (ADDR 3): Score Result       [bit 15:0: signed score cp]
    // 0x10 (ADDR 4): Piece Write FIFO   [bit 9:0: turn_feature, bit 25:16: opp_feature] (Writing triggers piece_valid)
    // 0x14 (ADDR 5): Benchmark Cycle Counter [bit 31:0: latency in clock cycles]

    // AXI-Lite Internal Registers
    logic [C_S_AXI_DATA_WIDTH-1:0] slv_reg_ctrl;
    logic [C_S_AXI_DATA_WIDTH-1:0] slv_reg_num_pieces;
    logic [C_S_AXI_DATA_WIDTH-1:0] slv_reg_cycle_cnt;
    logic                          cycle_counter_running;

    // DUT Handshake Signals
    logic                          dut_start;
    logic                          dut_ready;
    logic                          dut_done;
    logic signed [SCORE_WIDTH-1:0] dut_score;
    logic [5:0]                    dut_num_pieces;
    logic                          dut_piece_valid;
    logic                          dut_piece_ready;
    logic [FEATURE_ADDR_BITS-1:0]  dut_piece_turn;
    logic [FEATURE_ADDR_BITS-1:0]  dut_piece_opp;

    // AXI-Lite State Signals
    logic [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr;
    logic                          axi_awready;
    logic                          axi_wready;
    logic [1:0]                    axi_bresp;
    logic                          axi_bvalid;
    logic [C_S_AXI_ADDR_WIDTH-1:0] axi_araddr;
    logic                          axi_arready;
    logic [C_S_AXI_DATA_WIDTH-1:0] axi_rdata;
    logic [1:0]                    axi_rresp;
    logic                          axi_rvalid;

    assign s_axi_awready = axi_awready;
    assign s_axi_wready  = axi_wready;
    assign s_axi_bresp   = axi_bresp;
    assign s_axi_bvalid  = axi_bvalid;
    assign s_axi_arready = axi_arready;
    assign s_axi_rdata   = axi_rdata;
    assign s_axi_rresp   = axi_rresp;
    assign s_axi_rvalid  = axi_rvalid;

    // AXI-Lite Write Handshake
    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            axi_awready        <= 1'b0;
            axi_wready         <= 1'b0;
            axi_bvalid         <= 1'b0;
            axi_bresp          <= 2'b00;
            axi_awaddr         <= '0;
            slv_reg_ctrl       <= '0;
            slv_reg_num_pieces <= '0;
            dut_start          <= 1'b0;
            dut_piece_valid    <= 1'b0;
            dut_piece_turn     <= '0;
            dut_piece_opp      <= '0;
        end else begin
            // Default single-cycle strobes
            dut_start       <= 1'b0;
            dut_piece_valid <= 1'b0;

            // AWREADY generation
            if (!axi_awready && s_axi_awvalid && s_axi_wvalid) begin
                axi_awready <= 1'b1;
                axi_awaddr  <= s_axi_awaddr;
            end else begin
                axi_awready <= 1'b0;
            end

            // WREADY generation
            if (!axi_wready && s_axi_wvalid && s_axi_awvalid) begin
                axi_wready <= 1'b1;
            end else begin
                axi_wready <= 1'b0;
            end

            // Register write handling
            if (axi_awready && s_axi_awvalid && axi_wready && s_axi_wvalid) begin
                case (axi_awaddr[5:2])
                    4'h0: begin // 0x00: Control Register
                        slv_reg_ctrl <= s_axi_wdata;
                        if (s_axi_wdata[0]) dut_start <= 1'b1;
                    end
                    4'h2: begin // 0x08: Config Register (num_pieces)
                        slv_reg_num_pieces <= s_axi_wdata;
                    end
                    4'h4: begin // 0x10: Piece Write Port
                        dut_piece_turn  <= s_axi_wdata[FEATURE_ADDR_BITS-1:0];
                        dut_piece_opp   <= s_axi_wdata[16 +: FEATURE_ADDR_BITS];
                        dut_piece_valid <= 1'b1;
                    end
                    default: ;
                endcase
            end

            // BVALID response
            if (axi_awready && s_axi_awvalid && axi_wready && s_axi_wvalid && !axi_bvalid) begin
                axi_bvalid <= 1'b1;
                axi_bresp  <= 2'b00; // OKAY
            end else if (s_axi_bready && axi_bvalid) begin
                axi_bvalid <= 1'b0;
            end
        end
    end

    // AXI-Lite Read Handshake
    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            axi_arready <= 1'b0;
            axi_rvalid  <= 1'b0;
            axi_rresp   <= 2'b00;
            axi_rdata   <= '0;
            axi_araddr  <= '0;
        end else begin
            if (!axi_arready && s_axi_arvalid) begin
                axi_arready <= 1'b1;
                axi_araddr  <= s_axi_araddr;
            end else begin
                axi_arready <= 1'b0;
            end

            if (axi_arready && s_axi_arvalid && !axi_rvalid) begin
                axi_rvalid <= 1'b1;
                axi_rresp  <= 2'b00; // OKAY

                case (axi_araddr[5:2])
                    4'h0: axi_rdata <= slv_reg_ctrl;
                    4'h1: axi_rdata <= {29'd0, dut_piece_ready, dut_done, dut_ready};
                    4'h2: axi_rdata <= slv_reg_num_pieces;
                    4'h3: axi_rdata <= {{16{dut_score[15]}}, dut_score};
                    4'h4: axi_rdata <= {5'd0, dut_piece_opp, 6'd0, dut_piece_turn};
                    4'h5: axi_rdata <= slv_reg_cycle_cnt;
                    default: axi_rdata <= 32'hDEADBEEF;
                endcase
            end else if (axi_rvalid && s_axi_rready) begin
                axi_rvalid <= 1'b0;
            end
        end
    end

    // Benchmark Cycle Counter
    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            slv_reg_cycle_cnt     <= '0;
            cycle_counter_running <= 1'b0;
        end else begin
            if (dut_start) begin
                slv_reg_cycle_cnt     <= '0;
                cycle_counter_running <= 1'b1;
            end else if (cycle_counter_running) begin
                slv_reg_cycle_cnt <= slv_reg_cycle_cnt + 1'b1;
                if (dut_done) begin
                    cycle_counter_running <= 1'b0;
                end
            end
        end
    end

    assign dut_num_pieces = slv_reg_num_pieces[5:0];
    assign irq            = slv_reg_ctrl[1] & dut_done;

    // Instantiate Top NNUE Core
    nnue_top #(
        .BIASES_ACC_MEM_FILE(BIASES_ACC_MEM_FILE),
        .WEIGHTS_ACC_MEM_FILE(WEIGHTS_ACC_MEM_FILE),
        .BIASES_FC1_MEM_FILE(BIASES_FC1_MEM_FILE),
        .WEIGHTS_FC1_MEM_FILE(WEIGHTS_FC1_MEM_FILE),
        .BIASES_FC2_MEM_FILE(BIASES_FC2_MEM_FILE),
        .WEIGHTS_FC2_MEM_FILE(WEIGHTS_FC2_MEM_FILE)
    ) u_nnue_core (
        .clk(s_axi_aclk),
        .rst_n(s_axi_aresetn),
        .start(dut_start),
        .ready(dut_ready),
        .done(dut_done),
        .score(dut_score),
        .num_pieces(dut_num_pieces),
        .piece_valid(dut_piece_valid),
        .piece_ready(dut_piece_ready),
        .piece_turn(dut_piece_turn),
        .piece_opp(dut_piece_opp)
    );

endmodule : nnue_axi_lite_wrapper

`endif
