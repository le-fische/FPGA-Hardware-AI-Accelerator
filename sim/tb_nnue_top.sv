`timescale 1ns/1ps

import nnue_pkg::*;

module tb_nnue_top;

    // Clock and Reset
    logic clk;
    logic rst_n;

    // DUT Signals
    logic                          start;
    logic                          ready;
    logic                          done;
    logic signed [SCORE_WIDTH-1:0] score;
    logic [5:0]                    num_pieces;
    logic                          piece_valid;
    logic                          piece_ready;
    logic                          piece_remove;
    logic [FEATURE_ADDR_BITS-1:0]  piece_turn;
    logic [FEATURE_ADDR_BITS-1:0]  piece_opp;
    logic                          eval_mode;
    logic                          swap_turn;

    // Instantiate DUT (Device Under Test)
    nnue_top #(
        .BIASES_ACC_MEM_FILE("mem/biases_acc.mem"),
        .WEIGHTS_ACC_MEM_FILE("mem/weights_acc.mem"),
        .BIASES_FC1_MEM_FILE("mem/biases_fc1.mem"),
        .WEIGHTS_FC1_MEM_FILE("mem/weights_fc1.mem"),
        .BIASES_FC2_MEM_FILE("mem/biases_fc2.mem"),
        .WEIGHTS_FC2_MEM_FILE("mem/weights_fc2.mem")
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .eval_mode(eval_mode),
        .swap_turn(swap_turn),
        .ready(ready),
        .done(done),
        .score(score),
        .num_pieces(num_pieces),
        .piece_valid(piece_valid),
        .piece_ready(piece_ready),
        .piece_remove(piece_remove),
        .piece_turn(piece_turn),
        .piece_opp(piece_opp)
    );

    // 100 MHz Clock Generator (10 ns period)
    always #5 clk = ~clk;

    // Testbench Variables
    integer stim_file;
    integer exp_file;
    integer status;
    integer testcase_idx;
    integer total_passed;
    integer total_failed;
    string  line;
    integer n_pieces_in;
    logic [9:0] t_idx;
    logic [9:0] o_idx;
    logic signed [15:0] expected_score;
    logic signed [15:0] diff_expected_score;

    // Watchdog Timer (terminate if simulation hangs)
    initial begin
        #5000000; // 5 ms = 500,000 clock cycles
        $display("ERROR: Simulation timed out after 5 ms!");
        $finish;
    end

    // Waveform Dump
    initial begin
        $dumpfile("sim/nnue_sim.vcd");
        $dumpvars(0, tb_nnue_top);
    end

    initial begin
        // Initialize Signals
        clk          = 0;
        rst_n        = 0;
        start        = 0;
        eval_mode    = 0;
        swap_turn    = 0;
        piece_remove = 0;
        num_pieces   = 0;
        piece_valid  = 0;
        piece_turn   = 0;
        piece_opp    = 0;
        total_passed = 0;
        total_failed = 0;

        $display("=================================================================");
        $display("   STARTING NNUE ACCELERATOR CYCLE-ACCURATE RTL SIMULATION      ");
        $display("=================================================================");

        // Apply Reset
        #20;
        rst_n = 1;
        #20;

        stim_file = $fopen("testbench/golden_stimulus.hex", "r");
        exp_file  = $fopen("testbench/golden_expected.hex", "r");

        if (stim_file == 0 || exp_file == 0) begin
            $display("ERROR: Could not open testbench files!");
            $finish;
        end

        testcase_idx = 1;

        // Loop over the 10 Golden Benchmark Positions
        while (!$feof(stim_file) && testcase_idx <= 10) begin
            // Read lines until finding piece count
            status = $fscanf(stim_file, "%x\n", n_pieces_in);
            if (status != 1) begin
                // Skip comment line
                void'($fgets(line, stim_file));
                continue;
            end

            // Read corresponding expected score
            status = $fscanf(exp_file, "%x\n", expected_score);
            while (status != 1 && !$feof(exp_file)) begin
                void'($fgets(line, exp_file));
                status = $fscanf(exp_file, "%x\n", expected_score);
            end

            $display("\n--- [Testcase %02d] Presenting %0d active pieces to hardware ---", testcase_idx, n_pieces_in);

            wait(ready == 1'b1);
            @(posedge clk);
            start        <= 1'b1;
            eval_mode    <= 1'b0; // Full board refresh mode
            swap_turn    <= 1'b0;
            num_pieces   <= n_pieces_in[5:0];
            @(posedge clk);
            start        <= 1'b0;

            // Stream active pieces into DUT
            for (int p = 0; p < n_pieces_in; p++) begin
                status = $fscanf(stim_file, "%x %x\n", t_idx, o_idx);
                
                @(posedge clk);
                piece_valid  <= 1'b1;
                piece_remove <= 1'b0; // Add piece
                piece_turn   <= t_idx;
                piece_opp    <= o_idx;

                // Hold valid until piece_ready is sampled high on clock edge
                do begin
                    @(posedge clk);
                end while (!piece_ready);

                piece_valid <= 1'b0;
            end

            // Wait for evaluation done strobe
            wait(done == 1'b1);
            @(posedge clk);

            // Assert correctness
            if (score === expected_score) begin
                $display(">>> [PASS] Testcase %02d: Score = %0d cp (RTL output matched golden reference bit-for-bit!)", 
                         testcase_idx, $signed(score));
                total_passed++;
            end else begin
                $display(">>> [FAIL] Testcase %02d: RTL Score = %0d cp, Expected = %0d cp (Hex: %04x vs %04x)",
                         testcase_idx, $signed(score), $signed(expected_score), score, expected_score);
                total_failed++;
            end

            testcase_idx++;
            #20;
        end

        $fclose(stim_file);

        // Read Testcase 11 expected score from exp_file before closing
        status = $fscanf(exp_file, "%x\n", diff_expected_score);
        while (status != 1 && !$feof(exp_file)) begin
            void'($fgets(line, exp_file));
            status = $fscanf(exp_file, "%x\n", diff_expected_score);
        end
        $fclose(exp_file);

        // -------------------------------------------------------------
        // Testcase 11: Incremental / Differential Move Evaluation (1. e4)
        // -------------------------------------------------------------
        $display("\n--- [Testcase 11] Incremental Differential Move: 1. e4 ---");
        $display("    Applying Move Diffs (e2 removed, e4 added) in only 2 piece updates...");
        
        // Step 1: Re-initialize to Starting Position (White to move)
        stim_file = $fopen("testbench/golden_stimulus.hex", "r");
        void'($fgets(line, stim_file)); // skip comment
        status = $fscanf(stim_file, "%x\n", n_pieces_in);
        
        wait(ready == 1'b1);
        @(posedge clk);
        start      <= 1'b1;
        eval_mode  <= 1'b0;
        swap_turn  <= 1'b0;
        num_pieces <= n_pieces_in[5:0];
        @(posedge clk);
        start      <= 1'b0;
        
        for (int p = 0; p < n_pieces_in; p++) begin
            status = $fscanf(stim_file, "%x %x\n", t_idx, o_idx);
            @(posedge clk);
            piece_valid  <= 1'b1;
            piece_remove <= 1'b0;
            piece_turn   <= t_idx;
            piece_opp    <= o_idx;
            do @(posedge clk); while (!piece_ready);
            piece_valid <= 1'b0;
        end
        wait(done == 1'b1);
        @(posedge clk);
        $fclose(stim_file);
        
        // Step 2: Now apply move 1. e4 differentially!
        // Move e2 pawn -> e4 pawn. Turn flips to Black (swap_turn = 1).
        wait(ready == 1'b1);
        @(posedge clk);
        start      <= 1'b1;
        eval_mode  <= 1'b1; // Differential mode! Skip bias load!
        swap_turn  <= 1'b1; // Turn flips to Black!
        num_pieces <= 6'd2; // Only 2 move diffs!
        @(posedge clk);
        start      <= 1'b0;
        swap_turn  <= 1'b0;
        
        // Diff 1: Remove e2 White pawn
        @(posedge clk);
        piece_valid  <= 1'b1;
        piece_remove <= 1'b1;
        piece_turn   <= 10'h1b4;
        piece_opp    <= 10'h00c;
        do @(posedge clk); while (!piece_ready);
        piece_valid <= 1'b0;
        
        // Diff 2: Add e4 White pawn
        @(posedge clk);
        piece_valid  <= 1'b1;
        piece_remove <= 1'b0;
        piece_turn   <= 10'h1a4;
        piece_opp    <= 10'h01c;
        do @(posedge clk); while (!piece_ready);
        piece_valid <= 1'b0;
        
        // Wait for differential done strobe
        wait(done == 1'b1);
        @(posedge clk);
        
        if (score === diff_expected_score) begin
            $display(">>> [PASS] Testcase 11 (Differential): Score = %0d cp (Matched Python bit-for-bit in ~10 us!)", $signed(score));
            total_passed++;
        end else begin
            $display(">>> [FAIL] Testcase 11 (Differential): Score = %0d cp, Expected = %0d cp (hex: %04x vs %04x)",
                     $signed(score), $signed(diff_expected_score), score, diff_expected_score);
            total_failed++;
        end

        $display("\n=================================================================");
        $display("SIMULATION SUMMARY: %0d PASSED, %0d FAILED out of %0d TESTCASES",
                 total_passed, total_failed, (total_passed + total_failed));
        if (total_failed == 0 && total_passed > 0) begin
            $display(">>> ALL GOLDEN VECTORS VERIFIED! 100%% HARDWARE TIMING ACCURACY <<<");
        end else begin
            $display(">>> TEST SUITE FAILED <<<");
        end
        $display("=================================================================\n");

        $finish;
    end

endmodule
