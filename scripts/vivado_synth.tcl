# ==============================================================================
# Vivado Non-Project Batch Synthesis Script
# Target FPGA: AMD Xilinx Zynq-7000 (XC7Z020-1CLG400C) / PYNQ-Z2
# Target Frequency: 100 MHz (10.000 ns clock period)
# ==============================================================================

set PROJECT_NAME "nnue_hardware_accelerator"
set PART_NAME    "xc7z020clg400-1"
set TOP_MODULE   "nnue_top"

puts "=================================================================="
puts "  STARTING VIVADO SYNTHESIS FOR NNUE HARDWARE ACCELERATOR         "
puts "  Target Device: $PART_NAME                                       "
puts "  Top Module:    $TOP_MODULE                                      "
puts "=================================================================="

# Set output directories
set OUTPUT_DIR "build"
file mkdir $OUTPUT_DIR

# Read SystemVerilog RTL Sources
read_verilog -sv rtl/nnue_pkg.sv
read_verilog -sv rtl/nnue_acc_core.sv
read_verilog -sv rtl/nnue_fc1_engine.sv
read_verilog -sv rtl/nnue_fc2_engine.sv
read_verilog -sv rtl/nnue_top.sv
read_verilog -sv rtl/nnue_axi_lite_wrapper.sv

# Copy memory initialization files into working directory for synthesis
set mem_files [glob -nocomplain mem/*.mem]
foreach f $mem_files {
    file copy -force $f .
}

# Create 100 MHz Clock Constraint (10.0 ns)
create_clock -period 10.000 -name clk [get_ports clk]

# Run Out-Of-Context (OOC) Synthesis
puts "--- Running synth_design ---"
synth_design -top $TOP_MODULE -part $PART_NAME -mode out_of_context

# Generate Utilization and Timing Reports
puts "--- Generating Reports in $OUTPUT_DIR/ ---"
report_utilization -file "$OUTPUT_DIR/utilization_synth.rpt" -pb "$OUTPUT_DIR/utilization_synth.pb"
report_timing_summary -file "$OUTPUT_DIR/timing_synth.rpt"
report_power -file "$OUTPUT_DIR/power_synth.rpt"

# Write Checkpoint and Netlist
write_checkpoint -force "$OUTPUT_DIR/post_synth.dcp"
write_verilog -force -mode funcsim "$OUTPUT_DIR/netlist_synth.v"

puts "=================================================================="
puts "  SYNTHESIS COMPLETE! Reports written to $OUTPUT_DIR/            "
puts "=================================================================="
