# ==============================================================================
# Vivado IP Packaging Script
# Packages the NNUE Hardware Accelerator into a reusable Vivado IP Core
# ==============================================================================

set IP_NAME "nnue_accelerator"
set IP_VERSION "1.0"
set IP_DIR "ip_repo/nnue_accelerator_v1_0"

file mkdir $IP_DIR

create_project -force tmp_ip_project ./tmp_ip_proj -part xc7z020clg400-1

# Add RTL Files
add_files -norecurse {
    rtl/nnue_pkg.sv
    rtl/nnue_acc_core.sv
    rtl/nnue_fc1_engine.sv
    rtl/nnue_fc2_engine.sv
    rtl/nnue_top.sv
    rtl/nnue_axi_lite_wrapper.sv
}

# Add Memory Files
add_files -norecurse {
    mem/biases_acc.mem
    mem/weights_acc.mem
    mem/biases_fc1.mem
    mem/weights_fc1.mem
    mem/biases_fc2.mem
    mem/weights_fc2.mem
}

# Package IP
ipx::package_project -root_dir $IP_DIR -vendor "silicon_project" -library "user" -taxonomy "/Accelerators" -import_files -set_current false

set core [ipx::open_ipxact_file $IP_DIR/component.xml]
set_property name $IP_NAME $core
set_property version $IP_VERSION $core
set_property display_name "Stockfish-Style NNUE Hardware Accelerator" $core
set_property description "Cycle-accurate dual-perspective NNUE neural evaluation engine with AXI4-Lite CSR interface" $core
set_property vendor_display_name "The Silicon Project" $core

ipx::save_core $core
close_project -delete

puts "=================================================================="
puts "  IP Core packaged successfully at $IP_DIR/                      "
puts "=================================================================="
