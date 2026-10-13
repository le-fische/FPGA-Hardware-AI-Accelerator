# Makefile for FPGA Hardware AI Accelerator (NNUE Model B)
# Target Simulator: Icarus Verilog 13.0 / Verilator / Vivado xsim

IVERILOG ?= iverilog
VVP      ?= vvp
GTKWAVE  ?= gtkwave

RTL_SRCS = \
	rtl/nnue_pkg.sv \
	rtl/nnue_acc_core.sv \
	rtl/nnue_fc1_engine.sv \
	rtl/nnue_fc2_engine.sv \
	rtl/nnue_top.sv

TB_SRCS = sim/tb_nnue_top.sv
SIM_BIN = sim/sim_nnue
VCD_OUT = sim/nnue_sim.vcd

.PHONY: all sim wave clean

all: sim

$(SIM_BIN): $(RTL_SRCS) $(TB_SRCS)
	@mkdir -p sim
	$(IVERILOG) -g2012 -I rtl -o $(SIM_BIN) $(RTL_SRCS) $(TB_SRCS)

sim: $(SIM_BIN)
	$(VVP) $(SIM_BIN)

wave: sim
	@if command -v $(GTKWAVE) >/dev/null 2>&1; then \
		$(GTKWAVE) $(VCD_OUT) & \
	else \
		echo "GTKWave not installed. View $(VCD_OUT) in your waveform viewer."; \
	fi

clean:
	rm -f $(SIM_BIN) $(VCD_OUT)
