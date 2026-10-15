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

VERILATOR ?= verilator
CLANGXX   ?= clang++
PYTHON    ?= /Users/le/Documents/Projects/chessathon/project/.venv/bin/python

COSIM_LIB = sim/libnnue_cosim.dylib
VERILATOR_DIR = sim/verilator_build/obj_dir

.PHONY: all sim wave verilator cosim clean

all: sim

$(SIM_BIN): $(RTL_SRCS) $(TB_SRCS)
	@mkdir -p sim
	$(IVERILOG) -g2012 -I rtl -o $(SIM_BIN) $(RTL_SRCS) $(TB_SRCS)

sim: $(SIM_BIN)
	$(VVP) $(SIM_BIN)

$(COSIM_LIB): $(RTL_SRCS) sim/nnue_cosim_lib.cpp
	@mkdir -p $(VERILATOR_DIR)
	$(VERILATOR) --cc -Wall -Wno-UNUSEDPARAM -Wno-IMPORTSTAR -Irtl rtl/nnue_top.sv --Mdir $(VERILATOR_DIR)
	$(CLANGXX) -O3 -fPIC -shared -std=c++17 \
		-I$$(dirname $$(dirname $$(which $(VERILATOR))))/share/verilator/include \
		-I$$(dirname $$(dirname $$(which $(VERILATOR))))/share/verilator/include/vltstd \
		-I$(VERILATOR_DIR) \
		sim/nnue_cosim_lib.cpp \
		$(VERILATOR_DIR)/Vnnue_top.cpp \
		$(VERILATOR_DIR)/Vnnue_top___024root__0.cpp \
		$(VERILATOR_DIR)/Vnnue_top___024root__0__Slow.cpp \
		$(VERILATOR_DIR)/Vnnue_top___024root__Slow.cpp \
		$(VERILATOR_DIR)/Vnnue_top___024unit__Slow.cpp \
		$(VERILATOR_DIR)/Vnnue_top__Syms__Slow.cpp \
		$$(dirname $$(dirname $$(which $(VERILATOR))))/share/verilator/include/verilated.cpp \
		$$(dirname $$(dirname $$(which $(VERILATOR))))/share/verilator/include/verilated_threads.cpp \
		-o $(COSIM_LIB)

verilator: $(COSIM_LIB)

cosim: $(COSIM_LIB)
	$(PYTHON) tools/cosim_engine.py

wave: sim
	@if command -v $(GTKWAVE) >/dev/null 2>&1; then \
		$(GTKWAVE) $(VCD_OUT) & \
	else \
		echo "GTKWave not installed. View $(VCD_OUT) in your waveform viewer."; \
	fi

clean:
	rm -rf $(SIM_BIN) $(VCD_OUT) $(COSIM_LIB) sim/verilator_build

