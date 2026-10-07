# Makefile macOS : compile le noyau matvec int8 x ternaire en .dylib.
#
# Usage :
#   make              -> détecte l'arch (uname -m) et compile le bon noyau
#   make portable     -> force la version C portable
#   make clean        -> supprime build/
#
# Notes :
# - Compilateur : clang d'Apple (Xcode Command Line Tools). Pas de gcc, pas
#   d'OpenMP par défaut (clang Apple ne le fournit pas). Multithread via
#   Grand Central Dispatch (dispatch_apply), dispo en standard sur macOS.
#   Option : `brew install libomp` puis ajouter -Xpreprocessor -fopenmp si
#   vous voulez expérimenter, non requis.
# - arm64 (M-series) -> kernels/neon.c (intrinsics NEON).
# - x86_64 (Intel)   -> kernels/avx2.c (compilé avec -mavx2).
# - Secours          -> kernels/portable.c (C99 strict).

UNAME_M := $(shell uname -m)
BUILD   := build
OUT     := $(BUILD)/libmicro2bit.dylib

CC      ?= clang
CFLAGS  := -O3 -Wall -Wextra -std=c99 -fPIC -Ikernels
LDFLAGS := -dynamiclib

ifeq ($(UNAME_M),arm64)
SRC := kernels/neon.c
VARIANT := neon (arm64)
else ifeq ($(UNAME_M),x86_64)
SRC := kernels/avx2.c
CFLAGS += -mavx2
VARIANT := avx2 (x86_64)
else
SRC := kernels/portable.c
VARIANT := portable ($(UNAME_M))
endif

all:
	@echo "arch=$(UNAME_M) variant=$(VARIANT)"
	mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(SRC) -o $(OUT) $(LDFLAGS)
	@echo "OK -> $(OUT)"

portable:
	@echo "variante portable forcée"
	mkdir -p $(BUILD)
	$(CC) -O3 -Wall -Wextra -std=c99 -fPIC -Ikernels kernels/portable.c -o $(OUT) $(LDFLAGS)
	@echo "OK -> $(OUT)"

clean:
	rm -rf $(BUILD)

.PHONY: all portable clean
