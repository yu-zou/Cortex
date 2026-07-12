# Cortex ARM Scheduler Makefile
# Target: MPSoC Cortex-A53 (ARMv8-A)
# Compile: make
# Cross-compile example: make CROSS=arm-linux-gnueabihf-

CROSS ?= arm-linux-gnueabihf-
CC   = $(CROSS)gcc
CFLAGS = -static -Wall -O2 -pthread
TARGET = scheduler

.PHONY: all clean test

all: $(TARGET)

$(TARGET): scheduler.c
	$(CC) $(CFLAGS) -o $@ $<

# Prototype: compile with host GCC for local verification
host:
	gcc -Wall -O2 -pthread -DHOST_BUILD -o scheduler_host scheduler.c
	@echo "Built scheduler_host (host GCC, for local test)"

test: host
	./scheduler_host 0

clean:
	rm -f $(TARGET) scheduler_host
