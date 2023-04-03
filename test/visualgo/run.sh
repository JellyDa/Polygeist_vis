#!/bin/bash
../../build/bin/cgeist testDump.cpp -emit-llvm -S -o testDump.ll
export LD_LIBRARY_PATH=.
clang++-12 testDump.ll -l dump -L.
./a.out