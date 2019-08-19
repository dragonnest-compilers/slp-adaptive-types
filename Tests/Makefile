LLVM_PATH = /usr/lib/llvm/vwslp

#We are considering that you have already compiled the pass in LLVM_PATH
all:
	clang -emit-llvm -O0 -Xclang -disable-O0-optnone -c -S teste.c
	clang -emit-llvm -O0 -Xclang -disable-O0-optnone -c teste.c -o teste.bc
	opt -load $(LLVM_PATH)/build/lib/MiniSLP.so -mini-slp < teste.bc > teste-opt.bc
	llvm-dis teste-opt.bc -o teste-opt.ll
