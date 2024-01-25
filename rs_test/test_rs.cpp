#include <iostream>
#include <immintrin.h>
#include <sys/mman.h>
#include <stdio.h>
#include <memory.h>
#include <x86intrin.h>
#include "asmjit/src/asmjit/asmjit.h"

using namespace asmjit;

// inspired by:
// https://blog.stuffedcow.net/2013/05/measuring-rob-capacity/
// https://zhuanlan.zhihu.com/p/457709438?utm_id=0
// Signature of the generated function.

asmjit::FileLogger logger;

static uint64_t buf1[1024 * 1024 * 100];
static uint64_t buf2[1024 * 1024 * 100];

typedef int (*Func)(size_t count, void* p1, void* p2);

void test(size_t count)
{
    // logger.setFlags(asmjit::FormatFlags::kHexOffsets);
    // logger.setFile(stdout);
    for (int fma_num = 1; fma_num < 200; fma_num += 1)
    {
        JitRuntime rt;                           // Runtime specialized for JIT code execution.
        CodeHolder code;                         // Holds code and relocation information.

        code.init(rt.environment(),              // Initialize code to match the JIT environment.
                rt.cpuFeatures());
        x86::Compiler cc(&code);                 // Create and attach x86::Compiler to code.

        code.setLogger(&logger);
        FuncNode* funcNode = cc.addFunc(         // Begin the function of the following signature:
            FuncSignature::build<void,           //   Return value - void      (no return value).
            size_t,                              //   1st argument - uint32_t* (machine reg-size).
            void*,                               //   2nd argument - uint32_t* (machine reg-size).
            void*>());                           //   3rd argument - size_t    (machine reg-size).
        funcNode->frame().setAvxEnabled();
        funcNode->frame().setAvx512Enabled();

        Label L_Loop = cc.newLabel();            // Start of the loop.
        Label L_Exit = cc.newLabel();            // Used to exit early.
        // RDI, RSI, RDX, RCX, R8, R9
        x86::Gp cnt = cc.newIntPtr("count");      // Create `i` register (loop counter).
        x86::Gp p0 = cc.newUIntPtr("p0");       // Create `dst` register (destination pointer).
        x86::Gp p1 = cc.newUIntPtr("p1");       // Create `src` register (source pointer).

        funcNode->setArg(0, cnt);                // Assign `dst` argument.
        funcNode->setArg(1, p0);                // Assign `src` argument.
        funcNode->setArg(2, p1);                  // Assign `i` argument.
        x86::Ymm y0 = cc.newYmm("y0");
        x86::Ymm y1 = cc.newYmm("y1");
        x86::Ymm s0 = cc.newYmm("s0");
        x86::Ymm s1 = cc.newYmm("s1");
        cc.vpxor(s0, s0, s0);
        cc.vpxor(s1, s1, s1);

        x86::Gp p0_c = cc.newUInt64();
        x86::Gp p1_c = cc.newUInt64();

        cc.align(AlignMode::kCode, 32);
        cc.bind(L_Loop);                         // Bind the beginning of the loop here.
        cc.mov(p0, x86::ptr(p0));        // Load DWORD from [src] address.
        cc.vmovd(y0.xmm(), p0);
        for (size_t i = 0; i < fma_num; i++) {
            cc.vpaddd(s0, s0, y0);
        }
        cc.mov(p1, x86::ptr(p1));        // Load DWORD from [src] address.
        cc.vmovd(y1.xmm(), p1);
        for (size_t i = 0; i < fma_num; i++) {
            cc.vpaddd(s1, s1, y1);
        }

        cc.dec(cnt);                               // Loop until `i` is non-zero.
        cc.jnz(L_Loop);

        cc.endFunc();                            // End of the function body.

        cc.finalize();                           // Translate and assemble the whole 'cc' content.
        // ----> x86::Compiler is no longer needed from here and can be destroyed <----

        // Add the generated code to the runtime.
        Func fn;
        Error err = rt.add(&fn, &code);

        // Handle a possible error returned by AsmJit.
        if (err)
        {
            printf("error rt.add\n");
            return; // Handle a possible error returned by AsmJit.
        }

        unsigned int t0 = 0;
        unsigned int t1 = 0;
        u_int64_t t00 = 0;
        u_int64_t t01 = 0;

        _mm_mfence();
        t00 = __rdtscp(&t0);
        _mm_mfence();
        fn(count, (void*)buf1, (void*)buf2);

        _mm_mfence();
        t01 = __rdtscp(&t1);
        auto cycles = (int64_t)(t01 - t00);
        printf("fma_num %d total %ld\n", fma_num, cycles);

        rt.release(fn); // Explicitly remove the function from the runtime
        code.reset();
    }
}

// copy from https://blog.stuffedcow.net/wp-content/uploads/2013/05/robsize.cc
static inline unsigned long long my_rand (unsigned long long limit)
{
	return ((unsigned long long)(((unsigned long long)rand()<<48)^((unsigned long long)rand()<<32)^((unsigned long long)rand()<<16)^(unsigned long long)rand())) % limit;
}

void init(uint64_t* dbuf, int size, int cycle_length)
{
	for (int i=0;i<size;i++)
		dbuf[i] = (uint64_t)&dbuf[i];
	for (int i=size-1;i>0;i--)
	{
		if (i & 0x1ff) continue;
		if (i < cycle_length) continue;
		unsigned int k = my_rand(i/cycle_length) * cycle_length + (i%cycle_length);
		auto temp = dbuf[i];
		dbuf[i] = dbuf[k];
		dbuf[k] = temp;
	}
}

/*
 perf stat -r 5 -e cycles:u,instructions:u,cpu_core/event=0xA2,umask=0x4,name=RESOURCE_STALLS.RS/u,cpu_core/event=0xA2,umask=0x1,name=RESOURCE_STALLS.ANY/u,cpu_core/event=0xA2,umask=0x10,name=RESOURCE_STALLS.ROB/u,cpu_core/event=0x9C,umask=0x1,inv=1,cmask=1,name=IDQ_UOPS_NOT_DELIVERED.CYCLES_FE_WAS_OK/u,cpu_core/event=0xE,umask=0x1,inv=1,cmask=1,name=UOPS_ISSUED.STALL_CYCLES/u,cpu_core/event=0x79,umask=0x4,cmask=1,name=IDQ.MITE_CYCLES/u,cpu_core/event=0x5E,umask=0x1,name=RS_EVENTS.EMPTY_CYCLES/u,cpu_core/event=0x79,umask=0x8,cmask=1,name=IDQ.DSB_CYCLES/u -- numactl -C 8 ./mytest
*/

int main(int argc, char *argv[])
{
    init(buf1, sizeof(buf1) / sizeof(uint64_t), 8192);
    init(buf2, sizeof(buf2) / sizeof(uint64_t), 8192);
    test(1000000);
    return 0;
}