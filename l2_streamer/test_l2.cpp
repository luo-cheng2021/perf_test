// inspired by: https://abertschi.ch/blog/2022/prefetching/

#include <immintrin.h>
#include <sys/mman.h>
#include <stdio.h>
#include <memory.h>
#include <x86intrin.h>
/*
* Sample code snippet to illustrate the gist of the timing measurements.
* Here we measure prefetching impact when accessing TARGET,
* a single cache line.
*/
const int N = 1000;
const int M = 64;
const long CL = 64;
const int TARGET = 4;

char* probe_array;
void init() {
    probe_array = (char*)mmap(NULL, M * CL,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    memset(probe_array, 1, M * CL);
}

static inline int measure_access(char *addr) {
    unsigned int t0 = 0;
    unsigned int t1 = 0;
    u_int64_t t00 = 0;
    u_int64_t t01 = 0;

    _mm_mfence();
    t00 = __rdtscp(&t0);
    _mm_mfence();

    *(volatile char *) addr;

    _mm_mfence();
    t01 = __rdtscp(&t1);
    int cycles = (int) (t01 - t00);
    return cycles;
}
static char array[2 * 1024 * 1024];
void test() {
    size_t access_times[M];
    for(int i = 0; i < M; i++) {
        access_times[i] = 0;
    }

    // # of measurements
    for(int n = 0; n < N; n++) {

        // only probe one cacheline per experiment
        for(int el = 0; el < M; el++) {

            // flush the probing array
            for(int i = 0; i < M; i++) {
                _mm_clflush(probe_array + i * CL);
            }
            memset(array, 2, sizeof(array));
            for(int i = 0; i < M; i++) {
                _mm_clflush(probe_array + i * CL);
            }
            _mm_mfence();

            // Desired access pattern
            // auto a0 = _mm256_load_si256((__m256i*)(probe_array + (TARGET + 0) * CL));
            // auto a1 = _mm256_load_si256((__m256i*)(probe_array + (TARGET + 1) * CL));
            // auto a2 = _mm256_load_si256((__m256i*)(probe_array + (TARGET + 2) * CL));
            // auto a3 = _mm256_load_si256((__m256i*)(probe_array + (TARGET + 3) * CL));
            // auto a4 = _mm256_load_si256((__m256i*)(probe_array + (TARGET + 4) * CL));
            // auto b0 = _mm256_add_epi32(a0, a1);
            // auto b1 = _mm256_add_epi32(a2, a3);
            // b0 = _mm256_add_epi32(b0, b1);
            *(volatile char*)(probe_array + TARGET * CL);
            *(volatile char*)(probe_array + (TARGET + 1) * CL);
            *(volatile char*)(probe_array + (TARGET + 2) * CL);
            *(volatile char*)(probe_array + (TARGET + 3) * CL);
            // *(volatile char*)(probe_array + (TARGET + 4) * CL);
            // *(volatile char*)(probe_array + (TARGET + 5) * CL);
            _mm_mfence();
            *(volatile char*)(probe_array + (TARGET - TARGET + 8) * CL);
            *(volatile char*)(probe_array + (TARGET - TARGET + 13) * CL);
            *(volatile char*)(probe_array + (TARGET - TARGET + 14) * CL);
            // *(volatile char*)(probe_array + (TARGET - TARGET + 8) * CL);
            for(int x = 0; x < 3000; x++);
            _mm_mfence();

            // check which elements have been prefetched
            access_times[el] += measure_access(probe_array + el * CL);
        }
    }
    printf("result:\n");
    for(int i = 0; i < M; i++) {
        printf("%d: %ld\n", i, access_times[i]);
    }
}

// gcc test_l2.cpp -O0 -g -lstdc++ -lm -mavx2 -mfma -o test && numactl -C2 ./test
int main(int argc, char* argv[]) {
    init();
    test();
    return 0;
}