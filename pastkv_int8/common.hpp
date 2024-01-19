// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>


// avx512/avx2 register length in byte
static constexpr size_t vec_len_avx512 = 64lu;
static constexpr size_t vec_len_avx2 = 32lu;
// avx512/avx2 register length in float
static constexpr size_t vec_len_f32_avx512 = vec_len_avx512 / sizeof(float);
static constexpr size_t vec_len_f32_avx2 = vec_len_avx2 / sizeof(float);

#ifdef HAVE_AVX512F
    inline __m512 mm512_uni_loadu_ps(ov::bfloat16* a) {
        auto vec_bf16 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a));
        __m512i y = _mm512_cvtepu16_epi32(vec_bf16);
        return _mm512_castsi512_ps(_mm512_slli_epi32(y, 16));
    }
    inline __m512 mm512_uni_loadu_ps(float* a) {
        return _mm512_loadu_ps(a);
    }
    inline void mm512_uni_storeu_ps(float* a,  __m512 v) {
        _mm512_storeu_ps(a, v);
    }
#endif

#ifdef HAVE_AVX2
    inline __m256 mm256_uni_loadu_ps(float* a) {
        return _mm256_loadu_ps(a);
    }
    inline void mm256_uni_storeu_ps(float* a,  __m256 v) {
        _mm256_storeu_ps(a, v);
    }

    inline void hsum(__m256& x) {
        __m256 y;                             // x:  0 1 2 3   4 5 6 7
        y = _mm256_permute_ps(x, 0x39);       // y:  1 2 3 0   5 6 7 4
        x = _mm256_add_ps(x, y);              // X:  01 12 23 30  45 56 67 74
        y = _mm256_permute_ps(x, 0x4e);       // y:  23 30 01 12  67 74 45 56
        x = _mm256_add_ps(x, y);              // x: 0123 x x x   4567 x x x
        y = _mm256_permute2f128_ps(x, x, 1);  // y: 4567 x x x  0123 x x x
        x = _mm256_add_ps(x, y);              // x: 01234567 x x x x x x x
    }
    inline void hmax(__m256& x) {
        __m256 y;                             // x:  0 1 2 3   4 5 6 7
        y = _mm256_permute_ps(x, 0x39);       // y:  1 2 3 0   5 6 7 4
        x = _mm256_max_ps(x, y);              // X:  01 12 23 30  45 56 67 74
        y = _mm256_permute_ps(x, 0x4e);       // y:  23 30 01 12  67 74 45 56
        x = _mm256_max_ps(x, y);              // x: 0123 x x x   4567 x x x
        y = _mm256_permute2f128_ps(x, x, 1);  // y: 4567 x x x  0123 x x x
        x = _mm256_max_ps(x, y);              // x: 01234567 x x x x x x x
    }
    inline void hmin(__m256& x) {
        __m256 y;                             // x:  0 1 2 3   4 5 6 7
        y = _mm256_permute_ps(x, 0x39);       // y:  1 2 3 0   5 6 7 4
        x = _mm256_min_ps(x, y);              // X:  01 12 23 30  45 56 67 74
        y = _mm256_permute_ps(x, 0x4e);       // y:  23 30 01 12  67 74 45 56
        x = _mm256_min_ps(x, y);              // x: 0123 x x x   4567 x x x
        y = _mm256_permute2f128_ps(x, x, 1);  // y: 4567 x x x  0123 x x x
        x = _mm256_min_ps(x, y);              // x: 01234567 x x x x x x x
    }
#endif
