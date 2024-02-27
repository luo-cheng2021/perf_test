// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <float.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>

#include <immintrin.h>
#include <x86intrin.h>

#define HAVE_AVX2
#define OV_THREAD OV_THREAD_OMP
#include "parallel.hpp"
#include "common.hpp"
#include "softmax_kernel.hpp"
#include "plain_tensor.hpp"
#include <chrono>

using namespace ov;
template<typename T>
static void attn_acc_value(float* out, float weight, T* v, size_t S, float scale, float zp) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    auto attn_w_vec_fp32 = _mm512_set1_ps(weight);
    for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
        auto v_value = mm512_uni_loadu_ps(v + i);
        auto v_out = mm512_uni_loadu_ps(out + i);
        v_out = _mm512_fmadd_ps(attn_w_vec_fp32, v_value, v_out);
        _mm512_storeu_ps(out + i, v_out);
    }
#elif defined(HAVE_AVX2)
    auto attn_w_vec_fp32 = _mm256_set1_ps(weight);
    for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
        auto v_value = mm256_uni_loadu_ps(v + i);
        auto v_out = mm256_uni_loadu_ps(out + i);
        v_out = _mm256_fmadd_ps(attn_w_vec_fp32, v_value, v_out);
        mm256_uni_storeu_ps(out + i, v_out);
    }
#endif
    for (; i < S; i++) {
        out[i] += weight * v[i];
    }
}

static void attn_acc_value(float* out, float weight, uint8_t* v, size_t S, float scale, float zp) {
    size_t i = 0;
    weight *= scale;
// #if defined(HAVE_AVX512F)
//     auto attn_w_vec_fp32 = _mm512_set1_ps(weight);
//     for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
//         auto v_value = mm512_uni_loadu_ps(v + i);
//         auto v_out = mm512_uni_loadu_ps(out + i);
//         v_out = _mm512_fmadd_ps(attn_w_vec_fp32, v_value, v_out);
//         _mm512_storeu_ps(out + i, v_out);
//     }
#if defined(HAVE_AVX2)
    auto attn_w_vec_fp32 = _mm256_set1_ps(weight);
    auto v_zp = _mm256_set1_ps(zp);
    for (; i + 4 * vec_len_f32_avx2 <= S; i += 4* vec_len_f32_avx2) {
        auto v0_128 = _mm_loadu_si64(v + i);
        auto v1_128 = _mm_loadu_si64(v + i + vec_len_f32_avx2);
        auto v2_128 = _mm_loadu_si64(v + i + vec_len_f32_avx2 * 2);
        auto v3_128 = _mm_loadu_si64(v + i + vec_len_f32_avx2 * 3);

        auto v0_out = mm256_uni_loadu_ps(out + i);
        auto v1_out = mm256_uni_loadu_ps(out + i + vec_len_f32_avx2);
        auto v2_out = mm256_uni_loadu_ps(out + i + vec_len_f32_avx2 * 2);
        auto v3_out = mm256_uni_loadu_ps(out + i + vec_len_f32_avx2 * 3);

        auto v0_256 = _mm256_cvtepu8_epi32(v0_128);
        auto v1_256 = _mm256_cvtepu8_epi32(v1_128);
        auto v2_256 = _mm256_cvtepu8_epi32(v2_128);
        auto v3_256 = _mm256_cvtepu8_epi32(v3_128);

        auto v0_value = _mm256_cvtepi32_ps(v0_256);
        auto v1_value = _mm256_cvtepi32_ps(v1_256);
        auto v2_value = _mm256_cvtepi32_ps(v2_256);
        auto v3_value = _mm256_cvtepi32_ps(v3_256);

        v0_value = _mm256_sub_ps(v0_value, v_zp);
        v1_value = _mm256_sub_ps(v1_value, v_zp);
        v2_value = _mm256_sub_ps(v2_value, v_zp);
        v3_value = _mm256_sub_ps(v3_value, v_zp);

        v0_out = _mm256_fmadd_ps(attn_w_vec_fp32, v0_value, v0_out);
        v1_out = _mm256_fmadd_ps(attn_w_vec_fp32, v1_value, v1_out);
        v2_out = _mm256_fmadd_ps(attn_w_vec_fp32, v2_value, v2_out);
        v3_out = _mm256_fmadd_ps(attn_w_vec_fp32, v3_value, v3_out);

        mm256_uni_storeu_ps(out + i + vec_len_f32_avx2 * 0, v0_out);
        mm256_uni_storeu_ps(out + i + vec_len_f32_avx2 * 1, v1_out);
        mm256_uni_storeu_ps(out + i + vec_len_f32_avx2 * 2, v2_out);
        mm256_uni_storeu_ps(out + i + vec_len_f32_avx2 * 3, v3_out);
    }
    if (i + 2 * vec_len_f32_avx2 <= S) {
        auto v0_128 = _mm_loadu_si64(v + i);
        auto v1_128 = _mm_loadu_si64(v + i + vec_len_f32_avx2);

        auto v0_out = mm256_uni_loadu_ps(out + i);
        auto v1_out = mm256_uni_loadu_ps(out + i + vec_len_f32_avx2);

        auto v0_256 = _mm256_cvtepu8_epi32(v0_128);
        auto v1_256 = _mm256_cvtepu8_epi32(v1_128);

        auto v0_value = _mm256_cvtepi32_ps(v0_256);
        auto v1_value = _mm256_cvtepi32_ps(v1_256);

        v0_value = _mm256_sub_ps(v0_value, v_zp);
        v1_value = _mm256_sub_ps(v1_value, v_zp);

        v0_out = _mm256_fmadd_ps(attn_w_vec_fp32, v0_value, v0_out);
        v1_out = _mm256_fmadd_ps(attn_w_vec_fp32, v1_value, v1_out);

        mm256_uni_storeu_ps(out + i + vec_len_f32_avx2 * 0, v0_out);
        mm256_uni_storeu_ps(out + i + vec_len_f32_avx2 * 1, v1_out);
        i += 2 * vec_len_f32_avx2;
    }
    if (i + vec_len_f32_avx2 <= S) {
        auto v0_128 = _mm_loadu_si64(v + i);
        auto v0_out = mm256_uni_loadu_ps(out + i);
        auto v0_256 = _mm256_cvtepu8_epi32(v0_128);
        auto v0_value = _mm256_cvtepi32_ps(v0_256);
        v0_value = _mm256_sub_ps(v0_value, v_zp);
        v0_out = _mm256_fmadd_ps(attn_w_vec_fp32, v0_value, v0_out);
        mm256_uni_storeu_ps(out + i, v0_out);
        i += vec_len_f32_avx2;
    }
#endif
    for (; i < S; i++) {
        out[i] += weight * (v[i] - zp);
    }
}

template<typename T>
static float sum_q_head(T* a, size_t n) {
    float sum = 0.0f;
    size_t i = 0;
#if defined(HAVE_AVX512F)

#elif defined(HAVE_AVX2)
    auto vsum0 = _mm256_set1_ps(0.0f);
    auto vsum1 = _mm256_set1_ps(0.0f);
    auto vsum2 = _mm256_set1_ps(0.0f);
    auto vsum3 = _mm256_set1_ps(0.0f);
    for (; i + 4 * vec_len_f32_avx2 <= n; i += vec_len_f32_avx2 * 4) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);
        auto va2 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 2);
        auto va3 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 3);

        vsum0 = _mm256_add_ps(va0, vsum0);
        vsum1 = _mm256_add_ps(va1, vsum1);
        vsum2 = _mm256_add_ps(va2, vsum2);
        vsum3 = _mm256_add_ps(va3, vsum3);
    }
    if (i + 2 * vec_len_f32_avx2 <= n) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);

        vsum0 = _mm256_add_ps(va0, vsum0);
        vsum1 = _mm256_add_ps(va1, vsum1);
        i += 2 * vec_len_f32_avx2;
    }
    if (i + vec_len_f32_avx2 <= n) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        vsum0 = _mm256_add_ps(va0, vsum0);
        i += vec_len_f32_avx2;
    }
    vsum0 = _mm256_add_ps(vsum0, vsum1);
    vsum2 = _mm256_add_ps(vsum2, vsum3);
    vsum0 = _mm256_add_ps(vsum0, vsum2);
    hsum(vsum0);
    sum = _mm256_cvtss_f32(vsum0);
#endif

    for (; i < n; i++) {
        float tmp = a[i];
        sum += tmp;
    }
    return sum;
}

template<typename TA, typename TB>
static float dot_product(TA* a, TB* b, size_t n, float* scale, float* zp, float* head_sum) {
    size_t i = 0;
    float sum = 0.0f;
#if defined(HAVE_AVX512F)
    auto vsum = _mm512_setzero_ps();
    for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
        auto va = mm512_uni_loadu_ps(a + i);
        auto vb = mm512_uni_loadu_ps(b + i);
        vsum = _mm512_fmadd_ps(va, vb, vsum);
    }
    sum = _mm512_reduce_add_ps(vsum);
#elif defined(HAVE_AVX2)
    auto vsum = _mm256_set1_ps(0.0f);
    for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
        auto va = mm256_uni_loadu_ps(a + i);
        auto vb = mm256_uni_loadu_ps(b + i);
        vsum = _mm256_fmadd_ps(va, vb, vsum);
    }
    hsum(vsum);
    sum = _mm256_cvtss_f32(vsum);
#endif
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

template<typename TA>
static float dot_product(TA* a, uint8_t* b, size_t n, float* scale, float* zp, float* head_sum) {
    size_t i = 0;
    float sum = 0.0f;
// #if defined(HAVE_AVX512F)
//     auto vsum = _mm512_setzero_ps();
//     for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
//         auto va = mm512_uni_loadu_ps(a + i);
//         auto vb = mm512_uni_loadu_ps(b + i);
//         vsum = _mm512_fmadd_ps(va, vb, vsum);
//     }
//     sum = _mm512_reduce_add_ps(vsum);
#if defined(HAVE_AVX2)
    auto vsum0 = _mm256_set1_ps(0.0f);
    auto vsum1 = _mm256_set1_ps(0.0f);
    auto vsum2 = _mm256_set1_ps(0.0f);
    auto vsum3 = _mm256_set1_ps(0.0f);
    for (; i + 4 * vec_len_f32_avx2 <= n; i += vec_len_f32_avx2 * 4) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);
        auto va2 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 2);
        auto va3 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 3);

        auto vb0_128 = _mm_loadu_si64(b + i);
        auto vb1_128 = _mm_loadu_si64(b + i + vec_len_f32_avx2);
        auto vb2_128 = _mm_loadu_si64(b + i + vec_len_f32_avx2 * 2);
        auto vb3_128 = _mm_loadu_si64(b + i + vec_len_f32_avx2 * 3);

        auto vb0_256 = _mm256_cvtepu8_epi32(vb0_128);
        auto vb1_256 = _mm256_cvtepu8_epi32(vb1_128);
        auto vb2_256 = _mm256_cvtepu8_epi32(vb2_128);
        auto vb3_256 = _mm256_cvtepu8_epi32(vb3_128);

        auto vb0 = _mm256_cvtepi32_ps(vb0_256);
        auto vb1 = _mm256_cvtepi32_ps(vb1_256);
        auto vb2 = _mm256_cvtepi32_ps(vb2_256);
        auto vb3 = _mm256_cvtepi32_ps(vb3_256);

        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
        vsum2 = _mm256_fmadd_ps(va2, vb2, vsum2);
        vsum3 = _mm256_fmadd_ps(va3, vb3, vsum3);
    }
    if (i + 2 * vec_len_f32_avx2 <= n) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);

        auto vb0_128 = _mm_loadu_si64(b + i);
        auto vb1_128 = _mm_loadu_si64(b + i + vec_len_f32_avx2);

        auto vb0_256 = _mm256_cvtepu8_epi32(vb0_128);
        auto vb1_256 = _mm256_cvtepu8_epi32(vb1_128);

        auto vb0 = _mm256_cvtepi32_ps(vb0_256);
        auto vb1 = _mm256_cvtepi32_ps(vb1_256);

        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
        i += 2 * vec_len_f32_avx2;
    }
    if (i + vec_len_f32_avx2 <= n) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto vb0_128 = _mm_loadu_si64(b + i);
        auto vb0_256 = _mm256_cvtepu8_epi32(vb0_128);
        auto vb0 = _mm256_cvtepi32_ps(vb0_256);
        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        i += vec_len_f32_avx2;
    }
    vsum0 = _mm256_add_ps(vsum0, vsum1);
    vsum2 = _mm256_add_ps(vsum2, vsum3);
    vsum0 = _mm256_add_ps(vsum0, vsum2);
    hsum(vsum0);
    sum = _mm256_cvtss_f32(vsum0);
#endif
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    // B = scale * (b - zero)
    // Σ (A * B) = Σ (a * scale * (b - zero)) = scale * (Σ a * b - zero Σ a) = scale * (sum - zp * head_sum)
    return scale[0] * (sum - zp[0] * head_sum[0]);
}

template<typename T>
static void attn_reduce(T* dst, float* temp, size_t M, size_t S, size_t temp_stride) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    for (; i + vec_len_f32_avx512 <= S; i+= vec_len_f32_avx512) {
        auto* src = temp + i;
        auto result_vec_fp32 = _mm512_setzero_ps();
        for (size_t m = 0; m < M; m++) {
            //auto* temp = &m_temp.at({ithr, b, pq, h, 0});
            auto o_vec_fp32 = _mm512_loadu_ps(src);
            result_vec_fp32 = _mm512_add_ps(result_vec_fp32, o_vec_fp32);
            src += temp_stride;
        }
        // save to bf16
        mm512_uni_storeu_ps(dst + i, result_vec_fp32);
    }
#elif defined(HAVE_AVX2)
    for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
        auto* src = temp + i;
        auto result_vec_fp32 = _mm256_set1_ps(0.0f);
        for (size_t m = 0; m < M; m++) {
            auto o_vec_fp32 = mm256_uni_loadu_ps(src);
            result_vec_fp32 = _mm256_add_ps(result_vec_fp32, o_vec_fp32);
            src += temp_stride;
        }
        mm256_uni_storeu_ps(dst + i, result_vec_fp32);
    }
#endif
    for (; i < S; i++) {
        auto* src = temp + i;
        float sum = 0.0f;
        // sum result from all threads partition
        for (size_t m = 0; m < M; m++) {
            sum += src[0];
            src += temp_stride;
        }
        dst[i] = sum;
    }
}

#define prefetch_bytes(bytes, sel, advance, src) {  \
    int8_t *p = reinterpret_cast<int8_t *>(src);    \
    for (int i = 0; i < bytes; i+=64)               \
        _mm_prefetch(p + i + advance, sel);         \
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
// const int NN = 1000;
// const int M = 256;
const int NN = 1;
const int M = 1;
const long CL = 64;
size_t access_times[8][M];
struct exec {
    size_t cycles;
    size_t ns;
    size_t count;
    void reset() {
        memset(this, 0, sizeof(*this));
    }
} g_exec;
int g_nthr;
bool g_is_test = false;
static char array[8][3 * 1024 * 1024];
bool g_hit_cache = false;

template <typename T, typename T2>
static void mha_single_token_kernel(const PlainTensor& query,
                             const PlainTensor& present_key,
                             const PlainTensor& present_value,
                             const PlainTensor& alibi_mask,
                             const PlainTensor& attention_mask,
                             const PlainTensor& beams,
                             PlainTensor& output_emb,
                             PlainTensor& buf_attn_w,
                             PlainTensor& buf_attn_score,
                             bool has_out_transpose,
                             bool auto_causal,
                             float d_scale,
                             const PlainTensor& past_k_scale_zp,
                             const PlainTensor& past_v_scale_zp,
                             PlainTensor& head_sum,
                             int blocks) {
    PlainTensor causal_mask;
    bool select_nfltmax_at_0 = false;
    auto B = query.size(0);
    auto H = query.size(1);
    auto q_len = query.size(2);
    auto S = query.size(3);
    auto kv_len = present_key.size(2);
    auto h_group_num = present_key.size(1);
    size_t h_each_group_len = 1;
    if (h_group_num != H) {
        h_each_group_len = H / h_group_num;
    }
    if (d_scale == 0.0f)
        d_scale = 1.0f / sqrt(S);
    auto nthr = parallel_get_max_threads();

    // use per-token kernel, for each k,v token
    //  attn mask is a matrix of q_len(kv_len)
    buf_attn_w.resize<float>({B, H, q_len, kv_len});

    bool pastkv_is_int8 = past_k_scale_zp;
    if (pastkv_is_int8) {
        // be sure no false sharing
        head_sum.resize<float>({B, H, q_len, 16});
        parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
            head_sum.at<float>(b, h, pq, false) = sum_q_head(&query.at<T>(b, h, pq, false), S);
        });
    }
    // unsigned int _t0 = 0;
    // unsigned int _t1 = 0;
    // u_int64_t _t00 = 0;
    // u_int64_t _t01 = 0;

    // auto start_ns = std::chrono::steady_clock::now();
    // _t00 = __rdtscp(&_t0);

    parallel_nt_static(nthr, [&](const size_t ithr, const size_t nthr) {

    unsigned int _t0 = 0;
    unsigned int _t1 = 0;
    u_int64_t _t00 = 0;
    u_int64_t _t01 = 0;
    auto cur_kv_len = 1025;//kv_len;

    auto start_ns = std::chrono::steady_clock::now();
    _t00 = __rdtscp(&_t0);

        size_t start{0}, end{0};
        splitter(B * h_group_num * cur_kv_len, nthr, ithr, start, end);

        size_t b, h_group, pk;
        if (blocks > 0)
            end = start + blocks;
        if (start < end) {
            for (int nn = 0; nn < NN; nn++)
            for (int el = 0; el < M; el++) {
                parallel_it_init(start, b, B, h_group, h_group_num, pk, cur_kv_len);
                // for (size_t iwork = start; iwork < end; ++iwork) {
                //     auto b_kv = beams ? beams.at<int32_t>(b, pk) : b;
                //     auto p = &past_k_scale_zp.at<float>(b_kv, h_group, pk, false);
                //     buf_attn_w.at<float>(b, h_group, 0, pk) =
                //             dot_product(&query.at<T>(b, h_group, false), &present_key.at<T2>(b_kv, h_group, pk, false),
                //                 S, p, p + 1, &head_sum.at<float>(b, h_group, false));
                //     parallel_it_step(b, B, h_group, h_group_num, pk, cur_kv_len);
                // }
                auto p = &past_k_scale_zp.at<float>(b, h_group, pk, false);
                // [b, h, L0+L1, S]
                auto p_present_key = &present_key.at<T2>(b, h_group, pk, false);
                auto pp = p_present_key;
                auto p_q = &query.at<T>(b, h_group, false);
#if 0
                auto p_k = pp;
                for (size_t iwork = start; iwork < end; ++iwork) {
                    //auto p = &past_k_scale_zp.at<float>(b_kv, h_group, pk, false);
                    //auto p_k = &present_key.at<T2>(b_kv, h_group, pk, false);
                    // _mm_prefetch(p_k + 4096*1, _MM_HINT_T0);
                    // _mm_prefetch(p_k + 4096*1 + 64, _MM_HINT_T0);
                    for (size_t i = 0; i < S; i += 64) {
                        *(volatile int*)(p_k + i);
                    }
                    p_k += S;
                    if (++pk == cur_kv_len) {
                        pk = 0;
                        p_k += present_key.m_strides[1] - S * cur_kv_len;
                        if (++h_group == h_group_num) {
                            h_group = 0;
                            if (++b == B)
                                break;
                        }
                    }
                }
#else
                for (size_t iwork = start; iwork < end; ++iwork) {
                    auto b_kv = beams ? beams.at<int32_t>(b, pk) : b;
                    //auto p = &past_k_scale_zp.at<float>(b_kv, h_group, pk, false);
                    // if (g_hit_cache)
                    //     p_present_key = &present_key.at<T2>(b_kv / 8, h_group / 64, pk / 2048, false);
                    // else
                    //     p_present_key = &present_key.at<T2>(b_kv, h_group, pk, false);
                    _mm_prefetch(p_present_key + 4096*1, _MM_HINT_T0);
                    _mm_prefetch(p_present_key + 4096*1 + 64, _MM_HINT_T0);
                    buf_attn_w.at<float>(b, h_group, 0, pk) =
                            dot_product(p_q, p_present_key, //&present_key.at<T2>(b_kv, h_group, pk, false),
                                S, p, p + 1, &head_sum.at<float>(b, h_group, false));
                    //parallel_it_step(b, B, h_group, h_group_num, pk, cur_kv_len);
                    if (!g_hit_cache) {
                        p_present_key += S;
                        p += 2;
                    }
                    if (++pk == cur_kv_len) {
                        pk = 0;
                        if (!g_hit_cache) {
                            p_present_key += present_key.m_strides[1] - S * cur_kv_len;
                            p += past_k_scale_zp.m_strides[1] - 2 * cur_kv_len;
                        }
                        p_q += S;
                        if (++h_group == h_group_num) {
                            h_group = 0;
                            if (++b == B)
                                break;
                        }
                    }
                }
#endif
            }
        }
        _t01 = __rdtscp(&_t1);
        size_t cycles = (size_t) (_t01 - _t00);
        if (g_is_test && ithr == 0) {
            g_exec.count++;
            g_exec.cycles += cycles;
            auto end_ns = std::chrono::steady_clock::now();
            g_exec.ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end_ns - start_ns).count();
        }
    });
    // _t01 = __rdtscp(&_t1);
    // size_t cycles = (size_t) (_t01 - _t00);
    // if (g_is_test) {
    //     g_exec.count++;
    //     g_exec.cycles += cycles;
    //     auto end_ns = std::chrono::steady_clock::now();
    //     g_exec.ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end_ns - start_ns).count();
    // }
    if (g_hit_cache)
        return;

    parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
        // apply attention mask & sofmax
        auto ncausal = auto_causal ? (kv_len - q_len + pq + 1) : kv_len;
        float* alibi_ptr = alibi_mask ? &alibi_mask.at<float>({b, h, pq, 0}, true) : nullptr;
        float* attn_mask_ptr = attention_mask ? &attention_mask.at<float>({b, h, pq, 0}, true) : nullptr;
        uint8_t* cmask_ptr = causal_mask ? &causal_mask.at<uint8_t>({b, h, pq, 0}, true) : nullptr;
        attn_softmax_kernel(&buf_attn_w.at<float>(b, h, pq, false),
                            &buf_attn_w.at<float>(b, h, pq, false),
                            d_scale,
                            alibi_ptr,
                            attn_mask_ptr,
                            cmask_ptr,
                            select_nfltmax_at_0,
                            ncausal,
                            kv_len);
    });

    // attn_w * V
    buf_attn_score.resize<float>({static_cast<size_t>(nthr), B, q_len, H, S});
    // buf_attn_w {B, H, q_len, kv_len}
    parallel_nt_static(nthr, [&](const size_t ithr, const size_t nthr) {
        size_t start{0}, end{0};
        splitter(B * h_group_num * kv_len, nthr, ithr, start, end);

        memset(&buf_attn_score.at<float>({ithr, 0, 0, 0, 0}), 0, buf_attn_score.stride(0) * sizeof(float));

        size_t b, h_group, pv;
        if (start < end) {
            parallel_it_init(start, b, B, h_group, h_group_num, pv, kv_len);
            if (q_len == 1 && h_each_group_len == 1) {
                for (size_t iwork = start; iwork < end; ++iwork) {
                    auto b_kv = beams ? beams.at<int32_t>(b, pv) : b;
                    auto* v = &present_value.at<T2>(b_kv, h_group, pv, false);
                    auto p = &past_v_scale_zp.at<float>(b_kv, h_group, pv, false);
                    attn_acc_value(&buf_attn_score.at<float>(ithr, b, 0, h_group, false),
                                buf_attn_w.at<float>(b, h_group, 0, pv),
                                v,
                                S,
                                p[0],
                                p[1]);
                    parallel_it_step(b, B, h_group, h_group_num, pv, kv_len);
                }
            } else {
                for (size_t iwork = start; iwork < end; ++iwork) {
                    auto b_kv = beams ? beams.at<int32_t>(b, pv) : b;
                    auto* v = &present_value.at<T2>(b_kv, h_group, pv, false);
                    auto p = &past_v_scale_zp.at<float>(b_kv, h_group, pv, false);
                    for (size_t pq = 0; pq < q_len; pq++) {
                        for (size_t h = h_group * h_each_group_len; h < (h_group + 1) * h_each_group_len; h++) {
                            attn_acc_value(&buf_attn_score.at<float>(ithr, b, pq, h, false),
                                        buf_attn_w.at<float>(b, h, pq, pv),
                                        v,
                                        S,
                                        p[0],
                                        p[1]);
                        }
                    }
                    parallel_it_step(b, B, h_group, h_group_num, pv, kv_len);
                }
            }
        }
    });

    parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
        auto* temp = &buf_attn_score.at<float>(0, b, pq, h, false);
        size_t temp_stride = buf_attn_score.stride(0);
        auto* dst = has_out_transpose ? &output_emb.at<T>(b, pq, h * S) : &output_emb.at<T>(b, h, pq);
        attn_reduce(dst, temp, nthr, S, temp_stride);
    });
}

void mha_single_token(const PlainTensor& query,
                      const PlainTensor& present_key,
                      const PlainTensor& present_value,
                      const PlainTensor& alibi_mask,
                      const PlainTensor& attention_mask,
                      const PlainTensor& beams,
                      PlainTensor& output_emb,
                      PlainTensor& buf_attn_w,
                      PlainTensor& buf_attn_score,
                      bool has_out_transpose,
                      bool auto_causal,
                      float d_scale,
                      const PlainTensor& past_k_scale_zp,
                      const PlainTensor& past_v_scale_zp,
                      PlainTensor& head_sum,
                      int blocks) {
        mha_single_token_kernel<float, uint8_t>(query,
                                                present_key,
                                                present_value,
                                                alibi_mask,
                                                attention_mask,
                                                beams,
                                                output_emb,
                                                buf_attn_w,
                                                buf_attn_score,
                                                has_out_transpose,
                                                auto_causal,
                                                d_scale,
                                                past_k_scale_zp,
                                                past_v_scale_zp,
                                                head_sum,
                                                blocks);
}

size_t B = 4, H = 32, L0 = 1025 - 1, L1 = 1, S = 128;
const size_t N = 32;

PlainTensor query;
PlainTensor present_key[N];
PlainTensor present_value[N];
PlainTensor alibi_mask;
PlainTensor attention_mask;
PlainTensor beams;
PlainTensor output_emb[N];
PlainTensor buf_attn_w;
PlainTensor buf_attn_score;
bool has_out_transpose = false;
bool auto_causal = false;
float d_scale = 1.0f;
PlainTensor past_k_scale_zp[N];
PlainTensor past_v_scale_zp[N];
PlainTensor head_sum;

void init(bool touch = false) {
    beams.resize<int32_t>({B, L0 + L1});
    for (size_t l = 0; l < L0 + L1; l++)
        for (size_t b = 0; b < B; b++) {
            beams.at<int32_t>(b, l) = b;
        }
    for (size_t n = 0; n < N; n++) {
        query.resize<float>({B, H, L1, S});
        present_key[n].resize<uint8_t>({B, H, (L0 + L1) * 1, S});
        present_value[n].resize<uint8_t>({B, H, (L0 + L1) * 1, S});
        output_emb[n].resize<float>({B, H, L1, S});
        past_k_scale_zp[n].resize<float>({B, H, (L0 + L1) * 1, 2});
        past_v_scale_zp[n].resize<float>({B, H, (L0 + L1) * 1, 2});
        present_key[n].m_dims[2] = L0 + L1;
        present_value[n].m_dims[2] = L0 + L1;
        past_k_scale_zp[n].m_dims[2] = L0 + L1;
        past_v_scale_zp[n].m_dims[2] = L0 + L1;
        if (touch) {
            query = 1.0f;
            present_key[n] = uint8_t{1};
            present_value[n] = uint8_t{1};
            past_k_scale_zp[n] = 1.0f;
            past_v_scale_zp[n] = 1.0f;
        }
        mha_single_token(query,
            present_key[n],
            present_value[n],
            alibi_mask,
            attention_mask,
            beams,
            output_emb[n],
            buf_attn_w,
            buf_attn_score,
            has_out_transpose,
            auto_causal,
            d_scale,
            past_k_scale_zp[n],
            past_v_scale_zp[n],
            head_sum,
            1);
    }
}

void test(size_t count, int blocks) {
    for (size_t i = 0; i < count; i++){
        for (size_t n = 0; n < N; n++) {
            mha_single_token(query,
                present_key[n],
                present_value[n],
                alibi_mask,
                attention_mask,
                beams,
                output_emb[n],
                buf_attn_w,
                buf_attn_score,
                has_out_transpose,
                auto_causal,
                d_scale,
                past_k_scale_zp[n],
                past_v_scale_zp[n],
                head_sum,
                blocks);
        }
    }
}

/*
  source ~/intel/oneapi/setvars.sh
  icpx test.cpp -o2 -qopenmp -lstdc++ -lm -mavx2 -mfma -o test
  ocperf stat -e mem_load_retired.fb_hit,mem_load_retired.l2_miss,mem_load_retired.l2_hit -- numactl -C0,2,4,6,8,10,12,14 ./test
  ocperf stat -e unc_m_prefetch_rd -- numactl -C0,2,4,6,8,10,12,14 ./test
  ocperf stat -e cycles,OFFCORE_REQUESTS.DATA_RD,OFFCORE_REQUESTS.DEMAND_DATA_RD,OFFCORE_REQUESTS_OUTSTANDING.CYCLES_WITH_DATA_RD,OFFCORE_REQUESTS_OUTSTANDING.CYCLES_WITH_DEMAND_DATA_RD -- numactl -C0,2,4,6,8,10,12,14 ./test
  gcc test.cpp -O2 -fopenmp -lstdc++ -lm -mavx2 -mfma -g -o test && numactl -C0,2,4,6,8,10,12,14 ./test 1
*/
int main(int argc, char* argv[]) {
    g_nthr = parallel_get_max_threads();

    setlocale(LC_NUMERIC, "");

    init();
    g_exec.reset();
    g_hit_cache = true;
    g_is_test = true;
    int count = 1000;
    auto start = std::chrono::steady_clock::now();
    //for (int k = 1; k < 128; k++)
    {
        test(count, -1);
    }
    auto end = std::chrono::steady_clock::now();
    float c = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    size_t call_dot_product_cout_per_thread = B * H * (L0 + L1) * N / g_nthr;
    printf("[no miss] cost %'.3f ms\n", c / count);
    printf("[no miss] dot product(all) cost %'ld ns, cost %'ld cycles, count %'ld, dot_product(kernel) uses %.2lf cycles\n", g_exec.ns, g_exec.cycles, g_exec.count, double(g_exec.cycles) / call_dot_product_cout_per_thread / count);

    g_is_test = false;
    init(true);
    g_exec.reset();
    g_hit_cache = false;
    g_is_test = true;
    start = std::chrono::steady_clock::now();
    //for (int k = 1; k < 128; k++)
    {
        test(count, -1);
    }
    end = std::chrono::steady_clock::now();
    c = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("[miss] cost %'.3f ms\n", c / count);
    printf("[miss] dot product(all) cost %'ld ns, cost %'ld cycles, count %'ld, dot_product(kernel) uses %.2lf cycles\n", g_exec.ns, g_exec.cycles, g_exec.count, double(g_exec.cycles) / call_dot_product_cout_per_thread / count);
    size_t size_pastk = count * B * H * (L0 + L1) * S * N;
    size_t size_zp = count * B * H * (L0 + L1) * 2 * sizeof(float) * N;
    printf("bw(pastk) = %.3lf GB/s\n", (double)size_pastk / ((double)g_exec.ns));
    printf("bw(pastk+zp) = %.3lf GB/s\n", (double)(size_pastk + size_zp) / ((double)g_exec.ns));

    return 0;
}
