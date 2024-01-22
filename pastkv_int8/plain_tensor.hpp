// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdlib.h>
#include <cassert>
#include <climits>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#include <cstdlib>
#endif

#define PLAINTENSOR_RANK_MAX 8

struct PlainTensor {
    size_t m_strides[PLAINTENSOR_RANK_MAX];
    size_t m_dims[PLAINTENSOR_RANK_MAX];
    size_t m_rank = 0;
    std::shared_ptr<uint8_t> m_ptr;
    size_t m_capacity = 0;
    size_t m_element_size = 0;

    operator bool() const {
        return m_ptr != nullptr;
    }

    size_t size(int i) const {
        if (i < 0)
            i += m_rank;
        assert(static_cast<typename std::make_unsigned<decltype(i)>::type>(i) < m_rank);
        return m_dims[i];
    }
    size_t stride(int i) const {
        assert(i >= 0 && static_cast<typename std::make_unsigned<decltype(i)>::type>(i) < m_rank);
        return m_strides[i];
    }

    template<typename T>
    std::vector<T> get_strides() const {
        std::vector<T> strides(m_rank);
        for (size_t i = 0; i < m_rank; i++)
            strides[i] = static_cast<T>(m_strides[i]);
        return strides;
    }


    PlainTensor() = default;

    PlainTensor operator=(const PlainTensor& other) {
        memcpy(&m_strides, &other.m_strides, sizeof(m_strides));
        memcpy(&m_dims, &other.m_dims, sizeof(m_dims));
        m_rank = other.m_rank;
        m_ptr = other.m_ptr;
        m_element_size = other.m_element_size;
        m_capacity = other.m_capacity;
        return *this;
    }

    struct tensor_index {
        int start;
        int end;
        int step;
        int count;
        // select all
        tensor_index() {
            start = 0;
            end = INT_MAX;
            step = 1;
        }
        bool slice_with_squeeze() {
            return end == INT_MIN;
        }
        // tensor_index(start)            : select 1 element (with squeeze)
        // tensor_index(start, end, step) : select a range w/o squeeze
        tensor_index(int start, int end = INT_MIN, int step = 1) : start(start), end(end), step(step) {}

        void regularize(int size) {
            if (start < 0)
                start += size;
            assert(start >= 0 && start < size);
            if (end != INT_MIN) {
                if (end < 0)
                    end += size;
                if (end > size)
                    end = size;
                assert(end >= 0 && end <= size);
                count = (end - start + step - 1) / step;
            } else {
                count = 1;
            }
        }
    };

    PlainTensor index(const std::initializer_list<tensor_index>& indices) {
        PlainTensor sub_tensor;
        assert(indices.size() <= m_rank);
        int i_src = 0;
        int i_dst = 0;
        sub_tensor.m_capacity = 0;
        size_t off = 0;
        for (auto idx : indices) {
            auto src_dim = m_dims[i_src];
            auto src_stride = m_strides[i_src];
            idx.regularize(src_dim);
            off += idx.start * src_stride;
            if (idx.slice_with_squeeze()) {
                // no output dimension
                i_src++;
                continue;
            }
            sub_tensor.m_dims[i_dst] = idx.count;
            sub_tensor.m_strides[i_dst] = src_stride;
            i_dst++;
            i_src++;
        }
        sub_tensor.m_rank = i_dst;  // index may imply squeeze
        sub_tensor.m_ptr = std::shared_ptr<uint8_t>((m_ptr.get() + off * m_element_size), [](uint8_t*) {});
        sub_tensor.m_element_size = m_element_size;
        return sub_tensor;
    }

    // slice: return a sub-view (w/o ownership/refcount to original data)
    PlainTensor slice(int axis, int start, int end, int step = 1) const {
        PlainTensor sub_tensor;
        assert(axis >= 0 && static_cast<typename std::make_unsigned<decltype(axis)>::type>(axis) < m_rank);

        sub_tensor.m_capacity = 0;
        if (end > start) {
            sub_tensor.m_rank = m_rank;
            for (size_t i = 0; i < m_rank; i++) {
                sub_tensor.m_strides[i] = m_strides[i];
                sub_tensor.m_dims[i] = m_dims[i];
            }
            sub_tensor.m_dims[axis] = (end - start - 1) / step + 1;
        } else {
            // squeeze if end == start
            sub_tensor.m_rank = m_rank - 1;
            size_t k = 0;
            for (size_t i = 0; i < m_rank; i++) {
                if (i != static_cast<size_t>(axis)) {
                    sub_tensor.m_strides[k] = m_strides[i];
                    sub_tensor.m_dims[k] = m_dims[i];
                    k++;
                }
            }
        }

        auto off = start * m_strides[axis];
        sub_tensor.m_ptr = std::shared_ptr<uint8_t>(m_ptr.get() + off * m_element_size, [] (uint8_t* ) {});
        sub_tensor.m_element_size = m_element_size;

        return sub_tensor;
    }

    bool is_dense() const {
        // check if it's dense tensor
        size_t stride = 1;
        for (int i = m_rank - 1; i >= 0; i--) {
            if (m_strides[i] != stride)
                return false;
            stride *= m_dims[i];
        }
        return true;
    }

    /*
       suppose current shape is [a0,a1,...,am]
       and target shape is [b0,b1,...,bn]
       reshape is only valid when (a0*a1*...*am) == (b0*b1*...*bn) <======= (A)

       uniform a tensor's shape into groups from last to first, the dimension is merged
       into current group if the subtensor in the group is still dense after merge.
       otherwise a new group is formed.

       then reshape is performed on group basis, the check (A) is performed on group bases.
       which means any reshape inside the group is OK, but not across the group boundary.

       this can be done in one-loop, while group is forming, and checks are performed.

       simplified form is when whole tensor is dense
    */
    PlainTensor reshape(const std::vector<size_t>& target_shape) const {
        // only valid for dense memory
        PlainTensor new_tensor_view;
        assert(is_dense());
        new_tensor_view.resize(target_shape, m_element_size, static_cast<void*>(m_ptr.get()));
        return new_tensor_view;
    }

    PlainTensor permute(const std::vector<size_t>& order) const {
        PlainTensor new_tensor_view;
        assert(order.size() == m_rank);
        new_tensor_view.m_capacity = 0;
        // not hold memory reference
        new_tensor_view.m_ptr = std::shared_ptr<uint8_t>(m_ptr.get(), [] (uint8_t* ) {});;
        new_tensor_view.m_rank = m_rank;
        new_tensor_view.m_element_size = m_element_size;
        auto it_order = order.begin();
        // also should check order has no repeat element
        for (size_t i = 0; i < m_rank; i++) {
            auto j = *it_order++;
            assert(j >= 0 && j < m_rank);
            new_tensor_view.m_dims[i] = m_dims[j];
            new_tensor_view.m_strides[i] = m_strides[j];
        }
        return new_tensor_view;
    }

    void resize(const std::vector<size_t>& new_dims, size_t element_size, void* data = nullptr, const size_t* strides = nullptr) {
        m_element_size = element_size;
        // initialize strides for compact/dense tensor
        m_rank = new_dims.size();
        assert(m_rank <= PLAINTENSOR_RANK_MAX);
        size_t stride = 1;
        for (int i = m_rank - 1; i >= 0; i--) {
            m_dims[i] = new_dims[i];
            m_strides[i] = strides ? strides[i] : stride;
            stride *= new_dims[i];
        }

        if (!data) {
            auto capacity_new = m_strides[0] * m_dims[0] * m_element_size;
            if (capacity_new > m_capacity) {
                void* ptr;
                #ifdef _WIN32
                    ptr = _aligned_malloc(capacity_new, 64);
                #else
                    int rc = ::posix_memalign(&ptr, 4096, capacity_new);
                    if (rc) ptr = nullptr;
                #endif
                m_ptr = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(ptr), [](uint8_t* ptr) {
                    #ifdef _WIN32
                        _aligned_free(ptr);
                    #else
                        ::free(ptr);
                    #endif
                });
                m_capacity = capacity_new;
            }
        } else {
            // m_capacity is zero to indicate that we don't own the memory
            m_capacity = 0;
            m_ptr = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(data), [](uint8_t*) {});
        }
    }

    template<typename DT>
    void resize(const std::initializer_list<size_t>& new_dims, DT* data = nullptr, const size_t* strides = nullptr) {
        resize(new_dims, sizeof(DT), data, strides);
    }

    template <typename DT>
    DT* data() const {
        return reinterpret_cast<DT*>(m_ptr.get());
    }

    // when allow_broadcast is true, index to size-1 dim will always access 0.
    template <typename DT>
    DT& at(const std::initializer_list<size_t>& index, bool allow_broadcast = false) const {
        size_t off = 0;
        auto it = index.begin();
        for (size_t i = 0; i < m_rank; i++) {
            size_t coordinate = (it != index.end()) ? (*it++) : 0;
            if (allow_broadcast && m_dims[i] == 1) {
                // allow_broadcast only works when the dimension is really 1
                coordinate = 0;
            } else {
                assert(coordinate < m_dims[i]);
            }
            off += m_strides[i] * coordinate;
        }
        return (reinterpret_cast<DT*>(m_ptr.get() + off * m_element_size))[0];
    }

    // the following is used for fast access
    template <typename DT>
    DT& at(size_t dim0, bool last_dim_stride_is_one = true) const {
        size_t off;
        if (last_dim_stride_is_one)
            off = dim0;
        else
            off = m_strides[0] * dim0;
        return (reinterpret_cast<DT*>(m_ptr.get() + off * m_element_size))[0];
    }
    template <typename DT>
    DT& at(size_t dim0, size_t dim1, bool last_dim_stride_is_one = true) const {
        size_t off;
        if (last_dim_stride_is_one)
            off = m_strides[0] * dim0 + dim1;
        else
            off = m_strides[0] * dim0 + m_strides[1] * dim1;
        return (reinterpret_cast<DT*>(m_ptr.get() + off * m_element_size))[0];
    }
    template <typename DT>
    DT& at(size_t dim0, size_t dim1, size_t dim2, bool last_dim_stride_is_one = true) const {
        size_t off;
        if (last_dim_stride_is_one)
            off = m_strides[0] * dim0 + m_strides[1] * dim1 + dim2;
        else
            off = m_strides[0] * dim0 + m_strides[1] * dim1 + dim2 * m_strides[2];
        return (reinterpret_cast<DT*>(m_ptr.get() + off * m_element_size))[0];
    }
    template <typename DT>
    DT& at(size_t dim0, size_t dim1, size_t dim2, size_t dim3, bool last_dim_stride_is_one = true) const {
        size_t off;
        if (last_dim_stride_is_one)
            off = m_strides[0] * dim0 + m_strides[1] * dim1 + dim2 * m_strides[2] + dim3;
        else
            off = m_strides[0] * dim0 + m_strides[1] * dim1 + dim2 * m_strides[2] + dim3 * m_strides[3];
        return (reinterpret_cast<DT*>(m_ptr.get() + off * m_element_size))[0];
    }
    template <typename DT>
    DT& at(size_t dim0, size_t dim1, size_t dim2, size_t dim3, size_t dim4, bool last_dim_stride_is_one = true) const {
        size_t off;
        if (last_dim_stride_is_one)
            off = m_strides[0] * dim0 + m_strides[1] * dim1 + dim2 * m_strides[2] + dim3 * m_strides[3] + dim4;
        else
            off = m_strides[0] * dim0 + m_strides[1] * dim1 + dim2 * m_strides[2] + dim3 * m_strides[3] + dim4 * m_strides[4];
        return (reinterpret_cast<DT*>(m_ptr.get() + off * m_element_size))[0];
    }

    template <typename DT>
    PlainTensor& operator=(const DT& value) {
        // assign every element to value
        std::vector<size_t> index(m_rank, 0);
        auto* dst = reinterpret_cast<DT*>(m_ptr.get());
        while (1) {
            size_t off = 0;
            for (int i = m_rank - 1; i >= 0; i--) {
                if (index[i] >= m_dims[i]) {
                    // carry on
                    if (i == 0)
                        return *this;
                    index[i] = 0;
                    index[i - 1]++;
                }
                off += m_strides[i] * index[i];
            }
            dst[off] = value;
            // move to next coordinate
            index[m_rank - 1]++;
        }
        return *this;
    }

    template <typename DT>
    DT& operator()(const std::initializer_list<size_t>& index, bool allow_broadcast = false) const {
        return at<DT>(index, allow_broadcast);
    }

    friend std::ostream& operator<<(std::ostream& os, const PlainTensor& dt);
};
