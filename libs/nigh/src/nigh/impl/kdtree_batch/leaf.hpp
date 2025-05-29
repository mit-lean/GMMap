// Software License Agreement (BSD-3-Clause)
//
// Copyright 2018 The University of North Carolina at Chapel Hill
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.

//! @author Jeff Ichnowski

#pragma once
#ifndef NIGH_IMPL_KDTREE_BATCH_LEAF_HPP
#define NIGH_IMPL_KDTREE_BATCH_LEAF_HPP

#include "../atom.hpp"
#include "node.hpp"

namespace unc::robotics::nigh::impl::kdtree_batch {
    template <typename T, typename Space, typename Concurrency, std::size_t batchSize>
    class Leaf : public Node<T, Space, Concurrency> {
        using Base = Node<T, Space, Concurrency>;
        using Key = typename Space::Type;
        using Metric = typename Space::Metric;
        // std::aligned_storage is being deprecated.  See
        // http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1413r2.pdf
        // using AlignedStorage = std::aligned_storage_t<sizeof(T), alignof(T)>;
        struct AlignedStorage {
            alignas(T) std::byte buf_[sizeof(T)];
        };

        static constexpr bool concurrentWrites = std::is_same_v<Concurrency, Concurrent>;

        alignas(concurrentWrites ? cache_line_size : sizeof(int))
        Atom<int, concurrentWrites> size_;

        alignas(concurrentWrites ? cache_line_size : alignof(T)) alignas(T)
        AlignedStorage elements_[batchSize];

    public:
        //Leaf(const Metric& metric, const T& t) : Base(metric, t), size_(1) {
        template <typename Traversal>
        Leaf(const Space& space, const Traversal& traversal, const T& q, const Key& key)
            : Base(space, traversal, key)
            , size_(1)
        {
            put(0, q);
        }

        template <typename Traversal, typename GetKey, typename Iter>
        Leaf(const Space& space, Traversal& traversal, const GetKey& getKey, Iter first, Iter last)
            : Base(space, traversal, getKey(**first))
        {
            int i=0;
            put(i++, **first);
            while (++first != last) {
                traversal.grow(space, this->region(), getKey(**first));
                put(i++, **first);
            }

            size_.store(i, std::memory_order_relaxed);
        }

        ~Leaf() {
            std::destroy(elements(), elements() + std::abs(size()));
        }

        T* elements() {
            return reinterpret_cast<T*>(elements_);
        }

        const T* elements() const {
            return reinterpret_cast<const T*>(elements_);
        }

        int size() const {
            return size_.load(std::memory_order_acquire);
        }

        bool tryLock(int n) {
            return size_.compare_exchange_weak(
                n, -n, std::memory_order_release, std::memory_order_relaxed);
        }

        template <typename ... Args>
        void put(int index, Args&& ... args) {
            new (elements() + index) T (std::forward<Args>(args)...);
        }

        void setSize(int n, std::memory_order order) {
            size_.store(n, order);
        }
    };
}

#endif
