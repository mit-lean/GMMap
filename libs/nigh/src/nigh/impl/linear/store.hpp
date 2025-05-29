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
#ifndef NIGH_IMPL_LINEAR_STORE_HPP
#define NIGH_IMPL_LINEAR_STORE_HPP

#include <atomic>
#include <memory>
#include <vector>
#include "../atom.hpp"

namespace unc::robotics::nigh::impl::linear {
    template <typename T>
    struct Link {
        Link *next_;
        T value_;
        Link(const T& value, Link *next) : next_(next), value_(value) {}
    };

    template <typename T, typename Concurrency, typename Allocator>
    class Store;

    template <typename T, typename Allocator>
    class Store<T, Concurrent, Allocator>
        : std::allocator_traits<Allocator>::template rebind_alloc<Link<T>>
    {
        using LinkAllocator = typename std::allocator_traits<Allocator>::template rebind_alloc<Link<T>>;

        alignas(cache_line_size)
        std::atomic<std::size_t> size_{0};

        alignas(cache_line_size)
        std::atomic<Link<T>*> list_{nullptr};

    public:
        Store(Store&& other)
            : size_(other.size_.exchange(0))
            , list_(other.list_.exchange(nullptr))
        {
        }

        Store(const Allocator& allocator)
            : LinkAllocator(allocator)
        {
        }

        ~Store() {
            clear();
        }

        std::size_t size(std::memory_order order) const {
            return size_.load(order);
        }

        void clear() {
            Link<T> *n = list_.exchange(nullptr, std::memory_order_acquire);
            size_.store(std::size_t(0), std::memory_order_release);
            for (Link<T> *next ; n ; n = next) {
                next = n->next_;
                std::allocator_traits<LinkAllocator>::destroy(*this, n);
                std::allocator_traits<LinkAllocator>::deallocate(*this, n, 1);
                // delete n;
            }
        }

        void insert(const T& value) {
            Link<T> *n = std::allocator_traits<LinkAllocator>::allocate(*this, 1);
            std::allocator_traits<LinkAllocator>::construct(
                *this, n, value, list_.load(std::memory_order_relaxed));

            // Link *n = new Link(value, list_.load(std::memory_order_relaxed));
            while (!list_.compare_exchange_weak(
                       n->next_, n, std::memory_order_release, std::memory_order_relaxed))
                ;
            size_.fetch_add(std::size_t(1), std::memory_order_relaxed);
        }

        template <typename NearSet, typename Fn>
        void nearest(NearSet& nearSet, Fn dist) const {
            Link<T> *tail = nullptr;
            for (Link<T> *head ; (head = list_.load(std::memory_order_acquire)) != tail ; tail = head) {
                for (Link<T> *n = head ; n != tail ; n = n->next_)
                    nearSet.insert(n->value_, dist(n->value_));
            }
        }

        template <typename Fn>
        void visit(const Fn& fn) const {
            Link<T> *tail = nullptr;
            for (Link<T> *head ; (head = list_.load(std::memory_order_acquire)) != tail ; tail = head) {
                for (Link<T> *n = head ; n != tail ; n = n->next_)
                    fn(n->value_);
            }
        }

        std::vector<T> list() const {
            std::vector<T> result;
            result.reserve(static_cast<std::size_t>(size(std::memory_order_acquire) * 1.1));
            visit([&] (const T& t) { result.push_back(t); });
            return result;
        }
    };

    template <typename T, typename Allocator>
    class Store<T, ConcurrentRead, Allocator> {
        std::vector<T> values_;

    public:
        Store(Store&& other)
            : values_(std::move(other.values_))
        {
        }

        Store(const Allocator& allocator)
            : values_(allocator)
        {
        }

        std::size_t size(std::memory_order order) const {
            return values_.size();
        }

        void clear() {
            values_.clear();
        }

        void insert(const T& value) {
            values_.push_back(value);
        }

        template <typename NearSet, typename Fn>
        void nearest(NearSet& nearSet, Fn dist) const {
            nearSet.insert(values_.begin(), values_.end(), dist);
        }

        template <typename Fn>
        void visit(const Fn& fn) const {
            std::for_each(values_.begin(), values_.end(), fn);
        }

        const std::vector<T>& list() const {
            return values_;
        }
    };

    template <typename T, typename Allocator>
    class Store<T, NoThreadSafety, Allocator>
        : public Store<T, ConcurrentRead, Allocator>
    {
        using Store<T, ConcurrentRead, Allocator>::Store;
        // TODO: we may be able to go faster with NoThreadSafety by
        // using std::partial_sort or std::nth_element to implement
        // k-nearest.
    };
}

#endif
