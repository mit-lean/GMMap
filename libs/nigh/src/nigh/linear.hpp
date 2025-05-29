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
#ifndef NIGH_LINEAR_HPP
#define NIGH_LINEAR_HPP

#include "nigh_forward.hpp"
#include "metric/space.hpp"
#include "impl/nearest_base.hpp"
#include "impl/near_set.hpp"
#include "impl/atom.hpp"
#include "impl/linear/store.hpp"

namespace unc::robotics::nigh {

    struct Linear {};

    template <
        typename T,
        typename Metric,
        typename Member,
        typename Concurrency,
        typename Allocator>
    class Nigh<T, Metric, Member, Concurrency, Linear, Allocator>
        : public impl::NearestBase<
            Nigh<T, Metric, Member, Concurrency, Linear, Allocator>>
    {
        // TODO: we're storing allocator twice, once in Base, and once in Store.
        using Base = impl::NearestBase<Nigh<T, Metric, Member, Concurrency, Linear, Allocator>>;
        using Key = typename Base::Key;
        using Distance = typename Base::Distance;

        impl::linear::Store<T, Concurrency, Allocator> store_;

    public:
        Nigh(const Nigh&) = delete;
        Nigh(Nigh&& other)
            : store_(std::move(other.store_))
        {
        }

        Nigh(const Metric& metric = Metric(), const Member& member = Member(), const Allocator& alloc = Allocator())
            : Base(metric, member, alloc)
            , store_(alloc)
        {
        }

        std::size_t size(std::memory_order order = std::memory_order_relaxed) const {
            return store_.size(order);
        }

        std::size_t depth() const {
            return size();
        }

        void clear() {
            store_.clear();
        }

        void insert(const T& value) {
            store_.insert(value);
        }

        template <typename K>
        std::optional<std::pair<T, Distance>> nearest(const K& key) const {
            impl::Near1Set<T, Distance> nearSet;
            store_.nearest(nearSet, [&] (const T& t) { return Base::distToKey(t, key); });
            return nearSet.result();
        }

        template <typename K>
        std::optional<T> nearest(const K& key, Distance* dist) const {
            impl::Near1Set<T, Distance> nearSet;
            store_.nearest(nearSet, [&] (const T& t) { return Base::distToKey(t, key); });
            return nearSet.result(dist);
        }

        template <typename Tuple, typename K, typename ResultAllocator>
        void nearest(
            std::vector<Tuple, ResultAllocator>& nbh,
            const K& key,
            std::size_t k,
            Distance maxRadius = std::numeric_limits<Distance>::infinity()) const
        {
            impl::NearKSet<Tuple, Distance, ResultAllocator> nearSet(nbh, k, maxRadius);
            store_.nearest(nearSet, [&] (const T& t) { return Base::distToKey(t, key); });
            nearSet.sort();
        }

        template <typename Fn>
        void visit(const Fn& fn) const {
            store_.visit(fn);
        }

        std::vector<T> list() const {
            return store_.list();
        }
    };
}

#endif
