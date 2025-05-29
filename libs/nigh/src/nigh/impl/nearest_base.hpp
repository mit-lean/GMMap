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
#ifndef NIGH_NEAREST_BASE_HPP
#define NIGH_NEAREST_BASE_HPP

#include "../nigh_forward.hpp"
#include <type_traits>
#include <memory>
#include <vector>

namespace unc::robotics::nigh::impl {

    // template <typename T, typename KeyFn>
    // struct member_type_t {
    //     using type =
    // };

    template <typename Derived>
    class NearestBase;

    template <
        typename T,
        typename Space_,
        typename KeyFn,
        typename Concurrency,
        typename Strategy,
        typename Allocator>
    class NearestBase<
        Nigh<T, Space_, KeyFn, Concurrency, Strategy, Allocator>>
    {
        using Derived = Nigh<T, Space_, KeyFn, Concurrency, Strategy, Allocator>;

        static_assert(
            std::is_same_v<Concurrency, NoThreadSafety> ||
            std::is_same_v<Concurrency, ConcurrentRead> ||
            std::is_same_v<Concurrency, Concurrent>,
            "invalid type of concurrency.  must be Concurrent, ConcurrentRead, NoThreadSafety");

        static_assert(
            metric::is_space<Space_>::value,
            "metric space is invalid");

        static_assert(
            std::is_invocable_v<const KeyFn&, const T&>,
            "KeyFn must be a functor that accepts 'const T&' as an argument");

    public:
        using Type = T;
        using Space = Space_;
        using Key = typename Space::Type;

        static_assert(
            std::is_invocable_r_v<Key, const KeyFn&, const T&>,
            "KeyFn(T) must return a value compatible with Key type");
        
        // make sure Key is compatible with SpaceSpace::Type
        // using Key = std::decay_t<std::result_of_t<KeyFn(T)>>;
        // Note: this is may need some work to handle copyable
        // static_assert(
        //     std::is_assignable_v<Key&, std::result_of_t<KeyFn(T)>>,
        //     "key function result must be assignable to metric space state type");

        using Metric = typename Space::Metric;
        using Distance = typename Space::Distance;

        using concurrency_type = Concurrency;

        // mirror std container apis
        using value_type = T;
        using allocator_type = Allocator;

    private:
        Space space_;
        KeyFn keyFn_;
        Allocator allocator_;

    protected:
        decltype(auto) getKey(const T& q) const {
            return keyFn_(q);
        }

        const KeyFn& keyFn() const {
            return keyFn_;
        }

        auto distance(const Key& a, const Key& b) const {
            return space_.distance(a, b);
        }

        auto distToKey(const T& t, const Key& key) const {
            return space_.distance(getKey(t), key);
        }

        template <typename Obj, typename ... Args>
        Obj* alloc(Args&& ... args) {
            using A = typename std::allocator_traits<Allocator>::template rebind_alloc<Obj>;
            A alloc(allocator_);
            Obj *ptr = std::allocator_traits<A>::allocate(alloc, 1);
            std::allocator_traits<A>::construct(alloc, ptr, std::forward<Args>(args)...);
            return ptr;
        }

        template <typename Obj, typename ... Args>
        Obj* allocWithSpace(Args&& ... args) {
            return alloc<Obj>(space_, std::forward<Args>(args)...);
        }

        template <typename Obj>
        void dealloc(Obj* ptr) {
            using A = typename std::allocator_traits<Allocator>::template rebind_alloc<Obj>;
            A alloc(allocator_);
            std::allocator_traits<A>::destroy(alloc, ptr);
            std::allocator_traits<A>::deallocate(alloc, ptr, 1);
        }

    public:
        NearestBase(
            const Space& metricSpace = Space(),
            const KeyFn& keyFn = KeyFn(),
            const Allocator& allocator = Allocator())
            : space_(metricSpace)
            , keyFn_(keyFn)
            , allocator_(allocator)
        {
        }

        allocator_type get_allocator() const {
            return allocator_;
        }

        const Space& metricSpace() const {
            return space_;
        }

        template <typename Key>
        auto nearest(
            const Key& q,
            std::size_t k,
            Distance maxRadius = std::numeric_limits<Distance>::infinity()) const
        {
            std::vector<std::pair<T, Distance>> results;
            results.reserve(k+1);
            static_cast<const Derived*>(this)->nearest(results, q, k, maxRadius);
            return results;
        }
    };
}

#endif
