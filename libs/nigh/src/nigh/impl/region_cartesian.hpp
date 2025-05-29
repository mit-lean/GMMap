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
#ifndef NIGH_IMPL_REGION_CARTESIAN_HPP
#define NIGH_IMPL_REGION_CARTESIAN_HPP

#include "region.hpp"
#include "../metric/cartesian_state_element.hpp"

namespace unc::robotics::nigh::impl {

    template <std::size_t I, typename Key, typename Metric, typename Concurrency>
    using cartesian_region_element_t = Region<
        typename metric::cartesian_state_element<I, Key>::type,
        metric::cartesian_element_t<I, Metric>,
        Concurrency>;

    template <typename Key, typename M, typename Concurrency, typename Indices>
    class CartesianRegion;

    template <typename Key, typename Metric, typename Concurrency, std::size_t ... I>
    class CartesianRegion<Key, Metric, Concurrency, std::index_sequence<I...>>
        : public std::tuple<cartesian_region_element_t<I, Key, Metric, Concurrency>...>
    {
        using Space = metric::Space<Key, Metric>;
        using Distance = typename Space::Distance;

        using Base = std::tuple<cartesian_region_element_t<I, Key, Metric, Concurrency>...>;

    public:
        template <typename Traversal>
        CartesianRegion(const Space& space, const Traversal& traversal, const Key& key)
            : Base(
                (cartesian_region_element_t<I, Key, Metric, Concurrency>(
                    space.template get<I>(),
                    traversal.template get<I>(),
                    metric::cartesian_state_element<I, Key>::get(key))) ...)
        {
        }

        unsigned dimensions() const {
            return (std::get<I>(*this).dimensions() + ...);
        }

        template <std::size_t J>
        decltype(auto) get() { return std::get<J>(*this); }
        template <std::size_t J>
        decltype(auto) get() const { return std::get<J>(*this); }
    };


    template <typename Key, typename ... M, typename Concurrency>
    class Region<Key, metric::Cartesian<M...>, Concurrency>
        : public CartesianRegion<Key, metric::Cartesian<M...>, Concurrency, std::index_sequence_for<M...>>
    {
        using Base = CartesianRegion<Key, metric::Cartesian<M...>, Concurrency, std::index_sequence_for<M...>>;
    public:
        using Base::Base;
    };
}

#endif
