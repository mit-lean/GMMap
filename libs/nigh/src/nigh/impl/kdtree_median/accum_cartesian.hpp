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
#ifndef NIGH_IMPL_KDTREE_MEDIAN_ACCUM_CARTESIAN_HPP
#define NIGH_IMPL_KDTREE_MEDIAN_ACCUM_CARTESIAN_HPP

#include "accum.hpp"
#include "node.hpp"
#include "../cartesian_get.hpp"

namespace unc::robotics::nigh::impl::kdtree_median {

    template <std::size_t I, typename Key, typename Metric, typename Get>
    using cartesian_accum_element_t = Accum<
        typename metric::cartesian_state_element<I, Key>::type,
        metric::cartesian_element_t<I, Metric>,
        CartesianGet<Get, I>>;

    template <class Key, class Metric, class Get, class Indices>
    struct cartesian_accum_tuple;

    template <class Key, class Metric, class Get, std::size_t ... I>
    struct cartesian_accum_tuple<Key, Metric, Get, std::index_sequence<I...>> {
        using type = std::tuple<cartesian_accum_element_t<I, Key, Metric, Get>...>;
    };

    template <class Key, class Metric, class Get, class Indices>
    using cartesian_accum_tuple_t =
        typename cartesian_accum_tuple<Key, Metric, Get, Indices>::type;

    template <std::size_t I, std::size_t N, class Key, class Metric, class Get>
    struct CartesianAccumHelper {
        using Space = metric::Space<Key, Metric>;
        using Distance = typename Space::Distance;
        using Indices = std::make_index_sequence<N>;
        using Tuple = cartesian_accum_tuple_t<Key, Metric, Get, Indices>;

        using Next = CartesianAccumHelper<I+1, N, Key, Metric, Get>;
        
        template <class Builder, class Iter>
        static Distance selectAxis(
            Tuple& tuple, unsigned axisZero, Distance dBest,
            Builder& builder, const Space& space, unsigned *axis, Iter first, Iter last)
        {
            unsigned tmp;
            Distance d = std::get<I>(tuple).selectAxis(
                builder, space.template get<I>(), &tmp, first, last);
            
            if (I == 0 || d > dBest) {
                dBest = d;
                *axis = tmp + axisZero;
            }
            unsigned dim = space.template get<I>().dimensions();
            return Next::selectAxis(
                tuple, axisZero + dim, dBest,
                builder, space, axis, first, last);
        }
        
        template <class Builder, class Iter>
        static Node* partition(
            Tuple& tuple, unsigned axisZero,
            Builder& builder, const Space& space, unsigned axis, Iter first, Iter last)
        {
            if constexpr (I+1 < N) {
                unsigned dim = space.template get<I>().dimensions();
                if (axis >= dim)
                    return Next::partition(
                        tuple, axisZero + dim, builder, space, axis - dim, first, last);
            }
            
            Node *n = std::get<I>(tuple).partition(
                builder, space.template get<I>(), axis, first, last);
            n->updateAxis(axisZero);
            return n;
        }
    };

    template <std::size_t N, class Key, class Metric, class Get>
    struct CartesianAccumHelper<N, N, Key, Metric, Get> {
        using Space = metric::Space<Key, Metric>;
        using Distance = typename Space::Distance;
        using Indices = std::make_index_sequence<N>;
        using Tuple = cartesian_accum_tuple_t<Key, Metric, Get, Indices>;

        template <class Builder, class Iter>
        static Distance selectAxis(
            Tuple& tuple, unsigned axisZero, Distance dBest,
            Builder& builder, const Space& space, unsigned *axis, Iter first, Iter last)
        {
            return dBest;
        }
    };
    
    template <typename Key, typename ... M, typename Get>
    class Accum<Key, metric::Cartesian<M...>, Get> {
        using Metric = metric::Cartesian<M...>;
        using Space = metric::Space<Key, Metric>;
        using Distance = typename Space::Distance;
        static constexpr std::size_t N = sizeof...(M);
        using Indices = std::make_index_sequence<N>;

        cartesian_accum_tuple_t<Key, Metric, Get, Indices> tuple_;

    public:        
        template <typename Builder, typename Iter>
        Distance selectAxis(
            Builder& builder, const Space& space, unsigned *axis, Iter first, Iter last)
        {
            return CartesianAccumHelper<0, N, Key, Metric, Get>::selectAxis(
                tuple_, 0u, Distance(0), builder, space, axis, first, last);
        }
        
        template <typename Builder, typename Iter>
        Node *partition(
            Builder& builder, const Space& space, unsigned axis, Iter first, Iter last)
        {
            return CartesianAccumHelper<0, N, Key, Metric, Get>::partition(
                tuple_, 0u, builder, space, axis, first, last);
        }
    };
}

#endif
