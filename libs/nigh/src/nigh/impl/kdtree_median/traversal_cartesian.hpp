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
#ifndef NIGH_IMPL_KDTREE_MEDIAN_TRAVERSAL_CARTESIAN_HPP
#define NIGH_IMPL_KDTREE_MEDIAN_TRAVERSAL_CARTESIAN_HPP

#include "traversal.hpp"
#include "node.hpp"
#include "../cartesian_get.hpp"

namespace unc::robotics::nigh::impl::kdtree_median {
    template <std::size_t I, class Tree, class Key, class Metric, class Get>
    using cartesian_traversal_element_t = Traversal<
        Tree,
        typename metric::cartesian_state_element<I, Key>::type,
        metric::cartesian_element_t<I, Metric>,
        CartesianGet<Get, I>>;

    template <class Tree, class Key, class Metric, class Get, class Indices>
    struct cartesian_traversal_tuple;

    template <class Tree, class Key, class Metric, class Get, std::size_t ... I>
    struct cartesian_traversal_tuple<Tree, Key, Metric, Get, std::index_sequence<I...>> {
        using type = std::tuple<cartesian_traversal_element_t<I, Tree, Key, Metric, Get>...>;
    };

    template <class Tree, class Key, class Metric, class Get, class Indices>
    using cartesian_traversal_tuple_t =
        typename cartesian_traversal_tuple<Tree, Key, Metric, Get, Indices>::type;


    template <std::size_t I, std::size_t N, class Tree, class Key, class Metric, class Get>
    struct CartesianTraversalHelper {
        using Space = metric::Space<Key, Metric>;
        using Distance = typename Space::Distance;
        using Indices = std::make_index_sequence<N>;
        using Tuple = cartesian_traversal_tuple_t<Tree, Key, Metric, Get, Indices>;

        template <class Nearest, class Iter>
        static void follow(
            Tuple& tuple,
            Nearest& nearest, const Space& space,
            const Node *node, unsigned axis, const Key& key,
            Iter first, Iter last)
        {
            if constexpr (I+1 < N) {
                unsigned dim = space.template get<I>().dimensions();
                if (axis >= dim) {
                    CartesianTraversalHelper<I+1, N, Tree, Key, Metric, Get>::follow(
                        tuple, nearest, space, node, axis - dim, key, first, last);
                    return;
                }
            }

            std::get<I>(tuple).follow(
                nearest,
                space.template get<I>(),
                node, axis,
                metric::cartesian_state_element<I, Key>::get(key),
                first, last);
        }
    };
    
    template <class Tree, class Key, class ... M, class Get>
    class Traversal<Tree, Key, metric::Cartesian<M...>, Get> {
        using Metric = metric::Cartesian<M...>;
        using Space = metric::Space<Key, Metric>;
        using Distance = typename Space::Distance;
        
        static constexpr std::size_t N = sizeof...(M);
        using Indices = std::make_index_sequence<N>;

        cartesian_traversal_tuple_t<Tree, Key, Metric, Get, Indices> tuple_;

        template <std::size_t ... I>
        Traversal(const Space& space, std::index_sequence<I...>)
            : tuple_(cartesian_traversal_element_t<I, Tree, Key, Metric, Get>(
                         space.template get<I>())...)
        {            
        }

        template <std::size_t ... I>
        Distance distToRegion(std::index_sequence<I...>) const {
            return (std::get<I>(tuple_).distToRegion() + ...);
        }
        
    public:
        Traversal(const Space& space)
            : Traversal(space, Indices{})
        {
        }

        Distance distToRegion() const {
            return distToRegion(Indices{});
        }

        template <class Nearest, class Iter>
        void follow(
            Nearest& nearest,
            const Space& space,
            const Node *node, unsigned axis,
            const Key& key,
            Iter first, Iter last)
        {
            CartesianTraversalHelper<0, N, Tree, Key, Metric, Get>::follow(
                tuple_, nearest, space, node, axis, key, first, last);
        }
    };
}

#endif
