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
#ifndef NIGH_IMPL_KDTREE_BATCH_NEAREST_TRAVERSAL_CARTESIAN_HPP
#define NIGH_IMPL_KDTREE_BATCH_NEAREST_TRAVERSAL_CARTESIAN_HPP

#include "nearest_traversal.hpp"
#include "traversal_cartesian.hpp"
#include "types.hpp"

namespace unc::robotics::nigh::impl::kdtree_batch {
    template <std::size_t I, typename Tree, typename Key, typename Metric, typename Get>
    using cartesian_nearest_traversal_element_t = NearestTraversal<
        Tree,
        typename metric::cartesian_state_element<I, Key>::type,
        metric::cartesian_element_t<I, Metric>,
        CartesianGet<Get, I>>;

    template <typename Tree, typename Key, typename Metric, typename Get, typename Indices>
    class CartesianNearestTraversal;

    template <typename Tree, typename Key, typename Metric, typename Get, std::size_t ... I>
    class CartesianNearestTraversal<Tree, Key, Metric, Get, std::index_sequence<I...>> {
        using Concurrency = concurrency_t<Tree>;
        using Node = node_t<Tree>;
        using Space = metric::Space<Key, Metric>;
        static constexpr std::size_t N = sizeof...(I);

        // TODO: Key needs to be cartesian element
        std::tuple<cartesian_nearest_traversal_element_t<I, Tree, Key, Metric, Get>...> tuple_;

    public:
        explicit CartesianNearestTraversal(const Space& space)
            : tuple_(cartesian_nearest_traversal_element_t<I, Tree, Key, Metric, Get>(
                         space.template get<I>())...)
        {
        }

        auto distToRegion(const Key& key, const Region<Key, Metric, Concurrency>& region) const {
            return (std::get<I>(tuple_).distToRegion(
                        metric::cartesian_state_element<I, Key>::get(key),
                        region.template get<I>()) + ...);
        }

        template <typename Visitor>
        void traverse(Visitor& visitor, const Space& space, const Node *node, unsigned axis, const Key& key) {
            CartesianHelper<0, N, Tree, Key, Metric, Get>::traverse(
                tuple_, visitor, space, node, axis, key);
        }
    };

    template <typename Tree, typename Key, typename ... M, typename Get>
    class NearestTraversal<Tree, Key, metric::Cartesian<M...>, Get>
        : public CartesianNearestTraversal<Tree, Key, metric::Cartesian<M...>, Get, std::index_sequence_for<M...>>
    {
        using Base = CartesianNearestTraversal<Tree, Key, metric::Cartesian<M...>, Get, std::index_sequence_for<M...>>;

    public:
        using Base::Base;
    };
}


#endif
