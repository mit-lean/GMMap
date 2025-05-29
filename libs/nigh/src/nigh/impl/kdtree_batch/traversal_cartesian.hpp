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
#ifndef NIGH_IMPL_KDTREE_BATCH_TRAVERSAL_CARTESIAN_HPP
#define NIGH_IMPL_KDTREE_BATCH_TRAVERSAL_CARTESIAN_HPP

#include "traversal.hpp"
#include "types.hpp"
#include "../cartesian_get.hpp"

namespace unc::robotics::nigh::impl::kdtree_batch {

    template <std::size_t I, std::size_t N, typename Tree, typename Key, typename Metric, typename Get>
    struct CartesianHelper {
        using Node = node_t<Tree>;
        using Leaf = leaf_t<Tree>;
        using Distance = distance_t<Tree>;
        using NodePointer = node_pointer_t<Tree>;
        using Space = metric::Space<Key, Metric>;
        using KeyPart = typename metric::cartesian_state_element<I, Key>::type;
        using MetricPart = metric::cartesian_element_t<I, Metric>;
        using SpacePart = metric::Space<KeyPart, MetricPart>;

        using Next = CartesianHelper<I+1, N, Tree, Key, Metric, Get>;

        template <typename Tuple>
        static Distance selectAxis(
            Tuple& tuple, unsigned axisZero, Distance dBest,
            Tree& tree, const Space& space, const Leaf* leaf, const Key& key, unsigned *axis)
        {
            unsigned tmp;
            Distance d = std::get<I>(tuple).selectAxis(
                tree,
                space.template get<I>(),
                leaf,
                metric::cartesian_state_element<I, Key>::get(key),
                &tmp);
            if (I == 0 || d > dBest) {
                dBest = d;
                *axis = tmp + axisZero;
            }
            unsigned dim = space.template get<I>().dimensions();
            return Next::selectAxis(tuple, axisZero + dim, dBest, tree, space, leaf, key, axis);
        }

        template <typename Tuple, typename Traversal>
        static Node *split(
            Tuple& tuple, unsigned axisZero,
            Tree& tree, const Space& space, Leaf *leaf, const Key& key, unsigned axis, Traversal& traversal)
        {
            unsigned dim = space.template get<I>().dimensions();
            if (axis < dim) {
                Node *node = std::get<I>(tuple).split(
                    tree, space.template get<I>(), leaf, metric::cartesian_state_element<I, Key>::get(key), axis, traversal);
                node->axis_ += axisZero;
                return node;
            } else if constexpr (I+1 < N) {
                return Next::split(tuple, axisZero + dim, tree, space, leaf, key, axis - dim, traversal);
            } else {
                abort();
            }
        }

        template <typename Tuple>
        static NodePointer& follow(
            Tuple& tuple,
            const Space& space, Node *node, unsigned axis, const Key& key)
        {
            unsigned dim = space.template get<I>().dimensions();
            if (axis < dim) {
                return std::get<I>(tuple).follow(space.template get<I>(), node, axis, metric::cartesian_state_element<I, Key>::get(key));
            } else if constexpr (I+1 < N) {
                return Next::follow(tuple, space, node, axis - dim, key);
            } else {
                abort();
            }
        }

        template <typename Tuple, typename Visitor>
        static void traverse(
            Tuple& tuple, Visitor& visitor, const Space& space, const Node *node, unsigned axis, const Key& key)
        {
            unsigned dim = space.template get<I>().dimensions();
            if (axis < dim) {
                std::get<I>(tuple).traverse(visitor, space.template get<I>(), node, axis, metric::cartesian_state_element<I, Key>::get(key));
            } else if constexpr (I+1 < N) {
                Next::traverse(tuple, visitor, space, node, axis - dim, key);
            } else {
                abort();
            }
        }

        template <typename Tuple, typename Visitor>
        static void visit(
            Tuple& tuple, Visitor& visitor, const Space& space, const Node *node, unsigned axis)
        {
            unsigned dim = space.template get<I>().dimensions();
            if (axis < dim) {
                std::get<I>(tuple).visit(visitor, space.template get<I>(), node, axis);
            } else if constexpr (I+1 < N) {
                Next::visit(tuple, visitor, space, node, axis - dim);
            } else {
                abort();
            }
        }

        template <typename Tuple, typename Clear>
        static void clear(
            Tuple& tuple, Clear& visitor, const Space& space, Node *node, unsigned axis)
        {
            unsigned dim = space.template get<I>().dimensions();
            if (axis < dim) {
                std::get<I>(tuple).clear(visitor, space.template get<I>(), node, axis);
            } else if constexpr (I+1 < N) {
                Next::clear(tuple, visitor, space, node, axis - dim);
            } else {
                abort();
            }
        }
    };

    template <std::size_t N, typename Tree, typename Key, typename Metric, typename Get>
    struct CartesianHelper<N, N, Tree, Key, Metric, Get> {
        using Distance = distance_t<Tree>;
        using Space = metric::Space<Key, Metric>;

        template <typename Tuple, typename Leaf, typename Q>
        static Distance selectAxis(Tuple&, unsigned, Distance dBest, Tree&, const Space&, const Leaf *, const Q&, unsigned *) {
            return dBest;
        }
    };


    template <std::size_t I, typename Tree, typename Key, typename Metric, typename Get>
    using cartesian_traversal_element_t = Traversal<
        Tree,
        typename metric::cartesian_state_element<I, Key>::type,
        metric::cartesian_element_t<I, Metric>,
        CartesianGet<Get, I>>;

    template <typename Tree, typename Key, typename M, typename Get, typename Indices>
    class CartesianTraversal;

    template <typename Tree, typename Key, typename M, typename Get, std::size_t ...I>
    class CartesianTraversal<Tree, Key, M, Get, std::index_sequence<I...>> {
    protected:
        using Leaf = leaf_t<Tree>;
        using Node = node_t<Tree>;
        using Distance = distance_t<Tree>;
        using Metric = M;
        using Space = metric::Space<Key, Metric>;
        using Concurrency = concurrency_t<Tree>;
        using NodePointer = node_pointer_t<Tree>;

        static constexpr std::size_t N = sizeof...(I);

    private:
        std::tuple<cartesian_traversal_element_t<I, Tree, Key, Metric, Get>...> tuple_;

    public:
        explicit CartesianTraversal(const CartesianTraversal& traversal)
            : tuple_(traversal.tuple_)
        {
        }

        explicit CartesianTraversal(const Space& space)
            : tuple_(cartesian_traversal_element_t<I, Tree, Key, Metric, Get>(space.template get<I>())...)
        {
        }

        void grow(const Space& space, Region<Key, Metric, Concurrency>& region, const Key& q) {
            (std::get<I>(tuple_).grow(
                space.template get<I>(),
                region.template get<I>(),
                metric::cartesian_state_element<I, Key>::get(q)), ...);
        }

        Distance selectAxis(Tree& tree, const Space& space, const Leaf *leaf, const Key& q, unsigned *axis) {
            return CartesianHelper<0, N, Tree, Key, Metric, Get>::selectAxis(
                tuple_, 0, 0, tree, space, leaf, q, axis);
        }

        template <typename Traversal>
        Node *split(Tree& tree, const Space& space, Leaf *leaf, const Key& q, unsigned axis, Traversal& traversal) {
            return CartesianHelper<0, N, Tree, Key, Metric, Get>::split(
                tuple_, 0, tree, space, leaf, q, axis, traversal);
        }

        NodePointer& follow(const Space& space, Node *node, unsigned axis, const Key& key) {
            return CartesianHelper<0, N, Tree, Key, Metric, Get>::follow(
                tuple_, space, node, axis, key);
        }

        template <typename Visitor>
        void visit(Visitor& visitor, const Space& space, const Node *node, unsigned axis) {
            CartesianHelper<0, N, Tree, Key, Metric, Get>::visit(
                tuple_, visitor, space, node, axis);
        }

        template <typename Clear>
        void clear(Clear& visitor, const Space& space, Node *node, unsigned axis) {
            CartesianHelper<0, N, Tree, Key, Metric, Get>::clear(
                tuple_, visitor, space, node, axis);
        }

        template <std::size_t J>
        decltype(auto) get() { return std::get<J>(tuple_); }
        template <std::size_t J>
        decltype(auto) get() const { return std::get<J>(tuple_); }
    };

    template <typename Tree, typename Key, typename ... M, typename Get>
    class Traversal<Tree, Key, metric::Cartesian<M...>, Get>
        : public CartesianTraversal<Tree, Key, metric::Cartesian<M...>, Get, std::index_sequence_for<M...>>
    {
        using Base = CartesianTraversal<
            Tree,
            Key,
            metric::Cartesian<M...>,
            Get,
            std::index_sequence_for<M...>>;

    public:
        using Base::Base;
    };
}

#endif
