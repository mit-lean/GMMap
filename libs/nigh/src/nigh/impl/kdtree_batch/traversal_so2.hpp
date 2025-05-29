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
#ifndef NIGH_IMPL_KDTREE_BATCH_TRAVERSAL_SO2_HPP
#define NIGH_IMPL_KDTREE_BATCH_TRAVERSAL_SO2_HPP

#include "../constants.hpp"
#include "../so2.hpp"
#include "traversal.hpp"
#include "types.hpp"
#include "lp_branch.hpp"
#include <cmath>

namespace unc::robotics::nigh::impl::kdtree_batch {

    template <typename Tree, typename Key, int p, typename Get>
    class Traversal<Tree, Key, metric::SO2<p>, Get> {
    protected:
        using Metric = metric::SO2<p>;
        using Space = metric::Space<Key, Metric>;
        using Distance = typename Space::Distance;
        using Leaf = leaf_t<Tree>;
        using Node = node_t<Tree>;
        using Concurrency = concurrency_t<Tree>;
        using NodePointer = node_pointer_t<Tree>;

        static constexpr std::size_t batchSize = kBatchSize<Tree>;

        using SO2Branch = kdtree_batch::LPBranch<
            value_t<Tree>,
            space_t<Tree>,
            Concurrency,
            kBatchSize<Tree>,
            p>;

    public:
        Traversal(const Space&) {
        }

        void grow(const Space&, Region<Key, Metric, Concurrency>& region, const Key& key) {
            // region.grow(q);
        }

        Distance selectAxis(Tree& tree, const Space& space, const Leaf* leaf, const Key& key, unsigned *axis) const {
            Distance maxDist = -1;
            std::size_t dim = space.dimensions();
            std::array<Distance, batchSize> Q;
            *axis = 0;
            for (std::size_t a=0 ; a<dim ; ++a) {
                for (std::size_t j=0 ; j<batchSize ; ++j)
                    Q[j] = so2::bound(Space::coeff(Get::part(tree.getKey(leaf->elements()[j])), a));

                [[maybe_unused]] auto [distToSplit, split] = so2::split(Q);
                (void)split; // GCC seems to ignore [[maybe_unused]]

                // distToSplit will be in the range 0..2pi and is the
                // sum of the bounds distances to the split.  We wish
                // to find the split that minimizes this.
                if (distToSplit > maxDist) {
                    maxDist = distToSplit;
                    *axis = a;
                }
            }
            assert(maxDist != -1);
            return maxDist;
        }

        template <typename Traversal>
        Node *split(Tree& tree, const Space& space, Leaf *leaf, const Key& key, unsigned axis, Traversal& traversal) {
            using T = value_t<Tree>;
            std::array<Distance, batchSize> Q;
            for (std::size_t j=0 ; j<batchSize ; ++j)
                Q[j] = so2::bound(Space::coeff(Get::part(tree.getKey(leaf->elements()[j])), axis));

            // TODO: so2split works for all arrangements.  But we
            // can use an nth_element (e.g., faster) if we are in
            // a bounded subregion.  (Do same with following.)
            [[maybe_unused]] auto [distToSplit, split] = so2::split(Q);
            (void)distToSplit; // GCC seems to ignore [[maybe_unused]]

#if 0
            // version disabled with #if 0.  This does not work if
            // there are duplicate entries.
            std::size_t dst0 = 0;
            std::size_t dst1 = batchSize;
            T *elements[batchSize];
            for (std::size_t i=0 ; i<batchSize ; ++i) {
                T* t = leaf->elements() + i;
                if (so2::ccwDist(split, Space::coeff(Get::part(tree.getKey(*t)), axis)) > PI<Distance>) {
                    elements[--dst1] = t;
                } else {
                    elements[dst0++] = t;
                }
            }
            assert(dst0 == dst1);
            assert(dst0 == batchSize/2);

            Leaf *c0 = tree.template allocWithSpace<Leaf>(traversal, tree.keyFn(), elements, elements + batchSize/2);
            Leaf *c1 = tree.template allocWithSpace<Leaf>(traversal, tree.keyFn(), elements + batchSize/2, elements + batchSize);
#else
            std::array<T*, batchSize> ptrs;
            for (std::size_t i=0 ; i<batchSize ; ++i)
                ptrs[i] = leaf->elements() + i;

            std::nth_element(
                ptrs.begin(), ptrs.begin() + batchSize/2, ptrs.end(),
                [&tree, axis, splitVal=split - PI<Distance>] (const T* a, const T* b) {
                    return (so2::ccwDist(splitVal, Space::coeff(Get::part(tree.getKey(*a)), axis)) >
                            so2::ccwDist(splitVal, Space::coeff(Get::part(tree.getKey(*b)), axis)));
                });

            Leaf *c0 = tree.template allocWithSpace<Leaf>(traversal, tree.keyFn(), ptrs.begin(), ptrs.begin() + batchSize/2);
            Leaf *c1 = tree.template allocWithSpace<Leaf>(traversal, tree.keyFn(), ptrs.begin() + batchSize/2, ptrs.end());
#endif

            return tree.template alloc<SO2Branch>(leaf, axis, split, c0, c1);
        }

        NodePointer& follow(const Space& space, Node *node, unsigned axis, const Key& key) {
            SO2Branch *branch = static_cast<SO2Branch*>(node);
            // TODO: signed dist?
            Distance dist = so2::ccwDist(branch->split(), Space::coeff(key, axis));
            return branch->child(dist > PI<Distance>);
        }

        template <typename Visitor>
        void visit(Visitor& visitor, const Space&, const Node *node, unsigned) {
            const SO2Branch *branch = static_cast<const SO2Branch*>(node);
            visitor(branch->child(0));
            visitor(branch->child(1));
        }


        template <typename Clear>
        void clear(Clear& visitor, const Space&, Node *node, unsigned) {
            SO2Branch *branch = static_cast<SO2Branch*>(node);
            visitor(branch->child(0));
            visitor(branch->child(1));
            visitor.dealloc(branch);
        }
    };
}

#endif
