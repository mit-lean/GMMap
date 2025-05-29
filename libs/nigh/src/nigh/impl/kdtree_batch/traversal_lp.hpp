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
#ifndef NIGH_IMPL_KDTREE_BATCH_TRAVERSAL_LP_HPP
#define NIGH_IMPL_KDTREE_BATCH_TRAVERSAL_LP_HPP

#include "traversal.hpp"
#include "lp_branch.hpp"
#include "types.hpp"

namespace unc::robotics::nigh::impl::kdtree_batch {
    template <typename Tree, typename Key, int p, typename Get>
    class Traversal<Tree, Key, metric::LP<p>, Get> {
    protected:
        using Metric = metric::LP<p>;
        using Leaf = leaf_t<Tree>;
        using Node = node_t<Tree>;
        using Concurrency = concurrency_t<Tree>;
        using NodePointer = node_pointer_t<Tree>;
        using Space = metric::Space<Key, metric::LP<p>>;

        using LPBranch = kdtree_batch::LPBranch<
            value_t<Tree>,
            space_t<Tree>,
            Concurrency,
            kBatchSize<Tree>,
            p>;

    public:
        Traversal(const Space&) {
        }

        void grow(const Space& space, Region<Key, Metric, Concurrency>& region, const Key& q) {
            region.grow(space, q);
        }

        auto selectAxis(Tree&, const Space&, const Leaf *leaf, const Key& q, unsigned *axis) {
            return Get::part(leaf->region()).selectAxis(axis);
        }

        // TODO: remove Tree as argument for all traversals
        template <typename Traversal>
        Node *split(Tree& tree, const Space&, Leaf *leaf, const Key& key, unsigned axis, Traversal& traversal) {
            using T = value_t<Tree>;
            using Distance = typename metric::Space<Key, metric::LP<p>>::Distance;
            static constexpr std::size_t batchSize = kBatchSize<Tree>;

            std::array<T*, batchSize> ptrs;
            for (std::size_t i=0 ; i<batchSize ; ++i)
                ptrs[i] = leaf->elements() + i;

            std::nth_element(
                ptrs.begin(), ptrs.begin() + batchSize/2, ptrs.end(),
                [&] (const T* a, const T* b) {
                    return (Space::coeff(Get::part(tree.getKey(*a)), axis) <
                            Space::coeff(Get::part(tree.getKey(*b)), axis));
                });

            Leaf *c0 = tree.template allocWithSpace<Leaf>(traversal, tree.keyFn(), ptrs.begin(), ptrs.begin() + batchSize/2);
            Leaf *c1 = tree.template allocWithSpace<Leaf>(traversal, tree.keyFn(), ptrs.begin() + batchSize/2, ptrs.end());

            // std::cout << leaf->size() << ": " << c0->size() << " + " << c1->size() << std::endl;
            assert(c0->size() + c1->size() == batchSize);
            assert(Get::part(c0->region()).max(axis) <= Get::part(c1->region()).min(axis));

            Distance split = (Get::part(c0->region()).max(axis) + Get::part(c1->region()).min(axis)) / 2;
#ifndef NDEBUG
            {
		bool correctSplit = true;
                for (std::size_t i=0 ; i<batchSize/2 ; ++i)
		    correctSplit &= Space::coeff(Get::part(tree.getKey(c0->elements()[i])), axis) <= split;
                for (std::size_t i=0 ; i<batchSize/2 ; ++i)
		    correctSplit &= Space::coeff(Get::part(tree.getKey(c1->elements()[i])), axis) >= split;
		if (!correctSplit) {
		    // std::cout << "SPLIT: " << axis << ": " << split << std::endl;
		    // for (std::size_t i=0 ; i<batchSize/2 ; ++i)
		    //     std::cerr << "c0[" << i << "]: " << tree.getKey(c0->elements()[i]).transpose() << std::endl;
		    // for (std::size_t i=0 ; i<batchSize/2 ; ++i)
		    //     std::cerr << "c1[" << i << "]: " << tree.getKey(c1->elements()[i]).transpose() << std::endl;
		    assert(correctSplit);
		}
            }
#endif
            return tree.template alloc<LPBranch>(leaf, axis, split, c0, c1);
        }

        NodePointer& follow(const Space&, Node* node, unsigned axis, const Key& key) {
            LPBranch *branch = static_cast<LPBranch*>(node);
            int childNo = Space::coeff(key, axis) > branch->split();
            return branch->child(childNo);
        }

        template <typename Visitor>
        void traverse(Visitor& visitor, const Space&, const Node *node, unsigned axis, const Key& key) {
            const LPBranch *branch = static_cast<const LPBranch*>(node);
            int childNo = Space::coeff(key, axis) > branch->split();
            visitor(branch->child(childNo).load(std::memory_order_acquire));
            visitor(branch->child(!childNo).load(std::memory_order_acquire));
        }

        template <typename Visitor>
        void visit(Visitor& visitor, const Space&, const Node *node, unsigned) {
            const LPBranch *branch = static_cast<const LPBranch*>(node);
            visitor(branch->child(0));
            visitor(branch->child(1));
        }

        template <typename Clear>
        void clear(Clear& visitor, const Space&, Node *node, unsigned) {
            LPBranch *branch = static_cast<LPBranch*>(node);
            visitor(branch->child(0));
            visitor(branch->child(1));
            visitor.dealloc(branch);
        }

        auto distToRegion(const Key& key, const Region<Key, Metric, Concurrency>& region) const {
            return region.distTo(key);
        }
    };
}

#endif
