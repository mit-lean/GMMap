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
#ifndef NIGH_IMPL_KDTREE_BATCH_TRAVERSAL_SO3_HPP
#define NIGH_IMPL_KDTREE_BATCH_TRAVERSAL_SO3_HPP

#include "../so3.hpp"
#include "../so3_region.hpp"
#include "traversal.hpp"
#include "so3_branch.hpp"
#include "so3_root.hpp"
#include "types.hpp"
#include <numeric>

namespace unc::robotics::nigh::impl::kdtree_batch {
    template <typename Tree, typename Key, typename Get>
    class Traversal<Tree, Key, metric::SO3, Get> {
    protected:
        using T = value_t<Tree>;
        using Leaf = leaf_t<Tree>;
        using Node = node_t<Tree>;
        using KeyFn = keyfn_t<Tree>;
        using Concurrency = concurrency_t<Tree>;
        using NodePointer = node_pointer_t<Tree>;
        using Space = metric::Space<Key, metric::SO3>;
        using Distance = typename Space::Distance;

        static constexpr std::size_t batchSize = kBatchSize<Tree>;

        using SO3Branch = kdtree_batch::SO3Branch<T, space_t<Tree>, Concurrency, batchSize>;
        using SO3Root = kdtree_batch::SO3Root<T, space_t<Tree>, Concurrency, batchSize>;
        using Metric = metric::SO3;

        int vol_{-1};

    public:
        explicit Traversal(const Traversal& traversal)
            : vol_(traversal.vol_)
        {
        }

        explicit Traversal(const Space&) {
        }

        int vol() const { return vol_; }

        void grow(const Space&, Region<Key, Metric, Concurrency>& region, const Key& q) {
        }

        auto selectAxis(Tree& tree, const Space&, const Leaf *leaf, const Key& key, unsigned *axis) {
            if (vol_ < 0) {
                *axis = 0;
                return PI_2<Distance>;
            } else {
                SO3Region<Key> region(key, vol_);
                for (std::size_t i=0 ; i<batchSize ; ++i)
                    region.grow(Get::part(tree.getKey(leaf->elements()[i])), vol_);
                return region.selectAxis(axis, vol_);
            }
        }

        template <typename Traversal>
        Node *split(Tree& tree, const Space&, Leaf *leaf, const Key& key, unsigned axis, Traversal& traversal) {
            using T = value_t<Tree>;

            if (vol_ == -1) {
                struct Block {
                    std::array<T*, batchSize> ptrs_;
                    std::size_t count_{0};
                    void add(T* ptr) { ptrs_[count_++] = ptr; }
                    T** begin() { return ptrs_.begin(); }
                    T** end() { return begin() + count_; }
                };
                std::array<Block, 4> blocks;
                for (std::size_t i=0 ; i<batchSize ; ++i) {
                    T* t = leaf->elements() + i;
                    blocks[so3::volIndex(Get::part(tree.getKey(*t)))].add(t);
                }
                Leaf* leaves[4];
                for (vol_=0 ; vol_<4 ; ++vol_)
                    leaves[vol_] = blocks[vol_].count_
                        ? tree.template allocWithSpace<Leaf>(traversal, tree.keyFn(), blocks[vol_].begin(), blocks[vol_].end())
                        : nullptr;
                vol_ = -1;
                return tree.template alloc<SO3Root>(leaf, axis, leaves);
            } else {
                std::array<T*, batchSize> ptrs;
                std::iota(ptrs.begin(), ptrs.end(), leaf->elements());

                for (std::size_t i = 0 ;i<batchSize ; ++i)
                    assert(so3::volIndex(Get::part(tree.getKey(leaf->elements()[i]))) == (unsigned)vol_);

                auto cmp = [&] (const T* a, const T* b) {
                    return (so3::project1(Get::part(tree.getKey(*a)), vol_, axis) <
                            so3::project1(Get::part(tree.getKey(*b)), vol_, axis));
                };

                std::nth_element(ptrs.begin(), ptrs.begin() + batchSize/2, ptrs.end(), cmp);

                Leaf *c0 = tree.template allocWithSpace<Leaf>(
                    traversal, tree.keyFn(), ptrs.begin(), ptrs.begin() + batchSize/2);
                Leaf *c1 = tree.template allocWithSpace<Leaf>(
                    traversal, tree.keyFn(), ptrs.begin() + batchSize/2, ptrs.end());

                Eigen::Matrix<Distance, 2, 1> q0max = so3::project(
                    Get::part(tree.getKey(**std::max_element(ptrs.begin(), ptrs.begin() + batchSize/2, cmp))),
                    vol_, axis);
                Eigen::Matrix<Distance, 2, 1> q1min = so3::project(
                    Get::part(tree.getKey(*ptrs[batchSize/2])),
                    vol_, axis);

                assert(q0max[0] <= q1min[0]);

                Eigen::Matrix<Distance, 2, 1> split = (q0max + q1min).normalized();

#if 0
                {
                    std::cerr << "============= " << vol_ << ", " << axis << "\n";
                    std::size_t i = 0;
                    for ( ; i<batchSize/2 ; ++i)
                        std::cerr << so3::project1(Get::part(tree.getKey(*ptrs[i])), vol_, axis)
                                  << ": "
                                  << Get::part(tree.getKey(*ptrs[i])).coeffs().transpose()
                                  << std::endl;
                    std::cerr << "  [ " << c0->region().max(axis).transpose() << " ]\n";
                    std::cerr << "--- " << split.transpose() << std::endl;
                    std::cerr << "  [ " << c1->region().min(axis).transpose() << " ]\n";
                    for ( ; i<batchSize ; ++i)
                        std::cerr << so3::project1(Get::part(tree.getKey(*ptrs[i])), vol_, axis) << std::endl;
                }
#endif
#ifndef NDEBUG
                {
                    std::size_t i = 0;
                    for ( ; i<batchSize/2 ; ++i) {
                        const auto& q = Get::part(tree.getKey(*ptrs[i]));
                        Distance dot = so3::dotSplit(split, q, vol_, axis);
                        assert(dot <= 0);
                    }

                    for ( ; i<batchSize ; ++i) {
                        const auto& q = Get::part(tree.getKey(*ptrs[i]));
                        Distance dot = so3::dotSplit(split, q, vol_, axis);
                        assert(dot >= 0);
                    }
                }
#endif

                return tree.template alloc<SO3Branch>(leaf, axis, split, c0, c1);
            }
        }

        NodePointer& follow(const Space&, Node* node, unsigned axis, const Key& key) {
            if (vol_ == -1) {
                assert(axis == 0);
                SO3Root *branch = static_cast<SO3Root*>(node);
                return branch->child(vol_ = so3::volIndex(key));
            } else {
                assert(axis < 3);
                SO3Branch *branch = static_cast<SO3Branch*>(node);
                Distance dot = so3::dotSplit(branch->split(), key, vol_, axis);
                return branch->child(dot > 0);
            }
        }

        template <typename Visitor>
        void visit(Visitor& visitor, const Space&, const Node *node, unsigned) {
            if (vol_ == -1) {
                const SO3Root *branch = static_cast<const SO3Root*>(node);
                for (vol_ = 0 ; vol_ < 4 ; ++vol_)
                    if (const Node *child = branch->child(vol_))
                        visitor(child);
                vol_ = -1;
            } else {
                const SO3Branch *branch = static_cast<const SO3Branch*>(node);
                visitor(branch->child(0));
                visitor(branch->child(1));
            }
        }


        template <typename Clear>
        void clear(Clear& visitor, const Space&, Node *node, unsigned) {
            if (vol_ == -1) {
                SO3Root *branch = static_cast<SO3Root*>(node);
                for (vol_ = 0 ; vol_ < 4 ; ++vol_)
                    if (Node *child = branch->child(vol_))
                        visitor(child);
                visitor.dealloc(branch);
                vol_ = -1;
            } else {
                SO3Branch *branch = static_cast<SO3Branch*>(node);
                visitor(branch->child(0));
                visitor(branch->child(1));
                visitor.dealloc(branch);
            }
        }
    };
}

#endif
