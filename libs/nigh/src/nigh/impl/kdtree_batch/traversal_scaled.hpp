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
#ifndef NIGH_IMPL_KDTREE_BATCH_TRAVERSAL_SCALED_HPP
#define NIGH_IMPL_KDTREE_BATCH_TRAVERSAL_SCALES_HPP

#include "traversal.hpp"
#include "types.hpp"

namespace unc::robotics::nigh::impl::kdtree_batch {
    template <typename Tree, typename Key, typename M, typename W, typename Get>
    class ScaledTraversalBase : Traversal<Tree, Key, M, Get> {
        using Base = Traversal<Tree, Key, M, Get>;
        using Node = node_t<Tree>;
        using Leaf = leaf_t<Tree>;
        using NodePointer = node_pointer_t<Tree>;
        using Metric = metric::Scaled<M, W>;
        using Space = metric::Space<Key, Metric>;
        using Concurrency = concurrency_t<Tree>;

    protected:
        using Base::selectAxis;

    public:
        explicit ScaledTraversalBase(const Space& space)
            : Base(space.space())
        {
        }
        
        void grow(const Space& space, Region<Key, Metric, Concurrency>& region, const Key& q) {
            Base::grow(space.space(), region, q);
        }

        template <typename Traversal>
        Node *split(Tree& tree, const Space& space, Leaf *leaf, const Key& q, unsigned axis, Traversal& traversal) {
            return Base::split(tree, space.space(), leaf, q, axis, traversal);
        }

        NodePointer& follow(const Space& space, Node *node, unsigned axis, const Key& key) {
            return Base::follow(space.space(), node, axis, key);
        }

        template <typename Visitor>
        void visit(Visitor& visitor, const Space& space, const Node *node, unsigned axis) {
            Base::visit(visitor, space.space(), node, axis);
        }

        template <typename Clear>
        void clear(Clear& visitor, const Space& space, Node *node, unsigned axis) {
            Base::clear(visitor, space.space(), node, axis);
        }
    };

    template <typename Tree, typename Key, typename M, typename W, typename Get>
    class ScaledTraversal;
    
    template <typename Tree, typename Key, typename M, std::intmax_t num, std::intmax_t den, typename Get>
    class ScaledTraversal<Tree, Key, M, std::ratio<num, den>, Get>
        : public ScaledTraversalBase<Tree, Key, M, std::ratio<num, den>, Get>
    {
        using Base = ScaledTraversalBase<Tree, Key, M, std::ratio<num, den>, Get>;
        using Node = node_t<Tree>;
        using Leaf = leaf_t<Tree>;
        using Distance = distance_t<Tree>;
        using NodePointer = node_pointer_t<Tree>;
        using Weight = std::ratio<num, den>;
        using Metric = metric::Scaled<M, Weight>;
        using Space = metric::Space<Key, Metric>;
        using Concurrency = concurrency_t<Tree>;

    public:
        using Base::Base;

        Base& traversal() { return *this; }
        const Base& traversal() const { return *this; }

        auto distToRegion(const Key& key, const Region<Key, Metric, Concurrency>& region) const {
            // TODO: consistent scaling mechanism...
            return Base::distToRegion(key, region) * Weight::num / Weight::den;
        }

        Distance selectAxis(Tree& tree, const Space& space, const Leaf* leaf, const Key& q, unsigned *axis) {
            // TODO: consistent scaling mechanism...
            return Base::selectAxis(tree, space.space(), leaf, q, axis) * Weight::num / Weight::den;
        }
    };

    template <typename Tree, typename Key, typename M, typename Get>
    class ScaledTraversal<Tree, Key, M, void, Get>
        : public ScaledTraversalBase<Tree, Key, M, void, Get>
    {
        using Base = ScaledTraversalBase<Tree, Key, M, void, Get>;
        using Node = node_t<Tree>;
        using Leaf = leaf_t<Tree>;
        using Distance = distance_t<Tree>;
        using NodePointer = node_pointer_t<Tree>;
        using Weight = Distance; 
        using Metric = metric::Scaled<M, void>;
        using Space = metric::Space<Key, Metric>;
        using Concurrency = concurrency_t<Tree>;

        Weight weight_;
        
    public:
        ScaledTraversal(const Space& space)
            : Base(space)
            , weight_(space.weight())
        {
        }

        Base& traversal() { return *this; }
        const Base& traversal() const { return *this; }

        auto distToRegion(const Key& key, const Region<Key, Metric, Concurrency>& region) const {
            // TODO: make distToRegion take space as an argument, then
            // we will not need to keep around a copy of the space's
            // weight.
            return Base::distToRegion(key, region) * weight_;
        }

        Distance selectAxis(Tree& tree, const Space& space, const Leaf* leaf, const Key& q, unsigned *axis) {
            return Base::selectAxis(tree, space.space(), leaf, q, axis) * weight_;
        }
    };

    template <typename Tree, typename Key, typename M, typename W, typename Get>
    class Traversal<Tree, Key, metric::Scaled<M, W>, Get>
        : public ScaledTraversal<Tree, Key, M, W, Get>
    {
    public:
        using ScaledTraversal<Tree, Key, M, W, Get>::ScaledTraversal;
    };
}

#endif
