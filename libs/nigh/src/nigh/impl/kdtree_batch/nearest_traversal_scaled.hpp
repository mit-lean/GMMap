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
#ifndef NIGH_IMPL_KDTREE_BATCH_NEAREST_TRAVERSAL_SCALED_HPP
#define NIGH_IMPL_KDTREE_BATCH_NEAREST_TRAVERSAL_SCALED_HPP

#include "nearest_traversal.hpp"

namespace unc::robotics::nigh::impl::kdtree_batch {
    template <typename Tree, typename Key, typename M, typename W, typename Get>
    class ScaledNearestTraversal;

    template <typename Tree, typename Key, typename M, std::intmax_t num, std::intmax_t den, typename Get>
    class ScaledNearestTraversal<Tree, Key, M, std::ratio<num, den>, Get>
        : NearestTraversal<Tree, Key, M, Get>
    {
        using Base = NearestTraversal<Tree, Key, M, Get>;
        using Node = node_t<Tree>;
        using Weight = std::ratio<num, den>;
        using Metric = metric::Scaled<M, Weight>;
        using Space = metric::Space<Key, Metric>;
        using Concurrency = concurrency_t<Tree>;

    public:
        explicit ScaledNearestTraversal(const Space& space)
            : Base(space.space())
        {
        }

        auto distToRegion(const Key& key, const Region<Key, Metric, Concurrency>& region) const {
            return Base::distToRegion(key, region) * Weight::num / Weight::den;
        }

        template <typename Visitor>
        void traverse(Visitor& visitor, const Space& space, const Node *node, unsigned axis, const Key& key) {
            Base::traverse(visitor, space.space(), node, axis, key);
        }
    };

    template <typename Tree, typename Key, typename M, typename Get>
    class ScaledNearestTraversal<Tree, Key, M, void, Get>
        : NearestTraversal<Tree, Key, M, Get>
    {
        using Base = NearestTraversal<Tree, Key, M, Get>;
        using Node = node_t<Tree>;
        using Weight = distance_t<Tree>;
        using Metric = metric::Scaled<M, void>;
        using Space = metric::Space<Key, Metric>;
        using Concurrency = concurrency_t<Tree>;

        Weight weight_;
        
    public:
        explicit ScaledNearestTraversal(const Space& space)
            : Base(space.space())
            , weight_(space.weight())
        {
        }

        auto distToRegion(const Key& key, const Region<Key, Metric, Concurrency>& region) const {
            // TODO: make distToRegion take Space as an argument, then
            // we will not need to store the weight as a member of
            // this class.
            return Base::distToRegion(key, region) * weight_;
        }

        template <typename Visitor>
        void traverse(Visitor& visitor, const Space& space, const Node *node, unsigned axis, const Key& key) {
            Base::traverse(visitor, space.space(), node, axis, key);
        }
    };
    
    template <typename Tree, typename Key, typename M, typename W, typename Get>
    class NearestTraversal<Tree, Key, metric::Scaled<M, W>, Get>
        : public ScaledNearestTraversal<Tree, Key, M, W, Get>
    {
        using Base = ScaledNearestTraversal<Tree, Key, M, W, Get>;
    public:
        using Base::Base;
    };
}


#endif
