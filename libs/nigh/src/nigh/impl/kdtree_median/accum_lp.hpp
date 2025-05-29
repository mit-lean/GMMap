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
#ifndef NIGH_IMPL_KDTREE_MEDIAN_ACCUM_LP_HPP
#define NIGH_IMPL_KDTREE_MEDIAN_ACCUM_LP_HPP

#include "accum.hpp"
#include "node.hpp"
#include "lp_branch.hpp"
#include "../region.hpp"
#include "../regions.hpp"
#include <algorithm>

namespace unc::robotics::nigh::impl::kdtree_median {
    
    template <typename Key, int p, typename Get>
    class Accum<Key, metric::LP<p>, Get> {
        using Metric = metric::LP<p>;
        using Space = metric::Space<Key, metric::LP<p>>;
        using Distance = typename Space::Distance;

        Region<Key, Metric, NoThreadSafety> region_;
        
    public:
        // void init(const Space& space, const Key& q) {
        //     region_.init(space, q);
        // }

        // void grow(const Space& space, const Key& q) {
        //     region_.grow(space, q);
        // }

        // auto selectAxis(const Space&, unsigned *axis) {
        //     return region_.selectAxis(axis);
        // }

        template <typename Builder, typename Iter>
        Distance selectAxis(
            Builder& builder, const Space& space, unsigned *axis, Iter first, Iter last)
        {
            assert(first != last);
            Iter it = first;
            Region<Key, Metric, NoThreadSafety> region;
            region.init(space, Get::part(builder.getKey(*it)));
            while (++it != last)
                region.grow(space, Get::part(builder.getKey(*it)));

            return region.selectAxis(axis);
        }
        
        template <typename Builder, typename Iter>
        Node *partition(
            Builder& builder, const Space&, unsigned axis, Iter first, Iter last)
        {
            auto cmp = [&] (const auto& a, const auto& b) {
                return (Space::coeff(Get::part(builder.getKey(a)), axis) <
                        Space::coeff(Get::part(builder.getKey(b)), axis));
            };

            Iter mid = first + std::distance(first, last) / 2;
            std::nth_element(first, mid, last, cmp);
            Distance split =
                (Space::coeff(Get::part(builder.getKey(*std::max_element(first, mid, cmp))), axis) +
                 Space::coeff(Get::part(builder.getKey(*mid)), axis)) / 2;

            return builder.template allocate<LPBranch<Distance>>(
                axis, split,
                builder(first, mid),
                builder(mid, last));
        }
    };
}

#endif
