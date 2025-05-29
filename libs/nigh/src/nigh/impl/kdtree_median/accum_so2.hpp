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
#ifndef NIGH_IMPL_KDTREE_MEDIAN_ACCUM_SO2_HPP
#define NIGH_IMPL_KDTREE_MEDIAN_ACCUM_SO2_HPP

#include "accum.hpp"
#include "node.hpp"
#include "lp_branch.hpp"
#include "../so2.hpp"
#include <algorithm>
#include <iostream> // TODO:remove

namespace unc::robotics::nigh::impl::kdtree_median {
    
    template <typename Key, int p, typename Get>
    class Accum<Key, metric::SO2<p>, Get> {
        using Metric = metric::SO2<p>;
        using Space = metric::Space<Key, Metric>;
        using Distance = typename Space::Distance;

        std::vector<Distance> Q_;

        template <class Builder, class Iter>
        decltype(auto) splitAxis(Builder& builder, unsigned a, Iter first, Iter last) {
            std::size_t j=0;
            for (Iter it=first ; it != last ; ++it)
                    Q_[j++] = so2::bound(Space::coeff(Get::part(builder.getKey(*it)), a));
            return so2::split(Q_);
        }

    public:
        template <class Builder, class Iter>
        Distance selectAxis(
            Builder& builder, const Space& space, unsigned *axis, Iter first, Iter last)
        {
            Distance maxDist = -1;
            Q_.resize(std::distance(first, last));
            std::size_t dim = space.dimensions();
            *axis = 0;
            for (std::size_t a=0 ; a<dim ; ++a) {
                [[maybe_unused]] auto [distToSplit, split] = splitAxis(builder, a, first, last);
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
        
        template <class Builder, class Iter>
        Node *partition(
            Builder& builder, const Space&, unsigned axis, Iter first, Iter last)
        {
            assert(Q_.size() == std::distance(first, last));
            
            // TODO: so2split works for all arrangements.  But we
            // can use an nth_element (e.g., faster) if we are in
            // a bounded subregion.  (Do same with following.)
            [[maybe_unused]] auto [distToSplit, split] = splitAxis(builder, axis, first, last);
            (void)distToSplit; // GCC seems to ignore [[maybe_unused]]

            // partition such that [first, mid1) is less than the
            // split, [mid1, mid2) is equal to the split value, and
            // [mid2, last) is greater than the split.  Since so2split
            // finds a median point, we're guratanteed that the
            // half-way point will be between mid1 and mid2.
            Iter mid1 = std::partition(first, last, [&, s=split] (const auto& t) {
                return (so2::ccwDist(s, Space::coeff(Get::part(builder.getKey(t)), axis))
                        < PI<Distance>);
            });
            Iter mid2 = std::partition(mid1, last, [&, s=split] (const auto& t) {
                return (so2::ccwDist(s, Space::coeff(Get::part(builder.getKey(t)), axis))
                        <= PI<Distance>);                    
            });

            Iter mid = first + std::distance(first, last)/2;

            assert(mid1 <= mid && mid <= mid2);

            return builder.template allocate<LPBranch<Distance>>(
                axis, split,
                builder(first, mid),
                builder(mid, last));
        }
    };
}

#endif
