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
#ifndef NIGH_IMPL_KDTREE_MEDIAN_ACCUM_SO3_HPP
#define NIGH_IMPL_KDTREE_MEDIAN_ACCUM_SO3_HPP

#include "accum.hpp"
#include "node.hpp"
#include "so3_root.hpp"
#include "so3_branch.hpp"
#include "../so3_region.hpp"

namespace unc::robotics::nigh::impl::kdtree_median {
    
    template <typename Key, typename Get>
    class Accum<Key, metric::SO3, Get> {
        using Metric = metric::SO3;
        using Space = metric::Space<Key, Metric>;
        using Distance = typename Space::Distance;

        int vol_{-1};
        
    public:
        template <typename Builder, typename Iter>
        Distance selectAxis(
            Builder& builder, const Space&, unsigned *axis, Iter first, Iter last)
        {
            if (vol_ < 0) {
                *axis = 0;
                return PI_2<Distance>;
            } else {
                assert(first != last);
                Iter it = first;
                SO3Region<Key> region(Get::part(builder.getKey(*it)), vol_);
                while (++it != last)
                    region.grow(Get::part(builder.getKey(*it)), vol_);
                return region.selectAxis(axis, vol_);
            }
        }
        
        template <typename Builder, typename Iter>
        Node *partition(
            Builder& builder, const Space&, unsigned axis, Iter first, Iter last)
        {
            if (vol_ < 0) {
                Eigen::Array<std::size_t, 4, 1> counts;
                counts.setZero();
                for (Iter it = first ; it != last ; ++it)
                    ++counts[so3::volIndex(Get::part(builder.getKey(*it)))];
                
                std::array<Iter, 4> its;
                std::array<Iter, 3> stops;
                its[0] = first;
                for (unsigned i=1 ; i<4 ; ++i)
                    its[i] = stops[i-1] = its[i-1] + counts[i-1];
                for (unsigned i=0 ; i<3 ; ++i)
                    for (unsigned v ; its[i] != stops[i] ; ++(its[v]))
                        if ((v = so3::volIndex(Get::part(builder.getKey(*its[i])))) != i)
                            std::iter_swap(its[i], its[v]);

                SO3Root *root = builder.template allocate<SO3Root>(axis);
                for (unsigned i=0 ; i<3 ; ++i)
                    root->offset(i) = std::distance(first, stops[i]);

                vol_ = 0;
                root->child(0) = builder(first, stops[0]);
                vol_ = 1;
                root->child(1) = builder(stops[0], stops[1]);
                vol_ = 2;
                root->child(2) = builder(stops[1], stops[2]);
                vol_ = 3;
                root->child(3) = builder(stops[2], last);
                vol_ = -1;
                
                return root;
            } else {
                Iter mid = first + std::distance(first, last)/2;
                auto cmp = [&] (const auto& a, const auto& b) {
                    return (so3::project1(Get::part(builder.getKey(a)), vol_, axis) <
                            so3::project1(Get::part(builder.getKey(b)), vol_, axis));
                };
                std::nth_element(first, mid, last, cmp);
                auto max0 = std::max_element(first, mid, cmp);
                Eigen::Matrix<Distance, 2, 1> split =
                    (so3::project(Get::part(builder.getKey(*max0)), vol_, axis) +
                     so3::project(Get::part(builder.getKey(*mid)), vol_, axis)).normalized();

                return builder.template allocate<SO3Branch<Distance>>(
                    axis, split, builder(first, mid), builder(mid, last));
            }
        }
    };
}

#endif
