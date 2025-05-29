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
#ifndef NIGH_IMPL_KDTREE_MEDIAN_ACCUM_SCALED_HPP
#define NIGH_IMPL_KDTREE_MEDIAN_ACCUM_SCALED_HPP

#include "accum.hpp"
#include "node.hpp"

namespace unc::robotics::nigh::impl::kdtree_median {
    
    template <class Key, class M, class W, typename Get>
    class Accum<Key, metric::Scaled<M, W>, Get>
        : Accum<Key, M, Get>
    {
        using Base = Accum<Key, M, Get>;
        using Metric = metric::Scaled<M, W>;
        using Space = metric::Space<Key, Metric>;
        using Distance = typename Space::Distance;
    public:

        template <typename Builder, typename Iter>
        Distance selectAxis(
            Builder& builder, const Space& space, unsigned *axis,
            Iter first, Iter last)
        {
            return Base::selectAxis(builder, space.space(), axis, first, last) * space.weight();
        }

        template <typename Builder, typename Iter>
        Node *partition(
            Builder& builder, const Space& space, unsigned axis, Iter first, Iter last)
        {
            return Base::partition(builder, space.space(), axis, first, last);
        }
    };
}

#endif
