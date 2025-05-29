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
#ifndef NIGH_IMPL_KDTREE_MEDIAN_BUILDER_HPP
#define NIGH_IMPL_KDTREE_MEDIAN_BUILDER_HPP

#include "node.hpp"
#include "accum.hpp"
#include "accums.hpp"

namespace unc::robotics::nigh::impl::kdtree_median {
    template <typename Tree>
    class Builder {
        using Key = typename Tree::Key;
        using Metric = typename Tree::Metric;

        Tree& tree_;

        Accum<Key, Metric> accum_;
        
    public:
        Builder(Tree& tree)
            : tree_(tree)
        {
        }

        template <class T, class ... Args>
        T* allocate(Args&& ... args) {
            return tree_.blocks_.template allocate<T>(std::forward<Args>(args)...);
        }

        decltype(auto) getKey(const typename Tree::Type& t) {
            return tree_.getKey(t);
        }

        template <typename Iter>
        Node* operator() (Iter first, Iter last) {
            if (std::distance(first, last) <= 1)
                return nullptr;

            // TODO: we would get better cache locality if we
            // accumulated the region bounds in a loop here, instead
            // of having each space iterate individually.  The problem
            // is that SO(2) needs to sort the elements locally to
            // find a good partition.
            
            // Iter it = first;
            // accum_.init(tree_.metricSpace(), tree_.getKey(*it));
            // while (++it != last)
            //     accum_.grow(tree_.metricSpace(), tree_.getKey(*it));

            unsigned axis;
            accum_.selectAxis(*this, tree_.metricSpace(), &axis, first, last);
            return accum_.partition(*this, tree_.metricSpace(), axis, first, last);
        }
    };
}

#endif
