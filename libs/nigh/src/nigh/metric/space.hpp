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
#ifndef NIGH_METRIC_SPACE_HPP
#define NIGH_METRIC_SPACE_HPP

#include "metric.hpp"
#include <type_traits>

namespace unc::robotics::nigh::metric {
    // Space is a combination of storage type and metric.  It is
    // essentially a metric space where Metric is the metric, and T is
    // the type of element composing the set.  Thus, for example, R^3
    // with an L2 metric can be represented as Space<Eigen::Vector3d,
    // L2>.  The space must be specialized for each combindation of
    // 'T' and 'Metric', and the specializion must provide a small
    // number of member methods to make the type work.
    template <typename T, typename Metric, typename Enabled = void>
    struct Space;

    template <typename T>
    struct is_space : std::false_type {};

    template <typename T, typename Metric, typename Enabled>
    struct is_space<Space<T, Metric, Enabled>> : is_metric<Metric> {};

    template <typename T>
    constexpr bool is_space_v = is_space<T>::value;
}

#endif
