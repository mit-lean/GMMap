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
#ifndef NIGH_AUTO_STRATEGY_HPP
#define NIGH_AUTO_STRATEGY_HPP

#include "nigh_forward.hpp"
#include "linear.hpp"
#include "kdtree_batch.hpp"
#include "kdtree_median.hpp"
#include "gnat.hpp"
#include "metric/space.hpp"
#include <type_traits>

namespace unc::robotics::nigh {
    // auto_strategy<Space, Concurrency> attempts to automatically
    // determine the best nearest neighbor strategy to use based upon
    // arguments.  The priority from highest to lowest is:
    //
    // 1. the space is a decomposible metric space and we do not wish
    //    to wait for potentially long rebuilds, then KDTreeBatch is
    //    the best option.
    //
    // 2. the space is a decomposible metric space and pauses are ok,
    //    and we're not operating concurrently then KDTreeMedian may
    //    be the best option.  KDTreeBatch may perform better in
    //    practice, but the forced median balance of KDTreeMedian
    //    should provide better worst-case performance.
    //
    // 3. the space is NOT a decomposible metric spance, and there is
    //    no need for concurrent inserts, then GNAT is the best
    //    option.
    //
    // 4. otherwise, Linear is selected.  Note that is this case GNAT
    //    may perform better depending on concurrent demands placed
    //    upon the data structure.
    template <typename Space, typename Concurrency, bool pauseless,
              bool = metric::is_space_v<Space>>
    struct auto_strategy;
    // {
    //     static_assert(
    //         !std::is_same_v<Space, Space>,
    //         "could not automatically determine best strategy for concurrency and space");
    // };

    // The parameters for the automatic_strategy selection are:
    //
    //   Space: the metric space for the nearest neighbor
    //   Concurrency: concurrency level desired
    //   pauseless: true if the algorithm should avoid occasional long
    //       rebuild operations.
    template <typename Space, typename Concurrency, bool pauseless = true>
    using auto_strategy_t = typename auto_strategy<Space, Concurrency, pauseless>::type;

    template <typename Space, typename Concurrency, bool pauseless>
    struct auto_strategy<Space, Concurrency, pauseless, true> {
        using type = KDTreeBatch<>;
    };

    template <typename Space, bool pauseless>
    struct auto_strategy<Space, NoThreadSafety, pauseless, false> {
        using type = GNAT<>;
    };

    template <typename Space, bool pauseless>
    struct auto_strategy<Space, ConcurrentRead, pauseless, false> {
        using type = GNAT<>;
    };

    template <typename Space>
    struct auto_strategy<Space, NoThreadSafety, false, true> {
        using type = KDTreeMedian<>;
    };

    template <typename Space>
    struct auto_strategy<Space, ConcurrentRead, false, true> {
        using type = KDTreeMedian<>;
    };

    template <typename Space, bool pauseless>
    struct auto_strategy<Space, Concurrent, pauseless, false> {
        using type = Linear;
    };
}

#endif
