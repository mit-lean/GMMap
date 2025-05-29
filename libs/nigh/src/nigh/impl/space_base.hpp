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
#ifndef NIGH_METRIC_IMPL_SPACE_BASE_HPP
#define NIGH_METRIC_IMPL_SPACE_BASE_HPP

#include <cassert>
#include <limits>

namespace unc::robotics::nigh::impl {
    // This is equivalent to Eigen::Dynamic.  It is used when the
    // state storage type is not fixed at compile time.
    constexpr int Dynamic = -1;

    template <int dim>
    struct SpaceBase {
        static_assert(dim > 0, "dimensions must be positive or dynamic");

        static constexpr bool dynamic = false;
        static constexpr int kDimensions = dim;

        constexpr unsigned dimensions() const {
            return dim;
        }
    };

    template <>
    struct SpaceBase<Dynamic> {
        static constexpr bool dynamic = true;
        static constexpr int kDimensions = Dynamic;
    private:
        unsigned dimensions_;
    public:
        explicit SpaceBase(unsigned dimensions)
            : dimensions_(dimensions)
        {
            // dimensions must be positive
            assert(dimensions > 0);

            // something likely went wrong if dimensions is this big...
            assert(dimensions < static_cast<unsigned>(std::numeric_limits<int>::max()));
        }

        constexpr unsigned dimensions() const {
            return dimensions_;
        }
    };
}

#endif
