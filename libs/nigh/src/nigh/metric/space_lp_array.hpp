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
#ifndef NIGH_SPACE_ARRAY_LP_HPP
#define NIGH_SPACE_ARRAY_LP_HPP

#include "lp.hpp"
#include "space.hpp"
#include "../impl/space_base.hpp"
#include "../impl/lp_sum.hpp"
#include <array>
#include <algorithm>

namespace unc::robotics::nigh::metric {

    template <typename S, std::size_t n, int p>
    struct Space<std::array<S, n>, metric::LP<p>>
        : impl::SpaceBase<unsigned(n)>
    {
        using Type = std::array<S, n>;
        using Distance = S;
        using Metric = metric::LP<p>;

        static bool isValid(const Type& a) {
            static constexpr bool (*isfinite)(S) = &std::isfinite;
            return std::all_of(a.begin(), a.end(), isfinite);
        }

        static Distance coeff(const Type& a, std::size_t i) {
            assert(i < n);
            return a[i];
        }

        static Distance& coeff(Type& a, std::size_t i) {
            assert(i < n);
            return a[i];
        }

        Distance distance(const Type& a, const Type& b) const {
            impl::LPSum<p, Distance> sum(a[0] - b[0]);
            for (std::size_t i=1 ; i<n ; ++i)
                sum += a[i] - b[i];
            return sum;
        }
    };
}

#endif
