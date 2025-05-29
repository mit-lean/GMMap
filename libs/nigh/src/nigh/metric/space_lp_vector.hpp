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
#ifndef NIGH_SPACE_VECTOR_LP_HPP
#define NIGH_SPACE_VECTOR_LP_HPP

#include "lp.hpp"
#include "space.hpp"
#include "../impl/lp_sum.hpp"
#include <vector>

namespace unc::robotics::nigh::metric {

    template <typename S, int p>
    struct Space<::std::vector<S>, LP<p>>
        : impl::SpaceBase<impl::Dynamic>
    {
    private:
        using Base = impl::SpaceBase<impl::Dynamic>;
    public:
        using Type = std::vector<S>;
        using Distance = S;
        using Metric = LP<p>;

        using Base::Base;

        static bool isValid(const Type& a) {
            static constexpr bool (*isfinite)(S) = &std::isfinite;
            return std::all_of(a.begin(), a.end(), isfinite);
        }

        static Distance coeff(const Type& a, std::size_t index) {
            assert(index < a.size());
            return a[index];
        }

        static Distance& coeff(Type& a, std::size_t index) {
            assert(index < a.size());
            return a[index];
        }

        Distance distance(const Type& a, const Type& b) const {
            unsigned n = Base::dimensions();
            assert(a.size() == n && b.size() == n);
            impl::LPSum<p, Distance> sum(a[0] - b[0]);
            for (unsigned i=1 ; i<n ; ++i)
                sum += a[i] - b[i];
            return sum;
        }
    };
}

#endif
