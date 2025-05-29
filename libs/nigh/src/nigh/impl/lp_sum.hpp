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
#ifndef NIGH_METRIC_IMPL_LP_SUM_HPP
#define NIGH_METRIC_IMPL_LP_SUM_HPP

#include <cmath>
#include <algorithm>

namespace unc::robotics::nigh::impl {
    template <int p, typename Scalar>
    class LPSum {
        static_assert(p > 0, "p must be postitive or -1 for infinity");
        Scalar sum_;

    public:
        LPSum(Scalar d)
            : sum_(std::pow(std::abs(d), p))
        {
        }

        LPSum& operator += (Scalar d) {
            sum_ += std::pow(std::abs(d), p);
            return *this;
        }

        operator Scalar () const {
            return std::pow(sum_, 1/Scalar(p));
        }
    };

    template <typename Scalar>
    class LPSum<1, Scalar> {
        Scalar sum_;

    public:
        LPSum(Scalar d) : sum_(std::abs(d)) {}
        LPSum& operator += (Scalar d) {
            sum_ += std::abs(d);
            return *this;
        }
        operator Scalar () const {
            return sum_;
        }
    };

    template <typename Scalar>
    class LPSum<-1, Scalar> : public LPSum<1, Scalar> {
        using Base = LPSum<1, Scalar>;
        using Base::Base;

        LPSum& operator += (Scalar d) override {
            Base::sum_ = std::max(Base::sum_, std::abs(d));
            return *this;
        }
    };

    template <typename Scalar>
    class LPSum<2, Scalar> {
        Scalar sum_;

    public:
        LPSum(Scalar d) : sum_(d*d) {}
        LPSum& operator += (Scalar d) {
            sum_ += d*d;
            return *this;
        }
        operator Scalar () const {
            return std::sqrt(sum_);
        }
    };
}

#endif
