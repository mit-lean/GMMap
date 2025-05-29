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
#ifndef NIGH_METRIC_IMPL_SCALED_SPACE_BASE_HPP
#define NIGH_METRIC_IMPL_SCALED_SPACE_BASE_HPP

#include <ratio>

namespace unc::robotics::nigh::impl {
    template <typename S, typename M, typename W>
    struct ScaledSpaceBase;

    template <typename S, typename M, std::intmax_t num, std::intmax_t den>
    struct ScaledSpaceBase<S, M, std::ratio<num, den>>
        : Space<S, M>
    {
    private:
        using Base = Space<S, M>;
    public:
        using Distance = typename Base::Distance;
        using Base::Base;

        using Base::dimensions;

        constexpr Distance weight() const {
            return static_cast<Distance>(num) / static_cast<Distance>(den);
        }
    };

    template <typename S, typename M>
    struct ScaledSpaceBase<S, M, void>
        : Space<S, M>
    {
    private:
        using Base = Space<S, M>;
    public:
        // TODO: consider using std::common_type_t<typename
        // Base::Distance, W> for the distance type.  This would mean
        // that scaling by a double would result in an underlying
        // floating point space being cast to a double.  It is
        // unlikely that this will provide any benefit.
        using Distance = typename Base::Distance;
    private:
        Distance weight_;

    public:
        template <typename ... Args>
        ScaledSpaceBase(Distance w, Args&& ... args)
            : Base(std::forward<Args>(args)...)
            , weight_(w)
        {
        }

        using Base::dimensions;

        constexpr const Space<S, M>& base() const {
            return *this;
        }

        constexpr Distance weight() const {
            return weight_;
        }
    };
}

#endif
