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
#ifndef NIGH_TEST_IMPL_BOX_SAMPLER_HPP
#define NIGH_TEST_IMPL_BOX_SAMPLER_HPP

#include <random>
#include <array>
#include <cassert>
#include <Eigen/Dense>

namespace nigh_test {

    template <typename Scalar, int dim>
    struct BoxSamplerBase {
        std::uniform_real_distribution<Scalar> dist_;
        static constexpr unsigned dimensions = dim;

        BoxSamplerBase(unsigned d, Scalar min, Scalar max)
            : dist_(min, max)
        {
            assert(d == dim);
        }
    };

    template <typename Scalar>
    struct BoxSamplerBase<Scalar, -1> {
        std::uniform_real_distribution<Scalar> dist_;
        unsigned dimensions;

        BoxSamplerBase(unsigned dim, Scalar min, Scalar max)
            : dist_(min, max), dimensions(dim)
        {
        }
    };

    template <typename State, int dim, typename Enabled = void>
    struct BoxSampler : BoxSamplerBase<
        typename unc::robotics::nigh::metric::Space<State, unc::robotics::nigh::metric::L2>::Distance,
        unc::robotics::nigh::metric::Space<State, unc::robotics::nigh::metric::L2>::kDimensions>
    {
        using Base = BoxSamplerBase<
            typename unc::robotics::nigh::metric::Space<State, unc::robotics::nigh::metric::L2>::Distance,
            unc::robotics::nigh::metric::Space<State, unc::robotics::nigh::metric::L2>::kDimensions>;

        using Base::Base;

        template <typename RNG>
        State operator() (RNG& rng) {
            State q;
            for (unsigned i=0 ; i<Base::dimensions ; ++i)
                q[i] = Base::dist_(rng);
            return q;
        }
    };

    template <typename S, int dim>
    struct BoxSampler<S, dim, std::enable_if_t<std::is_floating_point_v<S>>>
        : BoxSamplerBase<S, dim>
    {
        using Scalar = S;
        using Base = BoxSamplerBase<Scalar, dim>;
        using Base::Base;

        template <typename RNG>
        Scalar operator() (RNG& rng) {
            return Base::dist_(rng);
        }
    };

    template <typename Scalar, int dim>
    struct BoxSampler<std::vector<Scalar>, dim> : BoxSamplerBase<Scalar, dim> {
        using Base = BoxSamplerBase<Scalar, dim>;
        using State = std::vector<Scalar>;
        using Base::Base;

        template <typename RNG>
        State operator() (RNG& rng) {
            State q;
            q.reserve(Base::dimensions);
            for (unsigned i=0 ; i<Base::dimensions ; ++i)
                q.push_back(Base::dist_(rng));
            return q;
        }
    };

    template <typename Scalar, std::size_t n, int dim>
    struct BoxSampler<std::array<Scalar, n>, dim> : BoxSamplerBase<Scalar, int(n)> {
        using Base = BoxSamplerBase<Scalar, int(n)>;
        using State = std::array<Scalar, n>;

        using Base::Base;

        template <typename RNG>
        State operator() (RNG& rng) {
            State q;
            for (std::size_t i=0 ; i<n ; ++i)
                q[i] = Base::dist_(rng);
            return q;
        }
    };

    template <typename State, int dim>
    struct EigenBoxSampler
        : BoxSamplerBase<typename State::Scalar, dim>
    {
        static_assert(
            State::RowsAtCompileTime != Eigen::Dynamic ||
            State::ColsAtCompileTime != Eigen::Dynamic,
            "unsupported: both dimensions are dynamic");

        using Base = BoxSamplerBase<typename State::Scalar, dim>;
        using Base::Base;

        template <typename RNG>
        State operator() (RNG& rng) {
            State q(
                State::RowsAtCompileTime == Eigen::Dynamic ? Base::dimensions : State::RowsAtCompileTime,
                State::ColsAtCompileTime == Eigen::Dynamic ? Base::dimensions : State::ColsAtCompileTime);
            for (unsigned i=0 ; i<Base::dimensions ; ++i)
                q[i] = Base::dist_(rng);
            return q;
        }
    };

    template <typename S, int R, int C, int opts, int maxR, int maxC, int dim>
    struct BoxSampler<Eigen::Matrix<S, R, C, opts, maxR, maxC>, dim>
        : EigenBoxSampler<Eigen::Matrix<S, R, C, opts, maxR, maxC>, dim>
    {
        using Scalar = S;
        using EigenBoxSampler<Eigen::Matrix<S, R, C, opts, maxR, maxC>, dim>::EigenBoxSampler;
    };

    template <typename S, int R, int C, int opts, int maxR, int maxC, int dim>
    struct BoxSampler<Eigen::Array<S, R, C, opts, maxR, maxC>, dim>
        : EigenBoxSampler<Eigen::Array<S, R, C, opts, maxR, maxC>, dim>
    {
        using Scalar = S;
        using EigenBoxSampler<Eigen::Array<S, R, C, opts, maxR, maxC>, dim>::EigenBoxSampler;
    };
}

#endif
