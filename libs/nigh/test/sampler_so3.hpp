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
#ifndef NIGH_TEST_IMPL_SAMPLER_SO3_HPP
#define NIGH_TEST_IMPL_SAMPLER_SO3_HPP

#include "sampler.hpp"
#include <nigh/impl/constants.hpp>
#include <nigh/metric/so3.hpp>
#include <nigh/metric/space_so3_quaternion.hpp>
#include <nigh/metric/space_so3_eigen.hpp>
#include <random>

namespace nigh_test {
    using namespace unc::robotics::nigh::metric;

    template <typename State>
    struct Sampler<State, SO3> {
        using Scalar = typename State::Scalar;
        using Metric = SO3;

        Sampler(const Space<State, Metric>&) {
        }

        template <typename RNG>
        State operator() (RNG& rng) {
            using namespace unc::robotics::nigh::impl;
            std::uniform_real_distribution<Scalar> dist01(0, 1);
            std::uniform_real_distribution<Scalar> dist2pi(0, 2*PI<Scalar>);
            Scalar a = dist01(rng);
            Scalar b = dist2pi(rng);
            Scalar c = dist2pi(rng);

            return State(
                std::sqrt(1-a)*std::sin(b),
                std::sqrt(1-a)*std::cos(b),
                std::sqrt(a)*std::sin(c),
                std::sqrt(a)*std::cos(c));
        }
    };
}

#endif

