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
#ifndef NIGH_TEST_IMPL_SAMPLER_CARTESIAN_HPP
#define NIGH_TEST_IMPL_SAMPLER_CARTESIAN_HPP

#include "sampler.hpp"
#include <nigh/metric/space_cartesian.hpp>
#include <nigh/se3_space.hpp>

namespace nigh_test {
    using namespace unc::robotics::nigh::metric;

    template <typename State, typename Metric, typename Indices>
    struct CartesianSampler;

    template <std::size_t I, typename S, typename M>
    using cartesian_sampler_element_t = Sampler<
        cartesian_state_element_t<I, S>,
        cartesian_element_t<I, M>>;

    template <typename State, typename Metric, std::size_t ... I>
    struct CartesianSampler<State, Metric, std::index_sequence<I...>>
        : std::tuple<cartesian_sampler_element_t<I, State, Metric>...>
    {
        static_assert(sizeof...(I) > 0, "empty cartesian metric");
        using Base = std::tuple<cartesian_sampler_element_t<I, State, Metric>...>;

        CartesianSampler(const Space<State, Metric>& space)
            : Base(cartesian_sampler_element_t<I, State, Metric>(space.template get<I>())...)
        {
        }

        template <typename RNG>
        State operator() (RNG& rng){
            State q;
            ((std::get<I>(q) = std::get<I>(*this)(rng)), ...);
            return q;
        }
    };

    template <typename State, typename ... M>
    struct Sampler<State, Cartesian<M...>>
        : CartesianSampler<State, Cartesian<M...>, std::index_sequence_for<M...>>
    {
        using Base = CartesianSampler<State, Cartesian<M...>, std::index_sequence_for<M...>>;
        using Base::Base;
    };

}

#endif

