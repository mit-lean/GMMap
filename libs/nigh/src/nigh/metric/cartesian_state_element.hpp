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
#ifndef NIGH_METRIC_CARTESIAN_STATE_ELEMENT_HPP
#define NIGH_METRIC_CARTESIAN_STATE_ELEMENT_HPP

#include <tuple>

namespace unc::robotics::nigh::metric {
    // Cartesian state handling.  Out of the box, Nigh supports any types 'T' for which
    //
    //   std::tuple_element<I, T>
    //   std::get<I>(T&)
    //
    // are defined (given a valid `I`).  Thus good candidates for a Cartesian state are:
    //
    //   std::tuple<...>
    //   std::pair<Q1, Q2>
    //   std::array<Q, N>.
    //
    // In order to support a different or custom type containing for a
    // cartesian state, there are two options.  Option 1: specialize
    // std::tuple_element and overload std::get<I>(), or Option 2:
    // specialize cartesian_state_element.

    template <std::size_t I, typename T>
    struct cartesian_state_element {
        using type = std::tuple_element_t<I, T>;

        // std::get provides 4 versions (&, const&, &&, const&&), we
        // only need const&.
        static constexpr const type& get(const T& q) {
            return std::get<I>(q);
        }

        // other libraries (e.g. mpt) may need this:
        static constexpr type& get(T& q) {
            return std::get<I>(q);
        }
    };

    template <std::size_t I, typename S>
    using cartesian_state_element_t = typename cartesian_state_element<I, S>::type;
}

#endif
