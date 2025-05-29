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
#ifndef NIGH_METRIC_CARTESIAN_HPP
#define NIGH_METRIC_CARTESIAN_HPP

#include <tuple>
#include "metric.hpp"
#include "cartesian_state_element.hpp"

namespace unc::robotics::nigh::metric {
    template <typename ... M>
    struct Cartesian : private std::tuple<M...> {
        static_assert(sizeof...(M) > 0, "Cartesian metric must not be empty");

    private:
        using Base = std::tuple<M...>;
        Base& tuple() { return *this; }
        const Base& tuple() const { return *this; }

        template <std::size_t I>
        decltype(auto) get() { return std::get<I>(tuple()); }
        template <std::size_t I>
        decltype(auto) get() const { return std::get<I>(tuple()); }
    };

    template <std::size_t I, typename M>
    struct cartesian_element;

    template <std::size_t I, typename ... M>
    struct cartesian_element<I, Cartesian<M...>>
        : std::tuple_element<I, std::tuple<M...>> {};

    template <std::size_t I, typename M>
    using cartesian_element_t = typename cartesian_element<I, M>::type;

    template <typename ... M>
    struct is_metric<Cartesian<M...>>
        : std::bool_constant<(is_metric_v<M> && ...)>
    {
    };

    template <class C>
    struct cartesian_size;

    template <class ... M>
    struct cartesian_size<Cartesian<M...>> : std::integral_constant<std::size_t, sizeof...(M)> {};

    template <class C>
    constexpr std::size_t cartesian_size_v = cartesian_size<C>::value;
}

// tuple_element is declared as a class and as a struct in some
// libraries.  there's no way to avoid the mismatched tag warning
// without just disabling it.
#ifdef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-tags"
#endif
namespace std {
    template <std::size_t I, typename ... M>
    struct tuple_element<I, unc::robotics::nigh::metric::Cartesian<M...>>
        : std::tuple_element<I, std::tuple<M...>> {};

    template <std::size_t I, typename ... M>
    decltype(auto) get(const unc::robotics::nigh::metric::Cartesian<M...>& metric) {
        return metric.template get<I>();
    }
}
#ifdef __clang__
#pragma GCC diagnostic pop
#endif

#include "../impl/metric_specializations.hpp"

#endif
