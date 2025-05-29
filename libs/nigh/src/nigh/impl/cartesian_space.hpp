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
#ifndef NIGH_METRIC_IMPL_CARTESIAN_SPACE_HPP
#define NIGH_METRIC_IMPL_CARTESIAN_SPACE_HPP

#include "../metric/cartesian.hpp"
#include "../metric/cartesian_state_element.hpp"
#include <tuple>

namespace unc::robotics::nigh::impl {
    template <std::size_t I, typename S, typename M>
    using cartesian_space_element = Space<
        cartesian_state_element_t<I, S>,
        cartesian_element_t<I, M>>;

    template <typename S, typename M, typename Indicies>
    struct CartesianSpace;

    template <typename S, typename M, std::size_t ... I>
    struct CartesianSpace<S, M, std::index_sequence<I...>>
        : private std::tuple<cartesian_space_element<I, S, M>...>
    {
        static_assert(sizeof...(I) > 0, "empty cartesian metric");

    private:
        using Base = std::tuple<cartesian_space_element<I, S, M>...>;

        constexpr const Base& tuple() const {
            return *this;
        }
    public:
        using Type = S;
        using Metric = M;

        using Distance = decltype(
            (std::declval<typename cartesian_space_element<I, S, M>::Distance>() + ...));

        CartesianSpace() {
        }
        
        template <typename ... Args>
        explicit CartesianSpace(Args&& ... args)
            : Base(std::forward<Args>(args)...)
        {
        }

        static bool isValid(const Type& a) {
            return (std::tuple_element_t<I, Base>::isValid(
                        cartesian_state_element<I, Type>::get(a)) && ...);
        }

        constexpr unsigned dimensions() const {
            return (std::get<I>(tuple()).dimensions() + ...);
        }

        Distance distance(const Type& a, const Type& b) const {
            return (std::get<I>(tuple()).distance(
                        cartesian_state_element<I, S>::get(a),
                        cartesian_state_element<I, S>::get(b)) + ...);
        }

        template <std::size_t J>
        decltype(auto) get() { return std::get<J>(tuple()); }
        template <std::size_t J>
        decltype(auto) get() const { return std::get<J>(tuple()); }
    };
}

namespace std {
    template <std::size_t I, typename S, typename M, typename J>
    decltype(auto) get(const unc::robotics::nigh::impl::CartesianSpace<S,M,J>& space) {
        return space.template get<I>();
    }
}


#endif
