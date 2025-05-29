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
#ifndef NIGH_IMPL_BITS_HPP
#define NIGH_IMPL_BITS_HPP

namespace unc::robotics::nigh::impl {
    
    // TODO: provide alternate implementations when compiler does not
    // have the builtin.
    
    constexpr int clz(unsigned x) {
        return __builtin_clz(x);
    }

    constexpr int clz(unsigned long x) {
        return __builtin_clzl(x);
    }
    
    constexpr int clz(unsigned long long x) {
        return __builtin_clzll(x);
    }

    constexpr int popcount(unsigned x) {
        return __builtin_popcount(x);
    }

    constexpr int popcount(unsigned long x) {
        return __builtin_popcountl(x);
    }

    constexpr int popcount(unsigned long long x) {
        return __builtin_popcountll(x);
    }

    template <typename T>
    constexpr std::enable_if_t<std::is_integral<T>::value, int>
    log2(T x) {
        return sizeof(x)*8 - 1 - clz(x);
    }
}

#endif

