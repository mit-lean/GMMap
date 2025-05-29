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
#ifndef NIGH_IMPL_ATOMIC_BS_HPP
#define NIGH_IMPL_ATOMIC_BS_HPP

#include <atomic>
#include <utility>
#include <type_traits>
#include <cstring>

namespace unc::robotics::nigh::impl {
    // Atomic 'B'yte 'S'tore to handle the BS of std::atomic<long
    // double>, but it may have other uses.  See atomic_selector<long
    // double>.
    template <typename T, typename Int, std::size_t bytes>
    class AtomicBS {
        static_assert(
            sizeof(T) <= sizeof(Int),
            "integral type must be large enough to hold target type");

        std::atomic<Int> value_;

        static Int floatToInt(T f) {
            Int i = 0;
            std::memcpy(&i, &f, bytes);
            return i;
        }

        static T intToFloat(Int i) {
            T f = 0;
            std::memcpy(&f, &i, bytes);
            return f;
        }

        // TODO: may need to add additional methods to match
        // std::atomic.
    public:
        AtomicBS() {}
        AtomicBS(T v) : value_(floatToInt(v)) {}

        T load(std::memory_order order = std::memory_order_seq_cst) const {
            return intToFloat(value_.load(order));
        }

        void store(T v, std::memory_order order = std::memory_order_seq_cst) {
            value_.store(floatToInt(v), order);
        }

        bool compare_exchange_weak(T& expected, T update, std::memory_order order) {
            Int e = floatToInt(expected);
            if (value_.compare_exchange_weak(e, floatToInt(update), order))
                return true;
            expected = intToFloat(e);
            return false;
        }

        bool compare_exchange_strong(T& expected, T update, std::memory_order order) {
            Int e = floatToInt(expected);
            if (value_.compare_exchange_strong(e, floatToInt(update), order))
                return true;
            expected = intToFloat(e);
            return false;
        }

        bool compare_exchange_weak(
            T& expected, T update,
            std::memory_order success,
            std::memory_order failure)
        {
            Int e = floatToInt(expected);
            if (value_.compare_exchange_weak(e, floatToInt(update), success, failure))
                return true;
            expected = intToFloat(e);
            return false;
        }

        bool compare_exchange_strong(
            T& expected, T update,
            std::memory_order success,
            std::memory_order failure)
        {
            Int e = floatToInt(expected);
            if (value_.compare_exchange_strong(e, floatToInt(update), success, failure))
                return true;
            expected = intToFloat(e);
            return false;
        }
    };

        // The atomic_select<T>::type selects which atomic implementation
    // to use.  Normally it resolves to std::atomic<T>
    template <typename T>
    struct atomic_selector {
        using type = std::atomic<T>;
    };

#ifdef _GLIBCXX_USE_INT128
    // We need to specially handle x86's 80-bit std::atomic<long
    // double>.  On x86 long double is an 80-bit value with a
    // sizeof==16 (128-bits).  It seems as though the unused 6 bytes
    // can spuriously change and cause CAS operations to fail.
    // Sometimes the failure occurs repeatedly.
    template <>
    struct atomic_selector<long double> {
        using limits = std::numeric_limits<long double>;

        // GCC warns about ISO C++ does not support '__int128'
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
        using int128_t = __int128;
#pragma GCC diagnostic pop


        // check the limits to see if we have the problematic x86
        // 80-bit/10-byte long double.
        //
        // digits == 64  => 8 bytes
        // exponent range = 32765   => 15 bits + 1 sign = 2 bytes
        using type = std::conditional_t<
            (sizeof(long double) == 16 &&
             limits::digits == 64 &&
             (limits::max_exponent - limits::min_exponent) == 32765),
            AtomicBS<long double, int128_t, 10>,
            std::atomic<long double>>;
    };
#endif // _GLIBCXX_USE_INT128

}
#endif
