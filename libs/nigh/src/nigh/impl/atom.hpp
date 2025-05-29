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
#ifndef NIGH_IMPL_ATOM_HPP
#define NIGH_IMPL_ATOM_HPP

#include "non_atomic.hpp"
#include "atomic_bs.hpp"

namespace unc::robotics::nigh::impl {

    // TODO: use std::hardware_destructive_interference_size instead of 64 (when available)
    constexpr std::size_t cache_line_size = 64;

    // atomic_selector<T, bool> selects between atomic and non-atomic
    // versions of the same type.  In theory, with a good optimizer,
    // this allow code that does not need atomic guarantees to be
    // reordered for better (single-threaded) performance.
    template <typename T, bool atomic>
    struct atomicity_selector;

    // Atomic specialization delegates to the atomic_selector choose
    // an atomic implementation.
    template <typename T>
    struct atomicity_selector<T, true> {
        using type = typename atomic_selector<T>::type;
    };

    // Non-atomic specialization returns a NonAtomic object that
    // mimics the std::atomic interface.
    template <typename T>
    struct atomicity_selector<T, false> {
        using type = NonAtomic<T>;
    };

    // Base atomic/non-atomic implementation selector.
    template <typename T, bool atomic>
    using Atom = typename atomicity_selector<T, atomic>::type;

    inline void relax_cpu() {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("rep;nop": : :"memory");
    // alternate method:
    // asm("rep; nop" ::: "memory");
#endif
    }
}

#endif // NIGH_IMPL_ATOM_HPP
