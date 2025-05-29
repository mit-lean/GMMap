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
#ifndef NIGH_IMPL_NON_ATOMIC_HPP
#define NIGH_IMPL_NON_ATOMIC_HPP

#include <atomic>

namespace unc::robotics::nigh::impl {
    // Non-atomic class to mimic std::atomic, but ignore memory
    // ordering constraints.
    template <typename T>
    class NonAtomicBase {
    protected:
        T value_;
    public:
        NonAtomicBase() = default;
        NonAtomicBase(T value) : value_(value) {}
        NonAtomicBase(const NonAtomicBase&) = delete;

        T operator=(T desired) { return value_ = desired; }
        T operator=(T desired) volatile { return value_ = desired; };
        NonAtomicBase& operator=(const NonAtomicBase&) = delete;
        NonAtomicBase& operator=(const NonAtomicBase&) volatile = delete;

        bool is_lock_free() const { return false; }
        bool is_lock_free() const volatile { return false; }

        void store(T desired, std::memory_order order = std::memory_order_seq_cst) {
            value_ = desired;
        }
        void store(T desired, std::memory_order order = std::memory_order_seq_cst) volatile {
            value_ = desired;
        }

        T load(std::memory_order order = std::memory_order_seq_cst) const {
            return value_;
        }
        T load(std::memory_order order = std::memory_order_seq_cst) const volatile {
            return value_;
        }

        operator T () const { return value_; }
        operator T () const volatile { return value_; }

        T exchange(T desired, std::memory_order order = std::memory_order_seq_cst) {
            return std::exchange(value_, desired);
        }
        T exchange(T desired, std::memory_order order = std::memory_order_seq_cst) volatile {
            return std::exchange(value_, desired);
        }

    private:
        bool cas(T& expected, T desired) {
            assert(value_ == expected);
            value_ = desired;
            return true;
        }

    public:
        // small deviation from std::atomic (in which order matters), there is no 3 argument version.
        bool compare_exchange_weak(
            T& expected, T desired,
            std::memory_order success = std::memory_order_seq_cst,
            std::memory_order failure = std::memory_order_seq_cst)
        {
            return cas(expected, desired);
        }

                // small deviation from std::atomic (in which order matters), there is no 3 argument version.
        bool compare_exchange_strong(
            T& expected, T desired,
            std::memory_order success = std::memory_order_seq_cst,
            std::memory_order failure = std::memory_order_seq_cst)
        {
            return cas(expected, desired);
        }
    };

    // Default non-atomic class.
    template <typename T, typename Enable = void>
    class NonAtomic : public NonAtomicBase<T> {
    public:
        using NonAtomicBase<T>::NonAtomicBase;
    };

    // Specialization for integral types
    template <typename T>
    class NonAtomic<T, typename std::enable_if_t<std::is_integral_v<T>>> : public NonAtomicBase<T> {
        using NonAtomicBase<T>::value_;

    public:
        using NonAtomicBase<T>::NonAtomicBase;

        T fetch_add(T arg, std::memory_order order = std::memory_order_seq_cst) {
            return std::exchange(value_, value_ + arg);
        }

        T fetch_sub(T arg, std::memory_order order = std::memory_order_seq_cst) {
            return std::exchange(value_, value_ - arg);
        }

        T fetch_and(T arg, std::memory_order order = std::memory_order_seq_cst) {
            return std::exchange(value_, value_ & arg);
        }

        T fetch_or(T arg, std::memory_order order = std::memory_order_seq_cst) {
            return std::exchange(value_, value_ | arg);
        }

        T fetch_xor(T arg, std::memory_order order = std::memory_order_seq_cst) {
            return std::exchange(value_, value_ ^ arg);
        }

        // TODO: volatile versions

        T operator++() { return ++value_; }
        T operator++(int) { return value_++; }
        T operator--() { return --value_; }
        T operator--(int) { return value_--; }

        T operator+=(T arg) { return value_ += arg; }
        T operator-=(T arg) { return value_ -= arg; }
        T operator&=(T arg) { return value_ &= arg; }
        T operator|=(T arg) { return value_ |= arg; }
        T operator^=(T arg) { return value_ ^= arg; }
    };

    // Sepcialization for pointers
    template <typename T>
    class NonAtomic<T*> : public NonAtomicBase<T*> {
        using NonAtomicBase<T*>::value_;
    public:
        using NonAtomicBase<T*>::NonAtomicBase;

        T* fetch_add(std::ptrdiff_t arg, std::memory_order order = std::memory_order_seq_cst) {
            return std::exchange(value_, value_ + arg);
        }

        T* fetch_sub(std::ptrdiff_t arg, std::memory_order order = std::memory_order_seq_cst) {
            return std::exchange(value_, value_ - arg);
        }

        T* operator++() { return ++value_; }
        T* operator++(int) { return value_++; }
        T* operator--() { return --value_; }
        T* operator--(int) { return value_--; }

        T* operator+=(std::ptrdiff_t arg) { return value_ += arg; }
        T* operator-=(std::ptrdiff_t arg) { return value_ -= arg; }
    };
}

#endif
