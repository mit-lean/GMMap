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
#ifndef NIGH_IMPL_LOCKED_NEAREST_HPP
#define NIGH_IMPL_LOCKED_NEAREST_HPP

#include <shared_mutex>
#include <vector>

namespace unc::robotics::nigh::impl {
    template <typename Base>
    class LockedNearest : Base {
        mutable std::shared_mutex mutex_;

        using ReadLock = std::shared_lock<std::shared_mutex>;
        using WriteLock = std::unique_lock<std::shared_mutex>;

    public:
        using Type = typename Base::Type;
        using Space = typename Base::Space;
        using Key = typename Base::Key;
        using Metric = typename Base::Metric;
        using Distance = typename Base::Distance;
        using concurrency_type = typename Base::concurrency_type;

        // explicit move and copy constructors are needed since we
        // cannot (and do not want to) move the mutex_.

        template <typename ... Args>
        LockedNearest(Args&& ... args)
            : Base(std::forward<Args>(args)...)
        {
        }

        using Base::metricSpace;

        std::size_t size() const {
            ReadLock lock(mutex_);
            return Base::size();
        }

        void clear() const {
            WriteLock lock(mutex_);
            Base::clear();
        }

        template <typename ... Args>
        void insert(Args&& ... args) {
            WriteLock lock(mutex_);
            Base::insert(std::forward<Args>(args)...);
        }

        template <typename ... Args>
        decltype(auto) nearest(Args&& ... args) const {
            ReadLock lock(mutex_);
            return Base::nearest(std::forward<Args>(args)...);
        }

        template <typename Fn>
        void visit(const Fn& fn) const {
            ReadLock lock(mutex_);
            Base::visit(fn);
        }

        decltype(auto) list() const {
            ReadLock lock(mutex_);
            return Base::list();
        }
    };
}

#endif
