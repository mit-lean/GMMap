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
#ifndef NIGH_IMPL_NEAR_SET_HPP
#define NIGH_IMPL_NEAR_SET_HPP

#include <algorithm>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace unc::robotics::nigh::impl {
    template <typename T, typename Distance>
    class Near1Set {
        std::optional<T> nearest_;
        Distance dist_{std::numeric_limits<Distance>::infinity()};

    public:
        Distance dist() const {
            return dist_;
        }

        bool full() const {
            return !!nearest_;
        }

        bool insert(const T& n, Distance d) {
            if (d < dist_) {
                dist_ = d;
                nearest_ = n;
                return true;
            }

            return false;
        }

        template <typename Iter, typename Fn>
        void insert(Iter first, Iter last, Fn dist) {
            for (Iter it = first ; it != last ; ++it)
                insert(*it, dist(*it));
        }

        std::optional<std::pair<T, Distance>> result() {
            if (nearest_)
                return std::pair<T, Distance>(*nearest_, dist_);

            return {};
        }

        std::optional<T> result(Distance *dist) {
            if (dist)
                *dist = nearest_ ? dist_ : std::numeric_limits<Distance>::quiet_NaN();

            // Should be able to do this:
            //
            //   return nearest_;
            //
            // However libstdc++ 8 (as of writing this comment) is
            // giving a message about std::optional's _M_payload
            // having a non-trivial copy constructor.  So instead, we
            // manually split it apart:
            if (nearest_) {
                return *nearest_;
            } else {
                return {};
            }
        }
    };

    template <typename Tuple, typename Distance, typename Allocator>
    class NearKSet {
        static const std::size_t kDistanceIndex =
            std::is_same_v<Distance, std::remove_cv_t<std::tuple_element_t<1, Tuple>>> ? 1 : 0;

        static_assert(
            std::is_same_v<Distance, std::remove_cv_t<std::tuple_element_t<kDistanceIndex, Tuple>>>,
            "nearest vector element type must have a distance element");

        std::vector<Tuple, Allocator>& set_;
        std::size_t k_;
        Distance dist_;

        struct Compare {
            bool operator() (const Tuple& a, const Tuple& b) const {
                return std::get<kDistanceIndex>(a) < std::get<kDistanceIndex>(b);
            }
        };

        template <typename T>
        void emplace(const T& n, Distance d) {
            if constexpr (kDistanceIndex == 1) {
                set_.emplace_back(n, d);
            } else {
                set_.emplace_back(d, n);
            }
        }

    public:
        NearKSet(
            std::vector<Tuple, Allocator>& set,
            std::size_t k,
            Distance dist)
            : set_(set), k_(k), dist_(dist)
        {
            set_.clear();
        }

        Distance dist() const {
            return dist_;
        }

        std::size_t size() const {
            return set_.size();
        }

        bool full() const {
            return set_.size() == k_;
        }

        template <typename T>
        bool insert(const T& n, Distance d) {
            if (d > dist_)
                return false;

            emplace(n, d);

            if (set_.size() <= k_) {
                if (set_.size() < k_)
                    return false;
                std::make_heap(set_.begin(), set_.end(), Compare{});
            } else {
                std::pop_heap(set_.begin(), set_.end(), Compare{});
                set_.pop_back();
            }

            dist_ = std::get<kDistanceIndex>(set_[0]);
            return true;
        }

        template <typename Iter, typename Fn>
        void insert(Iter first, Iter last, Fn dist) {
            Iter it = first;
            if (set_.size() < k_) {
                for (Distance d ; it != last ; ++it) {
                    if ((d = dist(*it)) <= dist_) {
                        emplace(*it, d);
                        if (set_.size() == k_) {
                            std::make_heap(set_.begin(), set_.end(), Compare{});
                            dist_ = std::get<kDistanceIndex>(set_[0]);
                            ++it;
                            goto heaped;
                        }
                    }
                }
            } else {
              heaped:
                for (Distance d ; it != last ; ++it) {
                    if ((d = dist(*it)) <= dist_) {
                        emplace(*it, d);
                        std::pop_heap(set_.begin(), set_.end(), Compare{});
                        set_.pop_back();
                        dist_ = std::get<kDistanceIndex>(set_[0]);
                    }
                }
            }
        }

        void sort() {
            if (set_.size() < k_) {
                std::sort(set_.begin(), set_.end(), Compare{});
            } else {
                std::sort_heap(set_.begin(), set_.end(), Compare{});
            }
        }
    };
}

#endif // NIGH_IMPL_NEAR_SET_HPP
