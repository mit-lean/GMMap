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
#ifndef NIGH_IMPL_REGION_LP_HPP
#define NIGH_IMPL_REGION_LP_HPP

#include "region.hpp"
#include "box_distance.hpp"
#include "lp_sum.hpp"
#include <Eigen/Dense>
#include <array>
#include <type_traits>

namespace unc::robotics::nigh::impl {

    template <typename Key, int dim, int p, typename Concurrency>
    class LPRegion {
        static constexpr int kDim = (dim == -1 ? Eigen::Dynamic : dim);
        using Space = metric::Space<Key, metric::LP<p>>;
        using Distance = typename Space::Distance;

        Eigen::Matrix<Distance, kDim, 1> min_;
        Eigen::Matrix<Distance, kDim, 1> max_;

    public:
        LPRegion() = default;
        
        LPRegion(const LPRegion& region)
            : min_(region.min_)
            , max_(region.max_)
        {
        }

        template <typename Traversal>
        LPRegion(const Space&, const Traversal&, const Key& q)
            : min_(q), max_(q)
        {
        }
        // LPRegion(const LPMetric<p>&, const T& q) : min_(q), max_(q) {}

        template <typename D>
        void init(const Space&, const Eigen::MatrixBase<D>& q) {
            min_ = q;
            max_ = q;
        }
        
        template <typename D>
        void grow(const Space&, const Eigen::MatrixBase<D>& q) {
            min_ = min_.cwiseMin(q);
            max_ = max_.cwiseMax(q);
        }

        template <typename D>
        void init(const Space& space, const Eigen::ArrayBase<D>& q) {
            init(space, q.matrix());
        }

        template <typename D>
        void grow(const Space& space, const Eigen::ArrayBase<D>& q) {
            grow(space, q.matrix());
        }

        template <std::size_t N>
        std::enable_if_t<N == dim>
        init(const Space& space, const std::array<Distance, N>& q) {
            init(space, Eigen::Map<const Eigen::Matrix<Distance, dim, 1>>(q.data()));
        }

        template <std::size_t N>
        std::enable_if_t<N == dim>
        grow(const Space& space, const std::array<Distance, N>& q) {
            grow(space, Eigen::Map<const Eigen::Matrix<Distance, dim, 1>>(q.data()));
        }

        template <typename Alloc>
        void init(const Space& space, const std::vector<Distance, Alloc>& q) {
            init(space, Eigen::Map<const Eigen::Matrix<Distance, dim, 1>>(q.data(), q.size()));
        }

        template <typename Alloc>
        void grow(const Space& space, const std::vector<Distance, Alloc>& q) {
            grow(space, Eigen::Map<const Eigen::Matrix<Distance, dim, 1>>(q.data(), q.size()));
        }

        constexpr unsigned dimensions() const {
            return min_.size();
        }

        Distance selectAxis(unsigned *axis) const {
            return (max_ - min_).maxCoeff(axis);
        }

        Distance distTo(const Key& key) const {
            return boxDistance<p>(min_, max_, key);
        }

        Distance min(int index) const { return min_[index]; }
        Distance max(int index) const { return max_[index]; }
    };

    template <typename Key, int dim, int p>
    class LPRegion<Key, dim, p, Concurrent> {
        static constexpr int kDim = dim;
        using Space = metric::Space<Key, metric::LP<p>>;
        using Distance = typename Space::Distance;

        std::array<Atom<Distance, true>, kDim> min_;
        std::array<Atom<Distance, true>, kDim> max_;

    public:
        LPRegion() = default;
        
        LPRegion(const LPRegion& other) {
            for (int i=0 ; i<kDim ; ++i) {
                min_[i].store(other.min_[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
                max_[i].store(other.max_[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
            }
        }

        template <typename Traversal>
        LPRegion(const Space&, const Traversal&, const Key& q) {
            // LPRegion(const metric::LP<p>&, const Q& q) {
            for (int i=0 ; i<kDim ; ++i) {
                min_[i].store(Space::coeff(q, i), std::memory_order_relaxed);
                max_[i].store(Space::coeff(q, i), std::memory_order_relaxed);
            }
        }

        constexpr unsigned dimensions() const {
            return kDim;
        }

        Distance min(int index) const { return min_[index].load(std::memory_order_relaxed); }
        Distance max(int index) const { return max_[index].load(std::memory_order_relaxed); }

        template <typename State>
        void grow(const Space&, const State& q) {
            for (int i=0 ; i<kDim ; ++i) {
                Distance v = Space::coeff(q, i);
                for (Distance e = min(i) ; v < e && !min_[i].compare_exchange_weak(e, v, std::memory_order_relaxed); );
                for (Distance e = max(i) ; v > e && !max_[i].compare_exchange_weak(e, v, std::memory_order_relaxed); );
                // std::cout << "region " << i << ": " << min_[i].load() << " to " << max_[i].load() << std::endl;
            }
        }

        Distance selectAxis(unsigned *axis) const {
            Distance dBest = max(0) - min(0);
            *axis = 0;

            for (int i=1 ; i<kDim ; ++i) {
                Distance dx = max(i) - min(i);
                if (dx > dBest) {
                    dBest = dx;
                    *axis = i;
                }
            }

            return dBest;
        }

        Distance distTo(const Key& key) const {
            Eigen::Matrix<Distance, kDim, 1> d;
            for (int i=0 ; i<kDim ; ++i)
                d[i] = std::max({
                        min_[i].load(std::memory_order_relaxed) - Space::coeff(key, i),
                        Space::coeff(key, i) - max_[i].load(std::memory_order_relaxed),
                        static_cast<Distance>(0)
                    });;
            return d.template lpNorm<p>();
        }
    };

    template <typename Key, int p>
    class LPRegion<Key, -1, p, Concurrent> {
        using Space = metric::Space<Key, metric::LP<p>>;
        using Distance = typename Space::Distance;

        // In order to put std::atomic into a vector, we need to
        // provide a move constructor.  Since we only expect the move
        // constructor to be used during initialization of the region,
        // we can implement it with relaxed memory order.
        struct Coeff : Atom<Distance, true> {
            using Base = Atom<Distance, true>;
            using Base::Base;
            Coeff(const Coeff& other) : Base(other.load(std::memory_order_relaxed)) {}
            Coeff(Coeff&& other) : Base(other.load(std::memory_order_relaxed)) {}
        };

        std::vector<Coeff> min_;
        std::vector<Coeff> max_;

    public:
        LPRegion() = default;
        
        LPRegion(const LPRegion& other)
            : min_(other.min_)
            , max_(other.max_)
        {
        }

        LPRegion(const Space&, LPRegion&& other)
            : min_(std::move(other.min_))
            , max_(std::move(other.max_))
        {
        }

        template <typename Traversal>
        LPRegion(const Space& space, const Traversal&, const Key& q) {
            unsigned dim = space.dimensions();
            min_.reserve(dim);
            max_.reserve(dim);
            for (unsigned i=0 ; i<dim ; ++i) {
                min_.emplace_back(Space::coeff(q, i));
                max_.emplace_back(Space::coeff(q, i));
            }
        }

        unsigned dimensions() const {
            return min_.size();
        }

        Distance min(int index) const { return min_[index].load(std::memory_order_relaxed); }
        Distance max(int index) const { return max_[index].load(std::memory_order_relaxed); }

        template <typename State>
        void grow(const Space& space, const State& q) {
            unsigned dim = space.dimensions();
            assert(dim == min_.size());
            for (unsigned i=0 ; i<dim ; ++i) {
                Distance v = Space::coeff(q, i);
                for (Distance e = min(i) ; v < e && !min_[i].compare_exchange_weak(e, v, std::memory_order_relaxed); );
                for (Distance e = max(i) ; v > e && !max_[i].compare_exchange_weak(e, v, std::memory_order_relaxed); );
                // std::cout << "region " << i << ": " << min(i) << " to " << max(i) << std::endl;
            }
        }

        Distance selectAxis(unsigned *axis) const {
            int dimensions = min_.size();
            Distance dBest = max(0) - min(0);
            *axis = 0;

            for (int i=1 ; i<dimensions ; ++i) {
                Distance dx = max(i) - min(i);
                if (dx > dBest) {
                    dBest = dx;
                    *axis = i;
                }
            }

            return dBest;
        }

        Distance distTo(const Key& key) const {
            std::size_t dimensions = min_.size();
            impl::LPSum<p, Distance> sum(std::max({Distance(0), min(0) - key[0], key[0] - max(0)}));
            for (std::size_t i=1 ; i<dimensions ; ++i)
                sum += std::max({Distance(0), min(i) - key[i], key[i] - max(i)});
            return sum;
        }
    };

    template <typename Key, int p, typename Concurrency>
    class Region<Key, metric::LP<p>, Concurrency>
        : public LPRegion<Key, metric::Space<Key, metric::LP<p>>::kDimensions, p, Concurrency>
    {
    public:
        using LPRegion<Key, metric::Space<Key, metric::LP<p>>::kDimensions, p, Concurrency>::LPRegion;
    };

}

#endif
