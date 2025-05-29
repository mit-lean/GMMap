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
#ifndef NIGH_IMPL_K_CENTERS_HPP
#define NIGH_IMPL_K_CENTERS_HPP

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <iostream> // TODO: REMOVE

namespace unc::robotics::nigh::impl {
    template <typename T, typename Distance, unsigned maxK, unsigned maxDataSize>
    class KCenters {
        Eigen::Matrix<Distance, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, maxDataSize, maxK> dists_;
        std::size_t numCenters_;
        std::array<unsigned, maxK> centers_;

    public:

        Distance dist(int i, int j) const {
            return dists_(i, j);
        }

        std::size_t numCenters() const {
            return numCenters_;
        }

        unsigned center(std::size_t i) const {
            assert(i < numCenters_);
            return centers_[i];
        }

        template <typename RNG, typename DistFn>
        void compute(const std::vector<T>& data, unsigned k, RNG& rng, const DistFn& distFn) {
            assert(k <= maxK);

            Eigen::Matrix<Distance, Eigen::Dynamic, 1, Eigen::AutoAlign, maxDataSize, 1> minDist(data.size(), 1);
            minDist.fill(std::numeric_limits<Distance>::infinity());
            numCenters_ = 0;

            dists_.resize(data.size(), k);

            std::uniform_int_distribution<unsigned> uniformInt(0, data.size() - 1);
            centers_[numCenters_++] = uniformInt(rng);

            for (unsigned i = 1 ; i<k ; ++i) {
                unsigned center = centers_[i-1];
                Distance maxDist = -std::numeric_limits<Distance>::infinity();
                unsigned int maxIndex = 0;
                for (unsigned j = 0 ; j<data.size() ; ++j) {
                    if ((dists_(j, i-1) = distFn(data[j], data[center])) < minDist[j])
                        minDist[j] = dists_(j, i-1);
                    if (minDist[j] > maxDist)
                        maxDist = minDist[maxIndex = j];
                }

                if (maxDist < std::numeric_limits<Distance>::epsilon())
                    break;

                centers_[numCenters_++] = maxIndex;
            }

            unsigned i = numCenters_ - 1;
            unsigned center = centers_[i];
            for (unsigned j=0 ; j<data.size() ; ++j)
                dists_(j, i) = distFn(data[j], data[center]);
        }
    };
}

#endif
