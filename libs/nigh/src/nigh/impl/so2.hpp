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
#ifndef NIGH_METRIC_IMPL_SO2_HPP
#define NIGH_METRIC_IMPL_SO2_HPP

#include <type_traits>
#include <cmath>
#include <Eigen/Dense>
#include "constants.hpp"

namespace unc::robotics::nigh::impl::so2 {
    template <typename S>
    std::enable_if_t<std::is_floating_point_v<S>, S>
    angularDistance(S a, S b) {
        S d = std::fmod(std::abs(a - b), impl::PI<S> * 2);
        return std::min(d, impl::PI<S>*2 - d);
    }

    template <int p, typename A, typename B>
    std::common_type_t<typename A::Scalar, typename B::Scalar>
    angularDistance(const Eigen::ArrayBase<A>& a, const Eigen::ArrayBase<B>& b) {
        using Scalar = std::common_type_t<typename A::Scalar, typename B::Scalar>;
        Eigen::Array<Scalar, A::RowsAtCompileTime, A::ColsAtCompileTime> r = (a - b).cwiseAbs();
        r -= (r / (2*impl::PI<Scalar>)).floor() * (2*impl::PI<Scalar>);
        r = (r < -impl::PI<Scalar>).select(r + 2*impl::PI<Scalar>, r);
        r = (r >  impl::PI<Scalar>).select(r - 2*impl::PI<Scalar>, r);
        return r.matrix().template lpNorm<p>();
    }

    template <int p, typename A, typename B>
    auto angularDistance(const Eigen::MatrixBase<A>& a, const Eigen::MatrixBase<B>& b) {
        return angularDistance<p>(a.array(), b.array());
    }

        // Returns the argument bound to the range -PI..PI
    template <typename Scalar>
    std::enable_if_t<std::is_floating_point_v<Scalar>, Scalar>
    bound(Scalar a) {
        if ((a = std::fmod(a, PI<Scalar>*2)) <= -PI<Scalar>)
            return a + PI<Scalar>*2;
        return (a > PI<Scalar>) ? a - PI<Scalar>*2 : a;
    }

    // computes the counter-clockwise distance from a to b
    template <typename Scalar>
    Scalar ccwDist(Scalar a, Scalar b) {
        Scalar d = std::fmod(b - a, PI<Scalar>*2);
        return d < 0 ? d + PI<Scalar>*2 : d;
    }

    // This method is exactly the same ad ccwDist except that two
    // overlapping values are considered 2*PI instead of 0.
    template <typename Scalar>
    Scalar ccwRange(Scalar a, Scalar b) {
        Scalar d = std::fmod(b - a, PI<Scalar>*2);
        return d <= 0 ? d + PI<Scalar>*2 : d;
    }

    template <typename Scalar>
    Scalar antipode(Scalar a) {
        return a <= 0 ? (a + PI<Scalar>) : (a - PI<Scalar>);
    }

    // Finds a split on a set of SO(2) values normalized to the range
    // [-pi,pi].  The split divides the values into two evenly sized
    // (_N/2) sets, and maximizes the distance of the bounds to the
    // split.
    template <typename Iter>
    auto split(Iter first, Iter last) {
        using Scalar = typename std::iterator_traits<Iter>::value_type;
        std::size_t n = std::distance(first, last);
        assert(n > 1); // can only split more than 1 element

        // The loop finds the `i` at which half elements are to the
        // left of the line bisecting line it define.  To find it we
        // test for `j = i + N/2`, that the ccw distance to `[j] < pi`
        // and `[j+1] > pi`.  The "best" split is the one that
        // maximizes the distance between the points closest to the
        // split, thus `[i]`, `[i+1]`, `[j]`, and `[j+1]`.

        // distances: pi-D(i,j), D(i,j+1)-pi, D(i,i+1)
        //
        //        i
        //        |@@@
        //        |@@@@
        //   -----X@@@@
        //   ####/|\@@@
        //    ##/ | \@
        //     j    j+1

        //          /
        // j+1 ----X
        //     ###/|\     |
        //     ##/ |@\    |
        //     #/  |@@\   |
        //     i       j

        //        j+1 j
        //      ###| /@
        //     ####|/@@@
        //     ####X@@@@
        //     ###/ \@@@
        //      #/   \@
        //      i

        std::sort(first, last);
        Scalar dBest = -1, split = 0;
        Iter i1 = first;
        Iter j1 = first + n/2;
        do {
            Iter i0 = i1;
            Iter j0 = j1;
            if (++i1 == last) i1 = first;
            if (++j1 == last) j1 = first;
            Scalar d0 = PI<Scalar> - so2::ccwDist(*i0, *j0); // vals[i], vals[j%n]);
            Scalar d1 = PI<Scalar> - so2::ccwDist(*j1, *i0); // vals[(j+1)%n], vals[i]);

            if (d0 >= 0 && d1 >= 0) {
                Scalar di = so2::ccwDist(*i0, *i1); // vals[i], vals[(i+1)%n]);
                // Scalar split = vals[i] + std::min(di, d1) * 0.5;

                Scalar range = 2*PI<Scalar> - (d0+d1);

                if (range < PI<Scalar>) {
                    // The range of values is less than half a circle.
                    // split halfway between i0 and i1
                    if (range > dBest) {
                        dBest = range;
                        split = *i0 + di * 0.5;
                    }
                } else {
                    // The range of values is more than half a circle
                    // split considering both sides of split plane.
                    // we cannot split halfway between i0 and i1 since
                    // it could result in moving j1 to the other side
                    // of the split.
                    //
                    // An easy split to do would be i0 + min(di,d1)/2,
                    // since that would be halfway from the split to
                    // the next bound.  This is also guaranteed to be
                    // in the bounds.
                    //
                    // A possibly better split is to try to maximize
                    // the sum of square distances from the split.
                    //
                    // define x as the split offset from i0.  The
                    // distances from the split plane are thus:
                    //
                    //   i0 to split = x
                    //   split to i1 = di - x
                    //   j0 to split = d0 + x
                    //   split to j1 = d1 - x
                    //
                    // summing the square of the above quantities,
                    // then differentiating and solving for 0, we get:
                    //
                    // x = (di + d1 - d0) / 4;

                    // Compute the range as the sum of distances from
                    // the split.  We add PI so that we prefer
                    // splitting these axes over axes that are already
                    // split.
                    Scalar dSum = PI<Scalar> + (di+d0+d1)/2; // std::min({di, d0, d1});
                    if (dSum > dBest) {
                        dBest = dSum;
                        split = *i0 + std::min(di, d1) * Scalar(0.5);
                    }
                }
            }
        } while (i1 != first);

        //if (sBest > PI<Scalar>) sBest -= 2*PI<Scalar>;
        split = so2::bound(split);

        assert(0 <= dBest && dBest <= 2*PI<Scalar>);
        assert(-PI<Scalar> <= split && split <= PI<Scalar>);

        return std::make_pair(dBest, split);
    }

    template <typename Container>
    auto split(Container& container) {
        return split(container.begin(), container.end());
    }
}

#endif
