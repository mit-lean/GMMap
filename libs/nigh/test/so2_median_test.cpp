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

#include "test.hpp"
#include <nigh/so2_space.hpp>
#include <nigh/impl/kdtree_batch/traversal_so2.hpp>
#include <random>
#include <array>

TEST(random_splits) {
    using namespace unc::robotics::nigh::impl;

    std::mt19937_64 rng;

    typedef double Scalar;
    static constexpr std::size_t N = 8;

    std::array<Scalar, N> vals;

    std::uniform_real_distribution<Scalar> urd(-PI<Scalar>, PI<Scalar>);

    for (std::size_t iter = 0 ; iter < 100000 ; ++iter) {
        std::generate(vals.begin(), vals.end(), [&] { return urd(rng); });

        auto [dist, split] = so2::split(vals);
        // for (std::size_t i=0 ; i<N ; ++i)
        //     std::cerr << vals[i] << ", ";
        // std::cerr << "split = " << split << ", " << split + PI<Scalar> << std::endl;
        EXPECT(dist) > 0;
        std::size_t n = 0;
        for (Scalar v : vals)
            n += (so2::ccwDist(split, v) < PI<Scalar>);
        EXPECT(n) == N/2;
    }
}

TEST(split_with_duplicated_value) {
    using namespace unc::robotics::nigh::impl;

    using Vec3 = Eigen::Vector3d;
    static constexpr std::size_t N = 8;
    static const std::array<Vec3, N> vectors = {
        Vec3{-1.9412700000000001, 2.47058,            2.1993800000000001},
        Vec3{-1.9494100000000001, 2.6524800000000002, 1.9875499999999999},
        Vec3{-1.98017,            2.49031,            2.0501200000000002},
        Vec3{-2.2128000000000001, 2.4399700000000002, 1.97763},

        Vec3{-2.2128000000000001, 2.4920900000000001, 2.03512},
        Vec3{-2.3332600000000001, 2.5669,             2.1341899999999998},
        Vec3{-2.4556399999999998, 2.71225,            1.9807600000000001},
        Vec3{-2.5425399999999998, 2.6987899999999998, 1.9787399999999999},
        // Vec3{-2.09 + 3.14, 2.4399700000000002, 1.97763},
    };

    std::array<double, N> values;
    for (std::size_t i=0 ; i<N ; ++i)
        values[i] = vectors[i][0];

    auto [dist, split] = so2::split(values);

    EXPECT(split) == -2.2128000000000001;
    EXPECT(dist) == Approx(-1.9412700000000001 - -2.5425399999999998);
}

TEST(split_with_duplicated_value_2) {
    using namespace unc::robotics::nigh::impl;

    static constexpr std::size_t N = 8;
    std::array<double, N> values{{
            -1.26332,
            -1.26457,
            -1.26532,
            -1.26729,
            -1.26729,
            -1.26866,
            -1.26874,
            -1.26892,
        }};

    auto [dist, split] = so2::split(values);
    EXPECT(split) == -1.26729;
}

TEST(split_evenly_spaced) {
    using namespace unc::robotics::nigh::impl;
    static constexpr std::size_t Nmax = 64;
    std::vector<double> vals;
    for (std::size_t n = 2 ; n<=Nmax ; n+=2) {
        vals.resize(n);
        for (std::size_t i = 0 ; i < n ; ++ i)
            vals[i] = 2*PI<double> * i / n;

        auto [dist, split] = so2::split(vals);

        EXPECT(std::abs(dist - (PI<double> + PI<double>*2/n))) < 1e-15;

        double s = split;
        if (s < PI<double>) s += 2*PI<double>;
        EXPECT(s >= 0) == true;
        while (s > 2*PI<double>/n) s -= 2*PI<double>/n;
        EXPECT(std::abs(s - PI<double>/n)) < 1e-13;
    }
}
