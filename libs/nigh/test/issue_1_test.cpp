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
//! @author clemense (GitHub)

#include "test.hpp"
#include <nigh/se3_space.hpp>
#include <nigh/kdtree_batch.hpp>

struct Key {
    template <class T>
    const T& operator () (const T& t) const {
        return t;
    }
};

TEST(assert_in_se3) {
    using namespace unc::robotics::nigh;

    using Scalar = float;
    using State = std::tuple<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>;
    using Metric = metric::Cartesian<L2, SO3>;
    using Space = metric::Space<State, Metric>;
        
    static State data[] = {
        { {-0.1613302401, -0.0386662401, -0.0376202401 }, { 0.0499525, 0.578929, 0.285496, 0.762127 } },
        { {-0.1613302401, -0.0386662401, -0.0376202401 }, { 0.762127, -0.285496, 0.578929, -0.0499526 } },
        { {-0.1613302401, -0.0336662401, -0.0526202401 }, { -0.106574, 0.576114, 0.0377605, 0.809511 } },
        { {-0.1613302401, -0.0336662401, -0.0526202401 }, { 4.5e-09, 0.566257, 0.112635, 0.816497 } },
        { {-0.1613302401, -0.0336662401, -0.0526202401 }, { 0.809512, -0.0377605, 0.576114, 0.106574 } },
        { {-0.1613302401, -0.0336662401, -0.0526202401 }, { 0.816496, -0.112635, 0.566257, -3.7e-09 } },
        { {-0.1613302401, -0.0336662401, -0.0476202401 }, { 0.149003, 0.536711, 0.358619, 0.749087 } },
        { {-0.1613302401, -0.0336662401, -0.0476202401 }, { 0.749087, -0.358619, 0.536711, -0.149003 } },
        { {-0.1613302401, -0.0336662401, -0.0426202401 }, { 0.72323, 0.207488, 0.611241, 0.245503 } },
        { {-0.1613302401, -0.0336662401, -0.0426202401 }, { -0.245503, 0.611241, -0.207488, 0.72323 } },
        { {-0.1613302401, -0.0336662401, -0.0376202401 }, { 0.749087, 0.12593, 0.633094, 0.149003 } },
        { {-0.1613302401, -0.0336662401, -0.0376202401 }, { -0.149003, 0.633094, -0.12593, 0.749087 } },
        { {-0.1613302401, -0.0286662401, -0.0576202401 }, { 4.5e-09, 0.566257, 0.112635, 0.816497 } },
        { {-0.1613302401, -0.0286662401, -0.0576202401 }, { 0.816496, -0.112635, 0.566257, -3.7e-09 } },
        { {-0.1613302401, -0.0286662401, -0.0526202401 }, { -0.106574, 0.576114, 0.0377605, 0.809511 } },
        { {-0.1613302401, -0.0286662401, -0.0526202401 }, { 4.5e-09, 0.566257, 0.112635, 0.816497 } },
        { {-0.1613302401, -0.0286662401, -0.0526202401 }, { 0.788675, 0.0377605, 0.576114, 0.211325 } },
        { {-0.1613302401, -0.0286662401, -0.0526202401 }, { 0.809512, -0.0377605, 0.576114, 0.106574 } },
        { {-0.1613302401, -0.0286662401, -0.0526202401 }, { 0.816496, -0.112635, 0.566257, -3.7e-09 } },
        { {-0.1613302401, -0.0286662401, -0.0526202401 }, { -0.211325, 0.576114, -0.0377604, 0.788675 } },
        { {-0.1613302401, -0.0286662401, -0.0476202401 }, { 0.72323, 0.207488, 0.611241, 0.245503 } },
        { {-0.1613302401, -0.0286662401, -0.0476202401 }, { -0.245503, 0.611241, -0.207488, 0.72323 } },
        { {-0.1613302401, -0.0286662401, -0.0476202401 }, { 0.149003, 0.536711, 0.358619, 0.749087 } },
        { {-0.1613302401, -0.0286662401, -0.0476202401 }, { 0.749087, -0.358619, 0.536711, -0.149003 } },
        { {-0.1613302401, -0.0286662401, -0.0426202401 }, { 0.72323, 0.207488, 0.611241, 0.245503 } },
        { {-0.1613302401, -0.0286662401, -0.0426202401 }, { 0.749087, 0.12593, 0.633094, 0.149003 } },
        { {-0.1613302401, -0.0286662401, -0.0426202401 }, { -0.245503, 0.611241, -0.207488, 0.72323 } },
        { {-0.1613302401, -0.0286662401, -0.0426202401 }, { -0.149003, 0.633094, -0.12593, 0.749087 } },
        { {-0.1613302401, -0.0286662401, -0.0376202401 }, { 0.749087, 0.12593, 0.633094, 0.149003 } },
    };

    Nigh<State, Space, Key, Concurrent, KDTreeBatch<>> nn;

    for (std::size_t i = 0 ; i < sizeof(data)/sizeof(data[0]) ; ++i)
        nn.insert(data[i]);


    Space space;
    for (std::size_t i = 0 ; i < sizeof(data)/sizeof(data[0]) ; ++i) {
        std::optional<std::pair<State, Scalar>> n = nn.nearest(data[i]);

        EXPECT(!!n) == true;

        // distance to self is not always exactly 0 due to floating
        // point precision issue.
        Scalar dist = space.distance(data[i], data[i]);        

        EXPECT(n->second) == dist;
        EXPECT(space.distance(n->first, data[i])) == dist;
    }
}
