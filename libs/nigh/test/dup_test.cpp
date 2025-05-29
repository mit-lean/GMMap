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
#include <nigh/lp_space.hpp>
#include <nigh/so3_space.hpp>
#include <nigh/kdtree_batch.hpp>

using namespace unc::robotics::nigh;

template <class Space, class Strategy, class ... Args>
void dupTest(Args&& ... args) {
    using State = typename Space::Type;
    using Scalar = typename Space::Distance;
    
    struct Node {
        State state_;
        std::size_t index_;
    };
    struct KeyFn {
        const State& operator() (const Node& n) const {
            return n.state_;
        }
    };

    State q{std::forward<Args>(args)...};
    Space space;

    Nigh<Node, Space, KeyFn, Concurrent, Strategy> nn(space);

    for (std::size_t i=0 ; i<100 ; ++i)
        nn.insert(Node{q, i});

    std::vector<std::pair<Scalar, Node>> nbh;
    nn.nearest(nbh, q, 100, 1e-9);
    EXPECT(nbh.size()) == 100;
    EXPECT(nbh[0].first) == 0.0;
    EXPECT(nbh[99].first) == 0.0;    
}
    

TEST(batch_l2_dups) {
    dupTest<L2Space<double, 3>, KDTreeBatch<>>(1.0, 2.0, 3.0);
}

TEST(batch_so3_dups) {
    Eigen::Quaternion<double> q;
    q.coeffs() << 1.0, 0.0, 0.0, 0.0;
    dupTest<SO3Space<double>, KDTreeBatch<>>(q);
}
