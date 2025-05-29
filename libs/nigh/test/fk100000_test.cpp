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
#include <iomanip>
#include <nigh/cartesian_space.hpp>
#include <nigh/lp_space.hpp>
#include <nigh/so2_space.hpp>
#include <nigh/kdtree_batch.hpp>
#include <fstream>

namespace nigh_test {
    template <typename S, typename Fn>
    void loadFK(const std::string& fileName, const Fn& fn) {
        using State = std::tuple<Eigen::Array<S, 3, 1>, Eigen::Matrix<S, 3, 1>>;

        std::ifstream in(fileName);
        EXPECT(!!in) == true; // check that file was opened

        std::string line;
        while (std::getline(in, line)) {
            if (line.empty())
                continue;

            std::istringstream iss(line, std::istringstream::in);
            State q;
            Eigen::Array<S, 3, 1> data;
            iss >> std::get<0>(q)[0]
                >> std::get<0>(q)[1]
                >> std::get<0>(q)[2]
                >> std::get<1>(q)[0]
                >> std::get<1>(q)[1]
                >> std::get<1>(q)[2]
                >> data[0]
                >> data[1]
                >> data[2];
            fn(q, data);
        }
    }
}

TEST(fkmap100000_double) {
    using namespace unc::robotics::nigh;

    using Scalar = double;
    using Strategy = KDTreeBatch<>;
    using Space = CartesianSpace<SO2LPSpace<Scalar, 3>, L1Space<Scalar, 3>>;
    using State = typename Space::Type;
    using Data = Eigen::Matrix<Scalar, 3, 1>;

    struct Node {
        State state_;
        Data data_;
    };

    struct KeyFn {
        const State& operator() (const Node& n) const {
            return n.state_;
        }
    };

    Space space;

    Nigh<Node, Space, KeyFn, Concurrent, Strategy> nn(space);

    nigh_test::loadFK<Scalar>(
        "fkmap100000.txt",
        [&] (const State& q, const Data& data) {
            nn.insert(Node{q, data});
        });
}
