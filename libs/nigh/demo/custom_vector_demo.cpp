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

// This file demonstrates how to use a custom data type for the key in
// Nigh.  In this file, the 'Vec3' class is a custom type for a
// 3-dimensional vector for which we wish to support nearest neighbor
// with the Euclidean metric.

// Since the demo uses Euclidean (L^2) metric, we need to include
// "lp_space.hpp"
#include <nigh/lp_space.hpp>
// This is the nearest neighbor strategy we wish to use.
#include <nigh/kdtree_batch.hpp>
// Other headers need for this demo
#include <random>
#include <iostream>
#include <cmath>

// This is the custom vector type.  Nigh supports using *any* type for
// a vector type as long as it is copyable.
struct Vec3 {
    double x;
    double y;
    double z;
};

// This is the stored data type that contains the vector.
struct MyNode {
    std::string name;
    Vec3 pt;
};

// This is the key accessor.  
struct MyNodeKey {
    const Vec3& operator () (const MyNode& node) const {
        return node.pt;
    }
};

// In order to support a custom vector type, we need to define a
// specialization of the metric::Space<> template with the first
// parameter as our data type, and the second parameter as the metric
// we wish to support.  To start the specialization declaration, we
// need to be in the same namespace as the template.
namespace unc::robotics::nigh::metric {
    
    // Here is the template specialization declaration.  It starts
    // with 'template <...>' and is then followed by the template name
    // 'Space'.  Note that we make the specialization operate on
    // templated types and metric parameters.  For an example of this
    // see the test file 'custom_vector_test.cpp'.  In this demo,
    // we're not specializing for just our concrete Vec3 class and in
    // the LP<2> (Euclidean) metric.
    template <>
    struct Space<Vec3, LP<2>> {

        // These type aliases are required.  Type should be the first
        // template parameter.
        using Type = Vec3;
        
        // This should be the second template parameter
        using Metric = LP<2>;

        // Distance alias should be both the result of the distance
        // function and the coefficient type.
        using Distance = double;


        // Declare the dimensions of Vec3 as a constant.  For
        // customizing types that can be of arbitrary dimensions,
        // kDimensions can be set to -1.
        static constexpr int kDimensions = 3;

        // Required method that checks if the vector is valid.  This
        // can be minimally implemented as 'return true;', but adding
        // additional checks for validity will help find invalid data
        // sooner.
        static bool isValid(const Vec3& v) {
            return std::isfinite(v.x) &&
                std::isfinite(v.y) &&
                std::isfinite(v.z);
        }

        // Required method that returns the coefficient using a
        // 0-based index.  In our example, (x, y, z) are in sequential
        // order, so we can do some pointer tricks to get the
        // zero-based index.  For other data layouts, if/else,
        // switch/case, etc... may be required.
        static double coeff(const Vec3& v, std::size_t i) {
            return *(&v.x + i);
        }

        // This constexpr method should return the number of
        // dimensions of the vector.  This method should match
        // kDimensions, except in the case where kDimensions is -1.
        // Note kDimensions is signed, while dimensions() is unsigned.
        constexpr unsigned dimensions() const {
            return 3;
        }

        // This required method computes and returns the distance
        // between two points.  The computation must match the metric
        // a specified by the second template parameter.  Note: Nigh
        // could implement this function based upon dimensions(), and
        // coeff(), however it is likely that a custom data type can
        // have a much faster implementation (e.g., based upon SIMD
        // instructions).
        static double distance(const Vec3& a, const Vec3& b) {
            return std::sqrt(
                std::pow(a.x - b.x, 2) +
                std::pow(a.y - b.y, 2) +
                std::pow(a.z - b.z, 2));
        }
    };
};

int main(int argc, char *argv[]) {
    namespace nigh = unc::robotics::nigh;

    // Declare our metric space.  The first parameter is our custom
    // type, and the second is the L^2 metric.  Using this metric
    // space will cause Nigh to make use of template specialization
    // above.
    using Vec3L2Space = nigh::metric::Space<Vec3, nigh::metric::LP<2>>;

    // Create our nearest-neighbot structure.  This will look like
    // other demos, except the metric-space parameter (the second
    // parameter).
    nigh::Nigh<MyNode, Vec3L2Space, MyNodeKey, nigh::Concurrent, nigh::KDTreeBatch<>> nn;

    // Generate and insert 100 random points
    std::size_t N = 100;
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;
    for (std::size_t i=0 ; i<N ; ++i) {
        Vec3 v{dist(rng), dist(rng), dist(rng)};
        nn.insert(MyNode{std::to_string(i), v});
    }

    // Declare our return type
    std::vector<std::tuple<MyNode, double>> nbh;

    // Query in the center of the space
    Vec3 q{0.5, 0.5, 0.5};

    // find the 10 nearest neighbors to q, storing the results in nbh.
    nn.nearest(nbh, q, 10);

    // Output the results
    for (auto& r : nbh) {
        std::cout << std::get<1>(r) << ": ["
                  << std::get<0>(r).pt.x << ", "
                  << std::get<0>(r).pt.y << ", "
                  << std::get<0>(r).pt.x << "]" << std::endl;
    }

    return 0;
}



