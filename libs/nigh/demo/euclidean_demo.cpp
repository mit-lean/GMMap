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

// This file demonstrates how to insert into a nearest neighbor data
// structure and they query for a point in it.  In this demos, the
// points are in R^2 and part of a class that has a point and a name.

// Include the header file for the space that we use.  Euclidean space
// has an L^2 metric, so we include a space file for general L^p
// metric spaces, which includes the L2Space alias.
#include <nigh/lp_space.hpp>

// Nigh support several nearest neighbor strategies, we will use the
// one that benchmarks the fastest and supports concurrent queries and
// inserts.
#include <nigh/kdtree_batch.hpp>

// Our actual points use vectors from the Eigen matrix library, since
// include the necessary header for it.
#include <Eigen/Dense>

// Other headers for the demo
#include <string>
#include <random>
#include <fstream>
#include <iostream>
#include <unistd.h>

// This is our node that we will insert into the nearest neighbor
// structure.  Note that nigh will hold a copy of the object in its
// internal data structure.  For objects that are expensive to copy,
// it may be better to store a pointer to a data structure instead.
struct MyNode {
    std::string name_;
    Eigen::Vector2f point_;

    MyNode(const std::string& name, const Eigen::Vector2f& pt)
        : name_(name)
        , point_(pt)
    {
    }
};

// Nigh must have a functor to access the stored object's key.  This
// allows arbitrary data types to be stored in the nearest neighbor
// data structure.
struct MyNodeKey {
    // The functor should return a constant reference to the data
    // member or a copy of it.
    const Eigen::Vector2f& operator () (const MyNode& node) const {
        return node.point_;
    }
};

static void usage(const char *argv0) {
    std::cout << "Usage: " << argv0 << " [options] [x y]\n"
        "Generates an image with random points and the result\n"
        "of a nearest neighbor search."
        "Options:\n"
        "  -o OUT      writes output to OUT (default is euclidean_demo.svg)\n"
        "  -r RADIUS   sets the maximum query radius (default is unbounded)\n"
        "  -k COUNT    sets the maximum number of results to return (default is 1, unless -r is specified)\n"
        "  -N COUNT    sets the numer of random points to generate (default is 100)\n" <<
        std::flush;
    std::exit(1);
}

int main(int argc, char *argv[]) {
    using Vec2 = Eigen::Vector2f;
    namespace nigh = unc::robotics::nigh;

    std::string fileName = "euclidean_demo.svg";
    std::size_t k = 0;
    std::size_t N = 100;
    float r = std::numeric_limits<float>::infinity();
    
    for (int ch ; (ch = getopt(argc, argv, "r:k:N:o:h")) != -1 ; ) {
        if (ch == 'r') {
            if ((r = std::strtof(optarg, nullptr)) < 0 || !std::isfinite(r)) {
                std::cerr << "-r must be a non-negative real" << std::endl;
                return 1;
            }
        } else if (ch == 'k') {
            if ((k = std::strtol(optarg, nullptr, 10)) < 1) {
                std::cerr << "-k must be at least 1" << std::endl;
                return 1;
            }
        } else if (ch == 'N') {
            if ((N = std::strtol(optarg, nullptr, 10)) > 1000000) {
                std::cerr << "invalid value for -N" << std::endl;
                return 1;
            }
        } else if (ch == 'o') {
            fileName = optarg;
        } else {
            usage(argv[0]);
        }
    }

    argc -= optind;
    argv += optind;
    
    // Declare the nearest neighbor data structure.  The template
    // parameters in order are:
    //   1. the contained data type
    //   2. the metric space of the key
    //   3. the functor type to access the key from the contained data type.
    //   4. the concurrency level required.
    //   5. the strategy, which will be automatically selected
    // Here we are using the default parameter types for:
    //   6. the allocator
    nigh::Nigh<MyNode, nigh::L2Space<float, 2>, MyNodeKey, nigh::Concurrent, nigh::KDTreeBatch<>> nn;

    // This demo generates an SVG file containing the random points,
    // and the result of the query.  This opens the file for writing
    // and outputs a header for the SVG file.  (SVG is a image format
    // that can be opened by modern browsers)
    std::cout << "generating image: " << fileName << std::endl;
    std::ofstream svg{fileName};
    svg << "<?xml version='1.0'?>\n"
        << "<svg xmlns='http://www.w3.org/2000/svg' width='500' height='500' "
        << "viewBox='0 0 1 1' style='stroke:none; fill:blue;'>\n";

    // Generate and insert random points
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;
    for (std::size_t i=0 ; i<N ; ++i) {
        // generate a random point
        Vec2 v{dist(rng), dist(rng)};

        // insert it into the nearest neighbor structure
        nn.insert(MyNode{std::to_string(i), v});

        // draw the point
        svg << "<circle cx='" << v[0] << "' cy='" << v[1] << "' r='0.002'/>\n";
    }
    
    Vec2 q;
    if (argc >= 2) {
        // Search for a user specified point
        q << std::strtof(argv[0], nullptr), std::strtof(argv[1], nullptr);
    } else {
        // Search for a random point
        q << dist(rng), dist(rng);
    }
    
    if (k <= 1 && r == std::numeric_limits<float>::infinity()) {
        // find the nearest point.  The return value is optional but
        // in this case guaranteed to be present.  The return value
        // will not be present when the nearest neighbor structure is
        // empty, and thus there is no nearest point.
        std::optional<std::pair<MyNode, float>> pt = nn.nearest(q);

        // Show the line connecting to the nearest point and the query
        // point
        svg << "<line x1='" << q[0] << "' y1='" << q[1]
            << "' x2='" << pt->first.point_[0] << "' y2='" << pt->first.point_[1]
            << "' style='stroke:green;stroke-width:0.005;'/>\n";
    } else {
        
        // if the user specified a search radius, display the radius
        if (r > 0)
            svg << "<circle cx='" << q[0] << "' cy='" << q[1] << "' r='" << r
                << "' style='fill:none;stroke:green;stroke-width:0.003;'/>\n";

        // if k is 0, and the radius is bounded, we set the k to be
        // effectively unbounded.
        if (k < 1 && std::isfinite(r))
            k = std::numeric_limits<std::size_t>::max();
        
        std::cout << "nearest: r=" << r << ", k=" << k << std::endl;

        // vector to store return value.  This is more efficient than
        // returning a vector when the nearest search method is
        // repeatedly called.
        std::vector<std::pair<MyNode, float>> nbh;

        // perform the nearest neighbors search.  Note that the last
        // parameter is optional and defaults to unbounded radius.
        // Thus if you're only performing a k-nearest search, the last
        // paraemter can be left unspecified.  For radius bounded
        // searchs, k must be specified to insure that the caller is
        // not unintentially requested a search of unbounded size.
        nn.nearest(nbh, q, k, r);

        // Draw lines to all the nearest neighbors found.
        for (auto& n : nbh)
            svg << "<line x1='" << q[0] << "' y1='" << q[1]
                << "' x2='" << n.first.point_[0] << "' y2='" << n.first.point_[1]
                << "' style='stroke:green;stroke-width:0.003;'/>\n";
    }
                

    // display a circle for query point
    svg << "<circle cx='" << q[0] << "' cy='" << q[1] << "' r='0.01' style='fill:red'/>\n"
        << "</svg>" << std::endl;

    return 0;
}
