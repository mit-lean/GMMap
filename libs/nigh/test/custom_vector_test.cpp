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

#include <nigh/metric/lp.hpp>
#include <nigh/kdtree_batch.hpp>
#include "test_template.hpp"
#include <memory>

namespace nigh_test {
    template <typename S>
    struct Vec3 {
        S x;
        S y;
        S z;

        // this operator ONLY required for the BoxSampler which is use
        // in testing, but is not otherwise required for use in the
        // Nigh API.  The Space<> specialization, however, does
        // require a coeff() operation.
        S& operator[] (std::size_t i) {
            return *(&x + i);
        }
    };

    // template <typename S>
    // struct BoxSampler<Vec3<S>, 3> {
    //     template <typename RNG>
    //     Vec3<S> operator () (RNG& rng) {
    //         std::uniform_real_distribution<S> dist(-5, 5);
    //         Vec3<S> q;
    //         q.x = dist(rng);
    //         q.y = dist(rng);
    //         q.z = dist(rng);
    //         return q;
    //     }
    // };

    // We can also can use an arbitrary array of scalars as a vector
    // type--in this case we test supporting a std::shared_ptr (Space
    // specialization defined below) so that our test can efficiently
    // clean up afterwareds.  For we need to define a specialization
    // of BoxSampler in order to allocate the state vector with the
    // proper number of elements.
    template <typename S, int dim>
    struct BoxSampler<std::shared_ptr<S[]>, dim>
        : BoxSamplerBase<S, dim>
    {
        using Base = BoxSamplerBase<S, dim>;
        using Base::Base;
        using State = std::shared_ptr<S[]>;

        template <typename RNG>
        State operator() (RNG& rng) {
            State q(new S[Base::dimensions]);
            for (unsigned i=0 ; i<Base::dimensions ; ++i)
                q[i] = Base::dist_(rng);
            return q;
        }
    };

}

namespace unc::robotics::nigh::metric {
    using namespace nigh_test;

    // In order to support searching on a new type, we need to define
    // a specialization for the type for each metric we wish to
    // support with it.  The Space specialization must have members
    // that match the class the follows.
    template <typename S, int p>
    struct Space<Vec3<S>, LP<p>> {
        using Type = Vec3<S>;
        using Distance = S;
        using Metric = LP<p>;

        static constexpr unsigned kDimensions = 3;

        static bool isValid(const Type& v) {
            return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
        }

        static Distance coeff(const Type& a, std::size_t i) {
            return *(&a.x + i);
        }

        constexpr unsigned dimensions() const {
            return 3;
        }

        Distance distance(const Type& a, const Type& b) const {
            using Map = Eigen::Map<const Eigen::Matrix<S, 3, 1>>;
            return (Map(&a.x) - Map(&b.x)).template lpNorm<p>();
        }
    };

    template <typename S, int p>
    struct Space<std::shared_ptr<S[]>, LP<p>, std::enable_if_t<std::is_floating_point_v<S>>> {
        using Type = std::shared_ptr<S[]>;
        using Distance = S;
        using Metric = LP<p>;

        static constexpr int kDimensions = -1;

    private:
        unsigned dimensions_;

    public:
        Space(unsigned dimensions) : dimensions_(dimensions) {}

        static bool isValid(const Type&) { return true; }

        static Distance coeff(const Type& a, std::size_t i) { return a[i]; }

        constexpr unsigned dimensions() const {
            return dimensions_;
        }

        Distance distance(const Type& a, const Type& b) const {
            using Map = Eigen::Map<const Eigen::Matrix<S, Eigen::Dynamic, 1>>;
            return (Map(a.get(), dimensions_) - Map(b.get(), dimensions_)).template lpNorm<p>();
        }
    };
}

TEST(custom_vector) {
    using namespace unc::robotics::nigh;
    using namespace unc::robotics::nigh::metric;
    using namespace nigh_test;
    using Metric = L2;
    using State = nigh_test::Vec3<double>;
    using Space = metric::Space<State, Metric>;
    using Concurrency = Concurrent;
    using Strategy = KDTreeBatch<>;
    Space space;
#ifdef BUILD_TYPE_VALGRIND
    static constexpr std::size_t N = 1000/10;
    static constexpr std::size_t K = 1;
#else
    static constexpr std::size_t N = 1000;
    static constexpr std::size_t K = 50;
#endif
    runTest<State, Space, Identity, Concurrency, Strategy>(space, N, K);
}

// TODO: not all C++ libraries support shared_ptr<T[]>
#if 0

TEST(custom_space_of_pointers) {
    using namespace unc::robotics::nigh;
    using namespace unc::robotics::nigh::metric;
    using namespace nigh_test;
    using Metric = L2;
    using State = std::shared_ptr<double[]>;
    using Space = metric::Space<State, Metric>;
    using Concurrency = Concurrent;
    using Strategy = KDTreeBatch<>;
#ifdef BUILD_TYPE_VALGRIND
    static constexpr std::size_t N = 1000/10;
    static constexpr std::size_t K = 1;
#else
    static constexpr std::size_t N = 1000;
    static constexpr std::size_t K = 50;
#endif
    Space space(3);
    runTest<Node<State>, Space, NodeKey, Concurrency, Strategy>(space, N, K);
}

#endif
