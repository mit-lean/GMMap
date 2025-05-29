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


// Parameters
//    METRIC_SPACE:
//       l1_3
//       l2_3
//       linf_3
//       so2
//       so3
//       so3_l2_3 (se3)
//       l2_2_so2 (se2)
//       so3x5_l2_3x7 (weighted)
//    STATE:
//       eigen_quaternion
//       eigen_vector
//       eigen_array
//       std_vector
//       std_array
//       std_tuple
//       scalar (for R^1 spaces)
//    SCALAR:
//       float
//       double
//       long_double
//    VALUE:
//       node     Node<State>    NodeKey
//       value    State          NodePtrKey
//       nodeptr  Node<State>*   Identity
//       valueptr State*         Deref
//    CONCURRENCY:
//       rw = Concurrent
//       ro = ConcurrentRead
//       nt = NoThreadSafety
//    STRATEGY:
//       batch_8 = KDTreeBatch<8>
//       linear  = Linear
//    K: (number of search results)
//    N: (number of inserts)


#define METRIC_L1   1001
#define METRIC_L2   1002
#define METRIC_LInf 1000
#define METRIC_SO2  1020
#define METRIC_SO3  1030
#define METRIC_SE2  1021
#define METRIC_SE3  1031


#include "test.hpp"
#include <nigh/lp_metric.hpp>
#include <nigh/so2_metric.hpp>
#include <nigh/so3_metric.hpp>
#include <nigh/cartesian_metric.hpp>
#include <nigh/scaled_metric.hpp>
#include <nigh/kdtree_batch.hpp>
#include <nigh/linear.hpp>
#include <random>
#include <algorithm>
#include <iomanip>

namespace nigh_test {
    template <typename Char, typename Traits, typename Tuple, std::size_t ... I>
    auto& write(std::basic_ostream<Char, Traits>& out, const Tuple& tuple, std::index_sequence<I...>) {
        out << "{";
        ((out << (I?", " : "") << std::get<I>(tuple)), ...);
        return out << "}";
    }
}

namespace std {
    template <typename Char, typename Traits, typename ... T>
    auto& operator << (std::basic_ostream<Char, Traits>& out, const std::tuple<T...>& q) {
        return nigh_test::write(out, q, std::index_sequence_for<T...>{});
    }
}

namespace Eigen {
    template <typename Char, typename Traits, typename Derived>
    auto& operator << (std::basic_ostream<Char, Traits>& out, const Eigen::QuaternionBase<Derived>& q) {
        return out << q.coeffs().transpose();
    }

    template <typename A, typename B>
    bool operator == (const Eigen::QuaternionBase<A>& a, const Eigen::QuaternionBase<B>& b) {
        return a.coeffs() == b.coeffs();
    }
}

namespace nigh_test {
    template <typename T>
    struct Node {
        T state_;
        std::string name_;

        Node(const std::string& name, const T& q)
            : state_(q), name_(name)
        {
        }

        bool operator == (const Node& other) const {
            return name_ == other.name_;
        }

        template <typename Char, typename Traits = std::char_traits<Char>>
        friend auto& operator << (std::basic_ostream<Char, Traits>& out, const Node& node) {
            return out << "{" << node.name_<< ": " << node.state_ << "}";
        }
    };

    struct NodeKey {
        template <typename T>
        constexpr const T& operator () (const Node<T>& n) const {
            return n.state_;
        }
    };

    struct NodePtrKey {
        template <typename T>
        constexpr const T& operator () (const Node<T>* n) const {
            return n->state_;
        }
    };

    struct Identity {
        template <typename T>
        constexpr const T& operator () (const T& q) const { return q; }
    };

    struct Deref {
        template <typename T>
        constexpr const T& operator () (const T* q) const { return *q; }
    };

    template <typename State, typename Metric>
    struct Sampler;

    template <typename State, typename Metric, typename Weight>
    struct Sampler<State, unc::robotics::nigh::ScaledMetric<Metric, Weight>>
        : Sampler<State, Metric>
    {
    };

    template <typename State, int p>
    struct Sampler<State, unc::robotics::nigh::LPMetric<p>> {
        using Scalar = typename State::Scalar;
        static_assert(std::is_floating_point_v<Scalar>, "expected floating point scalar for LP space");
        std::uniform_real_distribution<Scalar> dist_{-5, 5};

        template <typename RNG>
        State operator() (RNG& rng) {
            State q;
            for (std::size_t i=0 ; i<q.size() ; ++i)
                q[i] = dist_(rng);
            return q;
        }
    };

    template <typename State>
    struct Sampler<State, unc::robotics::nigh::SO3Metric> {
        using Scalar = typename State::Scalar;

        template <typename RNG>
        State operator() (RNG& rng) {
            using namespace unc::robotics::nigh::impl;
            std::uniform_real_distribution<Scalar> dist01(0, 1);
            std::uniform_real_distribution<Scalar> dist2pi(0, 2*PI<Scalar>);
            Scalar a = dist01(rng);
            Scalar b = dist2pi(rng);
            Scalar c = dist2pi(rng);

            return State(
                std::sqrt(1-a)*std::sin(b),
                std::sqrt(1-a)*std::cos(b),
                std::sqrt(a)*std::sin(c),
                std::sqrt(a)*std::cos(c));
        }
    };

    template <typename State, typename Metric, typename Indices>
    struct CartesianSampler;

    template <typename State, typename Metric, std::size_t ... I>
    struct CartesianSampler<State, Metric, std::index_sequence<I...>>
        : std::tuple<Sampler<
                         typename unc::robotics::nigh::cartesian_state_element<I, State>::type,
                         unc::robotics::nigh::cartesian_metric_element_t<I, Metric>>...>
    {
        template <typename RNG>
        State operator() (RNG& rng){
            State q;
            ((std::get<I>(q) = std::get<I>(*this)(rng)), ...);
            return q;
        }
    };

    template <typename State, typename ... M>
    struct Sampler<State, unc::robotics::nigh::CartesianMetric<M...>>
        : CartesianSampler<State, unc::robotics::nigh::CartesianMetric<M...>, std::index_sequence_for<M...>>
    {

    };

    template <typename T>
    struct TestHelper {
        static T& get(std::vector<T>& nodes, std::size_t i) { return nodes[i]; }
    };

    template <typename T>
    struct TestHelper<T*> {
        static T* get(std::vector<T>& nodes, std::size_t i) { return &nodes[i]; }
    };

    template <typename T>
    void emplace(std::vector<Node<T>>& nodes, const T& q) {
        nodes.emplace_back(std::to_string(nodes.size()), q);
    }

    template <typename T>
    void emplace(std::vector<T>& nodes, const T& q) {
        nodes.push_back(q);
    }

    template <typename T, typename Metric, typename Member, typename Concurrency, typename Strategy>
    void runInsertTest(
        std::size_t N = 5000,
        std::size_t K = 50,
        const Member& member = Member(),
        const Metric& metric = Metric())
    {
        using namespace unc::robotics::nigh;
        using Key = std::decay_t<std::result_of_t<Member(T)>>;
        using Distance = typename metric_space_traits<Key, Metric>::Distance;
        using Helper = TestHelper<T>;

        Nigh<T, Metric, Member, Concurrency, Strategy> nn(metric);

        Sampler<Key, Metric> sampler;
        std::vector<std::remove_pointer_t<T>> nodes;
        std::vector<int> linear;
        std::vector<std::tuple<T, Distance>> results;
        std::mt19937_64 rng;

        nodes.reserve(N);
        results.reserve(K+1);
        EXPECT(nn.size()) == 0;

        for (std::size_t size=1 ; size <= N ; ++size) {
            Key q = sampler(rng);
            emplace(nodes, q);
            T const& value = Helper::get(nodes, size-1);
            nn.insert(value);
            linear.push_back(size-1);
            EXPECT(nn.size()) == size;

            std::optional<std::pair<T, Distance>> result = nn.nearest(q);
            EXPECT(!!result) == true;
            EXPECT(result->first) == value;
            EXPECT(result->second) == distance(metric, q, q); //< 1e-9;

            // std::cout << "size = " << size << std::endl;
            std::size_t maxK = std::min(K, size);
            for (std::size_t k=1 ; k<=maxK ; ++k) {
                q = sampler(rng);
                nn.nearest(results, q, k);
                EXPECT(results.size()) == k;

                for (std::size_t i=0 ; i<k ; ++i) {
                    EXPECT(std::get<1>(results[i])) == distance(metric, q, member(std::get<0>(results[i])));
                    if (i>0)
                        EXPECT(std::get<1>(results[i-1]) <= std::get<1>(results[i])) == true;
                }

                // TODO: test radius based query
            }

            // check the last k-nn result against a brute-force search
            std::partial_sort(
                linear.begin(), linear.begin() + maxK, linear.end(),
                [&] (int a, int b) {
                    return distance(metric, q, member(Helper::get(nodes, a))) < distance(metric, q, member(Helper::get(nodes, b)));
                });

            // std::cerr << "=== size: " << size << ", k: " << maxK << std::endl;
            for (std::size_t i=0 ; i<maxK ; ++i) {
                // if (i+1<maxK)
                //     std::cerr << i << "+1: " // << linear[i] << " "
                //               << std::setprecision(std::numeric_limits<Distance>::max_digits10)
                //               << distance(metric, q, member(Helper::get(nodes, linear[i+1]))) << " "
                //               << std::get<1>(results[i+1])
                //               << std::endl;

                EXPECT(std::get<1>(results[i])) == distance(metric, q, member(Helper::get(nodes, linear[i])));
                // The following occasionally fails when two values
                // have the exact same distance, and get sorted
                // arbitrarily different:
                //
                // EXPECT(std::get<0>(results[i]) == Helper::get(nodes, linear[i])) == true;
            }
        }
    }
}

TEST(inserts_l2) {
    using namespace unc::robotics::nigh;
    using namespace nigh_test;

    using Scalar = double;
    constexpr int kDim = 3;
    using State = Eigen::Matrix<Scalar, kDim, 1>;
    using Metric = L2Metric;

    runInsertTest<Node<State>, Metric, NodeKey, Concurrent, KDTreeBatch<8>>();
    runInsertTest<Node<State>*, Metric, NodePtrKey, Concurrent, KDTreeBatch<8>>();
    runInsertTest<State, Metric, Identity, Concurrent, KDTreeBatch<8>>();
    runInsertTest<State*, Metric, Deref, Concurrent, KDTreeBatch<8>>();
}

TEST(inserts_l1) {
    using namespace unc::robotics::nigh;
    using namespace nigh_test;

    using Scalar = double;
    constexpr int kDim = 3;
    using State = Eigen::Matrix<Scalar, kDim, 1>;
    using Metric = L1Metric;

    // runInsertTest<Node<State>, Metric, NodeKey, Concurrent, KDTreeBatch<8>>();
    runInsertTest<Node<State>*, Metric, NodePtrKey, Concurrent, KDTreeBatch<8>>();
    // runInsertTest<State, Metric, Identity, Concurrent, KDTreeBatch<8>>();
    // runInsertTest<State*, Metric, Deref, Concurrent, KDTreeBatch<8>>();
}

TEST(inserts_linf) {
    using namespace unc::robotics::nigh;
    using namespace nigh_test;

    using Scalar = double;
    constexpr int kDim = 3;
    using State = Eigen::Matrix<Scalar, kDim, 1>;
    using Metric = LInfMetric;

    // runInsertTest<Node<State>, Metric, NodeKey, Concurrent, KDTreeBatch<8>>();
    runInsertTest<Node<State>*, Metric, NodePtrKey, Concurrent, KDTreeBatch<8>>();
    // runInsertTest<State, Metric, Identity, Concurrent, KDTreeBatch<8>>();
    // runInsertTest<State*, Metric, Deref, Concurrent, KDTreeBatch<8>>();
}

TEST(inserts_so3_quaternion) {
    using namespace unc::robotics::nigh;
    using namespace nigh_test;

    using Scalar = float;
    using State = Eigen::Quaternion<Scalar>;
    using Metric = SO3Metric;

    // runInsertTest<Node<State>, Metric, NodeKey, Concurrent, KDTreeBatch<8>>();
    runInsertTest<Node<State>*, Metric, NodePtrKey, Concurrent, KDTreeBatch<8>>();
    // runInsertTest<State, Metric, Identity, Concurrent, KDTreeBatch<8>>();
    // runInsertTest<State*, Metric, Deref, Concurrent, KDTreeBatch<8>>();
}

TEST(inserts_so3_vector) {
    using namespace unc::robotics::nigh;
    using namespace nigh_test;

    using Scalar = float;
    using State = Eigen::Matrix<Scalar, 4, 1>;
    using Metric = SO3Metric;

    // runInsertTest<Node<State>, Metric, NodeKey, Concurrent, KDTreeBatch<8>>();
    runInsertTest<Node<State>*, Metric, NodePtrKey, Concurrent, KDTreeBatch<8>>();
    // runInsertTest<State, Metric, Identity, Concurrent, KDTreeBatch<8>>();
    // runInsertTest<State*, Metric, Deref, Concurrent, KDTreeBatch<8>>();
}

TEST(inserts_l2x2) {
    using namespace unc::robotics::nigh;
    using namespace nigh_test;

    using Scalar = double;
    constexpr int kDim = 2;
    using State = std::tuple<
        Eigen::Matrix<Scalar, kDim, 1>,
        Eigen::Matrix<Scalar, kDim, 1>>;
    using Metric = CartesianMetric<L2Metric, L2Metric>;

    // runInsertTest<Node<State>, Metric, NodeKey, Concurrent, KDTreeBatch<8>>();
    runInsertTest<Node<State>*, Metric, NodePtrKey, Concurrent, KDTreeBatch<8>>();
    // runInsertTest<State, Metric, Identity, Concurrent, KDTreeBatch<8>>();
    // runInsertTest<State*, Metric, Deref, Concurrent, KDTreeBatch<8>>();
}

TEST(inserts_se3) {
    using namespace unc::robotics::nigh;
    using namespace nigh_test;

    using Scalar = double;
    using State = std::tuple<
        Eigen::Quaternion<Scalar>,
        Eigen::Matrix<Scalar, 3, 1>>;
    using Metric = CartesianMetric<SO3Metric, L2Metric>;

    // runInsertTest<Node<State>, Metric, NodeKey, Concurrent, KDTreeBatch<8>>();
    runInsertTest<Node<State>*, Metric, NodePtrKey, Concurrent, KDTreeBatch<8>>();
    // runInsertTest<State, Metric, Identity, Concurrent, KDTreeBatch<8>>();
    // runInsertTest<State*, Metric, Deref, Concurrent, KDTreeBatch<8>>();
}

TEST(inserts_scaled_se3) {
    using namespace unc::robotics::nigh;
    using namespace nigh_test;

    using Scalar = double;
    using State = std::tuple<
        Eigen::Quaternion<Scalar>,
        Eigen::Matrix<Scalar, 3, 1>>;
    using Metric = CartesianMetric<
        ScaledMetric<SO3Metric, std::ratio<7,2>>,
        L2Metric>;

    //runInsertTest<Node<State>, Metric, NodeKey, Concurrent, KDTreeBatch<8>>();
    runInsertTest<Node<State>*, Metric, NodePtrKey, Concurrent, KDTreeBatch<8>>();
    // runInsertTest<State, Metric, Identity, Concurrent, KDTreeBatch<8>>();
    // runInsertTest<State*, Metric, Deref, Concurrent, KDTreeBatch<8>>();
}

TEST(inserts_scaled_se3_linear_concurrent) {
    using namespace unc::robotics::nigh;
    using namespace nigh_test;

    using Scalar = double;
    using State = std::tuple<
        Eigen::Quaternion<Scalar>,
        Eigen::Matrix<Scalar, 3, 1>>;
    using Metric = CartesianMetric<
        ScaledMetric<SO3Metric, std::ratio<7,2>>,
        L2Metric>;

    runInsertTest<Node<State>*, Metric, NodePtrKey, Concurrent, Linear>(500);
}

TEST(inserts_scaled_se3_linear_concurrent_read) {
    using namespace unc::robotics::nigh;
    using namespace nigh_test;

    using Scalar = double;
    using State = std::tuple<
        Eigen::Quaternion<Scalar>,
        Eigen::Matrix<Scalar, 3, 1>>;
    using Metric = CartesianMetric<
        ScaledMetric<SO3Metric, std::ratio<7,2>>,
        L2Metric>;

    runInsertTest<Node<State>*, Metric, NodePtrKey, ConcurrentRead, Linear>(500);
}


