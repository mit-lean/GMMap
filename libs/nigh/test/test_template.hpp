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

#include "test.hpp"
// #include <nigh/lp_metric.hpp>
// #include <nigh/so2_metric.hpp>
// #include <nigh/so3_metric.hpp>
// #include <nigh/cartesian_metric.hpp>
// #include <nigh/scaled_metric.hpp>
#include <nigh/kdtree_batch.hpp>
#include <nigh/linear.hpp>
#include <nigh/impl/constants.hpp>
#include <random>
#include <algorithm>
#include <iomanip>
#include "sampler_lp.hpp"
#include "sampler_so2.hpp"
#include "sampler_so3.hpp"
#include "sampler_scaled.hpp"
#include "sampler_cartesian.hpp"

namespace nigh_test {
    template <typename Char, typename Traits, typename Tuple, std::size_t ... I>
    auto& write(std::basic_ostream<Char, Traits>& out, const Tuple& tuple, std::index_sequence<I...>) {
        out << "{";
        ((out << (I?", " : "") << std::get<I>(tuple)), ...);
        return out << "}";
    }

    template <typename Scalar>
    struct SE3State
        : std::tuple<Eigen::Quaternion<Scalar>, Eigen::Matrix<Scalar, 3, 1>>
    {
        using Tuple = std::tuple<Eigen::Quaternion<Scalar>, Eigen::Matrix<Scalar, 3, 1>>;
        using Tuple::Tuple;
    };
}

namespace std {

// tuple_element is declared as a class and as a struct in some
// libraries.  there's no way to avoid the mismatched tag warning
// without just disabling it.
// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored "-Wmismatched-tags"

    template <typename Scalar>
    struct tuple_element<0, nigh_test::SE3State<Scalar>> {
        using type = Eigen::Quaternion<Scalar>;
    };

    template <typename Scalar>
    struct tuple_element<1, nigh_test::SE3State<Scalar>> {
        using type = Eigen::Matrix<Scalar, 3, 1>;
    };

// #pragma GCC diagnostic pop

    template <typename Char, typename Traits, typename ... T>
    auto& operator << (std::basic_ostream<Char, Traits>& out, const std::tuple<T...>& q) {
        return nigh_test::write(out, q, std::index_sequence_for<T...>{});
    }

    template <typename Char, typename Traits, typename T, std::size_t n>
    auto& operator << (std::basic_ostream<Char, Traits>& out, const std::array<T, n>& q) {
        out << "{";
        for (std::size_t i=0 ; i<n ; ++i) {
            if (i) out << ", ";
            out << q[i];
        }
        return out << "}";
    }

    template <typename Char, typename Traits, typename T, typename A>
    auto& operator << (std::basic_ostream<Char, Traits>& out, const std::vector<T, A>& q) {
        std::size_t n = q.size();
        out << "{";
        for (std::size_t i=0 ; i<n ; ++i) {
            if (i) out << ", ";
            out << q[i];
        }
        return out << "}";
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

    template <typename T, typename Space, typename KeyFn, typename Concurrency, typename Strategy>
    void runTest(
        const Space& metricSpace = Space(),
        std::size_t N = 5000,
        std::size_t K = 50,
        const KeyFn& keyFn = KeyFn())
    {
        using namespace unc::robotics::nigh;
        using Key = std::decay_t<std::result_of_t<KeyFn(T)>>;
        using Distance = typename Space::Distance;
        using Helper = TestHelper<T>;

        Nigh<T, Space, KeyFn, Concurrency, Strategy> nn(metricSpace);

        Sampler<Key, typename Space::Metric> sampler(metricSpace);
        std::vector<std::remove_pointer_t<T>> nodes;
        std::vector<int> linear;
        std::vector<std::tuple<T, Distance>> results;
        std::mt19937_64 rng;

        nodes.reserve(N);
        results.reserve(K+1);
        EXPECT(nn.size()) == 0u;

        Key q = sampler(rng);
        std::optional<std::pair<T, Distance>> emptyResult = nn.nearest(q);
        EXPECT(!!emptyResult) == false;

        nn.nearest(results, q, 10);
        EXPECT(results.size()) == 0;

        for (std::size_t size=1 ; size <= N ; ++size) {
            q = sampler(rng);
            emplace(nodes, q);
            T const& value = Helper::get(nodes, size-1);
            nn.insert(value);
            // std::cout << "size = " << size << ", depth = " << nn.depth() << std::endl;
            linear.push_back(size-1);
            EXPECT(nn.size()) == size;

            std::optional<std::pair<T, Distance>> result = nn.nearest(q);
            EXPECT(!!result) == true;
            // TODO: EXPECT(result->first) == value;
            EXPECT(result->second) == metricSpace.distance(q, q); //< 1e-9;

            // std::cout << "size = " << size << std::endl;
            std::size_t maxK = std::min(K, size);
            for (std::size_t k=1 ; k<=maxK ; ++k) {
                q = sampler(rng);
                nn.nearest(results, q, k);
                EXPECT(results.size()) == k;

                for (std::size_t i=0 ; i<k ; ++i) {
                    EXPECT(std::get<1>(results[i])) == metricSpace.distance(q, keyFn(std::get<0>(results[i])));
                    if (i>0)
                        EXPECT(std::get<1>(results[i-1]) <= std::get<1>(results[i])) == true;
                }

                // TODO: test radius based query
            }

            // check the last k-nn result against a brute-force search
            std::partial_sort(
                linear.begin(), linear.begin() + maxK, linear.end(),
                [&] (int a, int b) {
                    return metricSpace.distance(q, keyFn(Helper::get(nodes, a))) < metricSpace.distance(q, keyFn(Helper::get(nodes, b)));
                });

            // std::cerr << "=== size: " << size << ", k: " << maxK << std::endl;
            for (std::size_t i=0 ; i<maxK ; ++i) {
                // if (i+1<maxK)
                //     std::cerr << i << "+1: " // << linear[i] << " "
                //               << std::setprecision(std::numeric_limits<Distance>::max_digits10)
                //               << distance(metric, q, keyFn(Helper::get(nodes, linear[i+1]))) << " "
                //               << std::get<1>(results[i+1])
                //               << std::endl;

                EXPECT(std::get<1>(results[i])) == metricSpace.distance(q, keyFn(Helper::get(nodes, linear[i])));
                // The following occasionally fails when two values
                // have the exact same distance, and get sorted
                // arbitrarily different:
                //
                // EXPECT(std::get<0>(results[i]) == Helper::get(nodes, linear[i])) == true;
            }
        }

        std::vector<T> list = nn.list();
        EXPECT(list.size()) == N;
    }
}


