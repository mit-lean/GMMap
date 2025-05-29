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
#ifndef NIGH_TEST_BENCH_TEMPLATE_HPP
#define NIGH_TEST_BENCH_TEMPLATE_HPP

#include <nigh/metric/lp.hpp>
#include <nigh/metric/so2.hpp>
#include <nigh/metric/so3.hpp>
#include <nigh/metric/cartesian.hpp>
#include <nigh/kdtree_batch.hpp>
#include <nigh/kdtree_median.hpp>
#include <nigh/linear.hpp>
#include <nigh/gnat.hpp>
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <getopt.h>
#include "sampler_lp.hpp"
#include "sampler_so2.hpp"
#include "sampler_so3.hpp"
#include "sampler_scaled.hpp"
#include "sampler_cartesian.hpp"

namespace nigh_test {

    template <typename C>
    struct Name;

    template <>
    struct Name<double> {
        static std::string name() {
            return "double";
        }
    };

    template <std::intmax_t num, std::intmax_t den>
    struct Name<std::ratio<num, den>> {
        static std::string name() {
            return std::to_string(num) + ":" + std::to_string(den);
        }
    };

    template <typename S, int r, int c>
    struct Name<Eigen::Matrix<S, r, c>> {
        static std::string name() {
            return "Eigen::Matric<" + Name<S>::name() + ", " + std::to_string(r) + ", " + std::to_string(c) + ">";
        }
    };

    template <typename S>
    struct Name<Eigen::Quaternion<S>> {
        static std::string name() {
            return "Eigen::Quaternion<" + Name<S>::name() + ">";
        }
    };

    template <typename F, typename ... R>
    struct Name<std::tuple<F, R...>> {
        static std::string name() {
            return "std::tuple<" + Name<F>::name() + ((", " + Name<R>::name()) + ...) + ">";
        }
    };

    template <int p>
    struct Name<unc::robotics::nigh::metric::LP<p>> {
        static std::string name() {
            return "LP<" + std::to_string(p) + ">";
        }
    };

    template <>
    struct Name<unc::robotics::nigh::metric::SO3> {
        static std::string name() {
            return "SO3";
        }
    };

    template <int p>
    struct Name<unc::robotics::nigh::metric::SO2<p>> {
        static std::string name() {
            return "SO2<" + std::to_string(p) + ">";
        }
    };

    template <class M, class W>
    struct Name<unc::robotics::nigh::metric::Scaled<M, W>> {
        static std::string name() {
            return "Scaled<" + Name<M>::name() + ", " + Name<W>::name() + ">";
        }
    };

    template <typename F, typename ... R>
    struct Name<unc::robotics::nigh::metric::Cartesian<F, R...>> {
        static std::string name() {
            return "Cartesian<" + Name<F>::name() + ((", " + Name<R>::name()) + ...) + ">";
        }
    };

    template <>
    struct Name<unc::robotics::nigh::Linear> {
        static std::string name() {
            return "Linear";
        }
    };

    template <>
    struct Name<unc::robotics::nigh::GNAT<>> {
        static std::string name() {
            return "GNAT<>";
        }
    };


    template <std::size_t batchSize>
    struct Name<unc::robotics::nigh::KDTreeBatch<batchSize>> {
        static std::string name() {
            return "KDTreeBatch<" + std::to_string(batchSize) + ">";
        }
    };

    template <std::size_t minTreeSize, std::size_t linearSearchSize>
    struct Name<unc::robotics::nigh::KDTreeMedian<minTreeSize, linearSearchSize>> {
        static std::string name() {
            return "KDTreeMedian<" + std::to_string(minTreeSize) + ", "
                + std::to_string(linearSearchSize) + ">";
        }
    };

    template <>
    struct Name<unc::robotics::nigh::Concurrent> {
        static std::string name() {
            return "Concurrent";
        }
    };

    template <>
    struct Name<unc::robotics::nigh::ConcurrentRead> {
        static std::string name() {
            return "ConcurrentRead";
        }
    };

    template <>
    struct Name<unc::robotics::nigh::NoThreadSafety> {
        static std::string name() {
            return "NoThreadSafety";
        }
    };

    struct Identity {
        template <typename T>
        constexpr const T& operator () (const T& q) const { return q; }
    };

    template <typename Clock>
    double nanosPerClock(typename Clock::duration stepDuration) {
        typename Clock::duration elapsed;
        std::size_t count = 0;
        auto start = Clock::now();
        do {
            ++count;
        } while ((elapsed = Clock::now() - start) < stepDuration);
        double nanos = std::chrono::duration<double, std::nano>(elapsed).count();
        double t = nanos / count;
        // std::cerr << "Overhead of Clock::now() " << t << " ns after " << count << " calls" << std::endl;
        return t;
    }

    template <typename T, typename Space, typename KeyFn, typename Concurrency, typename Strategy>
    void runBench(
        int argc,
        char *argv[],
        const Space& space = Space(),
        const KeyFn& keyFn = KeyFn(),
        const Concurrency& = Concurrency{},
        const Strategy& = Strategy{})
    {
        using Clock = std::chrono::steady_clock;
        using Key = typename Space::Type;
        using Metric = typename Space::Metric;
        using Distance = typename Space::Distance;
        using namespace std::literals::chrono_literals;
        using namespace unc::robotics::nigh;

        static constexpr std::size_t nTrees = 4;
        static constexpr std::size_t nQueries = 1024*1024;

        std::size_t N = 100000; // std::is_same_v<Linear, Strategy> ? 10000 : 100000;
        std::size_t K = 20;
        Clock::duration stepDuration = 100ms;

        for (int opt ; (opt = getopt(argc, argv, "n:k:d:")) != -1 ; ) {
            switch (opt) {
            case 'n':
                N = std::atoi(optarg);
                break;
            case 'k':
                K = std::atoi(optarg);
                break;
            case 'd':
                stepDuration = std::chrono::milliseconds(std::atoi(optarg));
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " [-n max-nn-size] [-k query-size]" << std::endl;
                return;
            }
        }


        std::vector<Nigh<T, Space, KeyFn, Concurrency, Strategy>> nns;
        nns.reserve(nTrees);
        while (nns.size() < nTrees)
            nns.emplace_back(space, keyFn);

        Sampler<Key, Metric> sampler(space);
        std::mt19937_64 rng;

        std::vector<std::remove_pointer_t<T>> nodes;
        nodes.reserve(N * nTrees);

        std::vector<Key> queries;
        queries.reserve(nQueries);
        while (queries.size() < nQueries)
            queries.push_back(sampler(rng));

        double stepsPerExp = 10;
        std::size_t size = 10;

        unsigned statIndex = static_cast<unsigned>(std::log(size) / std::log(10) * stepsPerExp);

        std::vector<std::pair<T, Distance>> nbh;
        nbh.reserve(K+1);

        double clockTime = nanosPerClock<Clock>(stepDuration);

        std::cout << "# State = " << Name<typename Space::Type>::name() << std::endl;
        std::cout << "# Metric = " << Name<typename Space::Metric>::name() << std::endl;
        std::cout << "# Strategy = " << Name<Strategy>::name() << std::endl;
        std::cout << "# Concurrency = " << Name<Concurrency>::name() << std::endl;
        std::cout << "# k = " << K << std::endl;
        std::cout << "# size nanos_elapsed query_count ms_per_query" << std::endl;

        std::size_t queryNo = 0;
        for (;;) {
            for (std::size_t i=0 ; i<nTrees ; ++i) {
                while (nns[i].size() < size) {
                    nodes.emplace_back(sampler(rng));
                    nns[i].insert(nodes.back());
                }
            }

            std::size_t prevQueryNo = queryNo;
            Clock::duration elapsed;
            auto start = Clock::now();
            do {
                nns[queryNo % nTrees].nearest(nbh, queries[queryNo % nQueries], K);
                ++queryNo;
            } while ((elapsed = Clock::now() - start) < stepDuration);

            std::size_t queryCount = (queryNo - prevQueryNo);
            auto nanos = std::chrono::duration<long long, std::nano>(elapsed).count()
                - static_cast<long long>(clockTime * queryCount);

            std::cout << size
                      << '\t' << nanos
                      << '\t' << queryCount
                      << '\t' << (nanos * 1e-6 / queryCount)
                      << std::endl;

            if (size >= N)
                break;

            std::size_t prevSize = size;
            while ((size = static_cast<std::size_t>(std::pow(10, ++statIndex/stepsPerExp) + 0.5)) <= prevSize)
                ;
            size = std::min(size, N);
        }
    }

}

#endif
