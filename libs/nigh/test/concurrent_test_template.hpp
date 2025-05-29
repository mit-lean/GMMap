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
#include <nigh/metric/lp.hpp>
#include <nigh/metric/so2.hpp>
#include <nigh/metric/so3.hpp>
#include <nigh/metric/cartesian.hpp>
#include <nigh/metric/scaled.hpp>
#include <nigh/kdtree_batch.hpp>
#include <nigh/linear.hpp>
#include <nigh/gnat.hpp>
#include <random>
#include <algorithm>
#include <thread>
#include "sampler_lp.hpp"
#include "sampler_so2.hpp"
#include "sampler_so3.hpp"
#include "sampler_scaled.hpp"
#include "sampler_cartesian.hpp"

namespace nigh_test {
    template <typename T>
    struct TestNode {
        unsigned thread_;
        std::size_t index_;
        T key_;

        TestNode(unsigned t, std::size_t i, const T& q)
            : thread_(t), index_(i), key_(q)
        {
        }
    };

    template <typename T>
    struct TestNodeKey {
        const T& operator() (const TestNode<T>& n) const {
            return n.key_;
        }
    };

    template <typename NN>
    struct TestThread {
        NN& nn_;
        unsigned no_;
        std::size_t n_;
        std::size_t k_;

        std::exception_ptr eptr_;
        std::thread thread_;

        void run() {
            try {
                // std::cerr << "starting thread " << no_ << std::endl;

                using Space = typename NN::Space;
                using Key = typename Space::Type;
                using Node = TestNode<Key>;
                using Distance = typename Space::Distance;

                std::mt19937_64 rng(no_);
                Sampler<Key, typename Space::Metric> sampler(nn_.metricSpace());

                std::vector<std::tuple<Node, Distance>> nbh;
                std::set<std::pair<unsigned, unsigned>> set;

                for (std::size_t i=0 ; i<n_ ; ++i) {
                    Key q = sampler(rng);
                    nn_.insert(Node(no_, i, q));
                    nn_.nearest(nbh, q, k_);

                    // make sure that nearest does not return the
                    // same value twice
                    set.clear();
                    for (auto& n : nbh) {
                        bool inserted = set.insert(std::make_pair(std::get<0>(n).thread_, std::get<0>(n).index_)).second;
                        EXPECT(inserted) == true;
                    }

                    // make sure that nearest returns the inserted
                    // value
                    EXPECT(set.find(std::make_pair(no_, i)) == set.end()) == false;
                }

                // std::cerr << "ending thread " << no_ << std::endl;
            } catch (...) {
                eptr_ = std::current_exception();
            }
        }

        TestThread(TestThread&& other)
            : nn_(other.nn_)
        {
            abort();
        }

        TestThread(NN& nn, unsigned no, std::size_t n, std::size_t k)
            : nn_(nn), no_(no), n_(n), k_(k), thread_(&TestThread::run, this)
        {
        }
    };

    template <typename Strategy, typename Space>
    void runConcurrentTest(
        const Space& space = Space(),
        std::size_t N = 10000,
        std::size_t K = 20)
    {
        using namespace unc::robotics::nigh;
        using Key = typename Space::Type;
        // using Distance = typename Space::Distance;

        using NN = Nigh<TestNode<Key>, Space, TestNodeKey<Key>, Concurrent, Strategy>;

        NN nn(space);

        unsigned nThreads = std::thread::hardware_concurrency();
        EXPECT(nThreads) > 1u;

        std::vector<TestThread<NN>> threads;
        threads.reserve(nThreads);
        for (unsigned i=0 ; i<nThreads ; ++i)
            threads.emplace_back(nn, i, N, K);

        for (auto& t : threads)
            t.thread_.join();

        for (auto& t : threads)
            if (t.eptr_)
                std::rethrow_exception(t.eptr_);

        EXPECT(nn.size()) == N*nThreads;

        std::vector<TestNode<Key>> list = nn.list();
        EXPECT(list.size()) == N*nThreads;
        std::vector<std::vector<bool>> bits;
        bits.reserve(nThreads);
        for (unsigned i=0 ; i<nThreads ; ++i)
            bits.emplace_back(N, false);

        for (auto& n : list) {
            EXPECT(n.thread_) < nThreads;
            EXPECT(n.index_) < N;
            EXPECT(!!bits[n.thread_][n.index_]) == false;
            bits[n.thread_][n.index_] = true;
        }
    }
}
