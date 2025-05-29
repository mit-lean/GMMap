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
#ifndef NIGH_KDTREE_BATCH_HPP
#define NIGH_KDTREE_BATCH_HPP

#include "nigh_forward.hpp"
#include "metric/space.hpp"
#include "impl/nearest_base.hpp"
#include "impl/atom.hpp"
#include "impl/near_set.hpp"
#include "impl/kdtree_batch/strategy.hpp"
#include "impl/kdtree_batch/node.hpp"
#include "impl/kdtree_batch/leaf.hpp"
#include "impl/kdtree_batch/branch.hpp"
#include "impl/kdtree_batch/traversals.hpp"
#include "impl/kdtree_batch/nearest.hpp"
#include "impl/kdtree_batch/clear.hpp"

namespace unc::robotics::nigh {

    template <
        typename T,
        typename Space,
        typename KeyFn,
        typename Concurrency,
        std::size_t batchSize,
        typename Allocator>
    class Nigh<T, Space, KeyFn, Concurrency, KDTreeBatch<batchSize>, Allocator>
        : public impl::NearestBase<
            Nigh<T, Space, KeyFn, Concurrency, KDTreeBatch<batchSize>, Allocator>>
    {
        using Base = impl::NearestBase<Nigh>;
    public:
        using Key = typename Base::Key;
        using Distance = typename Base::Distance;

    private:
        static_assert(
            std::is_floating_point_v<Distance>,
            "distance must be a floating point type");

        static constexpr bool concurrentWrites = std::is_same_v<Concurrency, Concurrent>;
        static constexpr bool concurrentReads = !std::is_same_v<Concurrency, NoThreadSafety>;

        using Node = impl::kdtree_batch::Node<T, Space, Concurrency>;
        using Leaf = impl::kdtree_batch::Leaf<T, Space, Concurrency, batchSize>;

        template <typename, typename, typename, typename>
        friend class impl::kdtree_batch::Traversal;
        template <typename, typename>
        friend class impl::kdtree_batch::Nearest;
        template <typename>
        friend class impl::kdtree_batch::Clear;

        impl::Atom<Node*, concurrentWrites> root_{nullptr};
        impl::Atom<std::size_t, concurrentWrites> size_{0};

        impl::Atom<unsigned, concurrentWrites> depth_{0};

    public:
        Nigh(const Nigh&) = delete;
        Nigh(Nigh&& other);

        explicit Nigh(
            const Space& metric = Space(),
            const KeyFn& member = KeyFn(),
            Allocator allocator = Allocator())
            : Base(metric, member, allocator)
        {
        }

        ~Nigh();

        std::size_t size(std::memory_order order = std::memory_order_relaxed) const {
            return size_.load(order);
        }

        unsigned depth() const {
            return depth_.load(std::memory_order_relaxed);
        }

        void clear();
        void insert(const T& value);

        template <typename K>
        std::optional<std::pair<T, Distance>> nearest(const K& q) const;

        template <typename K>
        std::optional<T> nearest(const K& q, Distance* dist) const;

        template <typename Tuple, typename K, typename ResultAllocator>
        void nearest(
            std::vector<Tuple, ResultAllocator>& nbh,
            const K& q,
            std::size_t k,
            Distance maxRadius = std::numeric_limits<Distance>::infinity()) const;

        template <typename Fn>
        void visit(const Fn& fn) const {
            struct Visitor {
                const Nigh& tree_;
                impl::kdtree_batch::Traversal<Nigh> traversal_;
                Fn fn_;

                Visitor(const Nigh& tree, const Fn& fn) : tree_(tree), traversal_(tree.metricSpace()), fn_(fn) {}
                void operator() (const Node *node) {
                    if (!node->isLeaf()) {
                        traversal_.visit(*this, tree_.metricSpace(), node, node->axis());
                    } else {
                        const Leaf *leaf = static_cast<const Leaf*>(node);
                        int size = std::abs(leaf->size());
                        std::for_each(leaf->elements(), leaf->elements() + size, fn_);
                    }
                }
            };
            Visitor visitor(*this, fn);
            visitor(root_.load(std::memory_order_acquire));
        }

        std::vector<T> list() const {
            std::vector<T> result;
            result.reserve(static_cast<std::size_t>(size()*1.1));
            visit([&] (const T& t) { result.push_back(t); });
            return result;
        }
    };

    template <typename T, typename Space, typename KeyFn, typename Concurrency, std::size_t batchSize, typename Allocator>
    Nigh<T, Space, KeyFn, Concurrency, KDTreeBatch<batchSize>, Allocator>::Nigh(Nigh&& other)
        : root_(other.root_.exchange(nullptr))
        , size_(other.size_.exchange(0))
        , depth_(other.depth_.exchange(0))
    {
    }

    template <typename T, typename Space, typename KeyFn, typename Concurrency, std::size_t batchSize, typename Allocator>
    Nigh<T, Space, KeyFn, Concurrency, KDTreeBatch<batchSize>, Allocator>::~Nigh() {
        clear();
    }

    template <typename T, typename Space, typename KeyFn, typename Concurrency, std::size_t batchSize, typename Allocator>
    void Nigh<T, Space, KeyFn, Concurrency, KDTreeBatch<batchSize>, Allocator>::clear() {
        size_.store(0, std::memory_order_relaxed);
        depth_.store(0, std::memory_order_relaxed);
        if (Node *root = root_.exchange(nullptr, std::memory_order_release))
            (impl::kdtree_batch::Clear<Nigh>{*this})(root);
    }

    template <typename T, typename Space, typename KeyFn, typename Concurrency, std::size_t batchSize, typename Allocator>
    template <typename K>
    auto Nigh<T, Space, KeyFn, Concurrency, KDTreeBatch<batchSize>, Allocator>::nearest(const K& q) const
        -> std::optional<std::pair<T, Distance>>
    {
        impl::kdtree_batch::Nearest<Nigh, impl::Near1Set<T, Distance>> nearest(*this, q);
        if (Node *root = root_.load(std::memory_order_acquire))
            nearest(root);
        return nearest.result();
    }

    template <typename T, typename Space, typename KeyFn, typename Concurrency, std::size_t batchSize, typename Allocator>
    template <typename K>
    auto Nigh<T, Space, KeyFn, Concurrency, KDTreeBatch<batchSize>, Allocator>::nearest(const K& q, Distance *dist) const
        -> std::optional<T>
    {
        impl::kdtree_batch::Nearest<Nigh, impl::Near1Set<T, Distance>> nearest(*this, q);
        if (Node *root = root_.load(std::memory_order_acquire))
            nearest(root);
        return nearest.result(dist);
    }

    template <typename T, typename Space, typename KeyFn, typename Concurrency, std::size_t batchSize, typename Allocator>
    template <typename Tuple, typename K, typename ResultAllocator>
    void Nigh<T, Space, KeyFn, Concurrency, KDTreeBatch<batchSize>, Allocator>::nearest(
        std::vector<Tuple, ResultAllocator>& nbh,
        const K& q,
        std::size_t k,
        Distance maxRadius) const
    {
        impl::kdtree_batch::Nearest<Nigh, impl::NearKSet<Tuple, Distance, ResultAllocator>> nearest(*this, q, nbh, k, maxRadius);
        if (Node *root = root_.load(std::memory_order_acquire))
            nearest(root);
        nearest.sort();
    }

    template <typename T, typename Space, typename KeyFn, typename Concurrency, std::size_t batchSize, typename Allocator>
    void Nigh<T, Space, KeyFn, Concurrency, KDTreeBatch<batchSize>, Allocator>::insert(const T& q) {
        auto* p = &root_;

        impl::kdtree_batch::Traversal<Nigh> traversal(Base::metricSpace());
        const auto key = Base::getKey(q);
        assert(Space::isValid(key));
        unsigned depth = 1;
        for (;;++depth) {
            Node *node = p->load(std::memory_order_acquire);

            if (node == nullptr) {
                // TODO: allocator!
                Leaf *leaf = new Leaf(Base::metricSpace(), traversal, q, key);
                if (p->compare_exchange_strong(
                        node, static_cast<Node*>(leaf),
                        std::memory_order_release, std::memory_order_relaxed)) {
                    //std::cout << "new leaf" << std::endl;
                    break;
                }
                delete leaf;
            }

            if (node->isLeaf()) {
                Leaf *leaf = static_cast<Leaf*>(node);
                int n = leaf->size();
                if (concurrentWrites && (n < 0 || !leaf->tryLock(n))) {
                    impl::relax_cpu();
                    continue;
                }

                if (n < static_cast<int>(batchSize)) {
                    traversal.grow(Base::metricSpace(), leaf->region(), key);
                    leaf->put(n, q);

                    // std::cout << "inserted value at index " << n << std::endl;

                    // linearization point
                    leaf->setSize(n+1, std::memory_order_release);
                    break;
                }

                unsigned axis;
                // auto d =
                traversal.selectAxis(*this, Base::metricSpace(), leaf, key, &axis);
                // std::cerr << "splitting on axis " << axis << ", d = " << d << ", depth = " << depth << std::endl;
                node = traversal.split(*this, Base::metricSpace(), leaf, key, axis, traversal);

                // linearization point
                p->store(node, std::memory_order_release);
            }

            traversal.grow(Base::metricSpace(), node->region(), key);
            p = &traversal.follow(Base::metricSpace(), node, node->axis(), key);
        }

        size_.fetch_add(static_cast<std::size_t>(1), std::memory_order_relaxed);

        for (unsigned curDepth = depth_.load(std::memory_order_relaxed) ;
             curDepth < depth &&
            !depth_.compare_exchange_weak(curDepth, depth, std::memory_order_relaxed) ;)
            ;
    }
}

#endif
