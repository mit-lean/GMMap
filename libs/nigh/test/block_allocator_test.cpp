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

#include <nigh/impl/block_allocator.hpp>
#include <atomic>
#include "test.hpp"

template <typename T>
std::uintptr_t addr(const T* ptr) {
    return reinterpret_cast<std::uintptr_t>(ptr);
}

template <class T>
class TestAllocator : std::allocator<T> {
    using Base = std::allocator<T>;

    template <typename U>
    friend class TestAllocator;
    
    std::shared_ptr<std::atomic_uint> count_;
    std::shared_ptr<std::atomic_size_t> bytes_;
    
public:
    using value_type = typename Base::value_type;

    TestAllocator()
        : count_(std::make_shared<std::atomic_uint>(0))
        , bytes_(std::make_shared<std::atomic_size_t>(0))
    {
    }

    template <typename U>
    TestAllocator(const TestAllocator<U>& other)
        : Base(other)
        , count_(other.count_)
        , bytes_(other.bytes_)
    {
    }

    template <typename U>
    TestAllocator(TestAllocator<U>&& other)
        : Base(std::move(other))
        , count_(std::move(other.count_))
        , bytes_(std::move(other.bytes_))
    {
    }

    using Base::deallocate;
    
    typename Base::pointer allocate(std::size_t n) {
        typename Base::pointer ptr = Base::allocate(n);
        (*bytes_) += n * sizeof(T);
        ++(*count_);
        return ptr;
    }

    typename Base::pointer allocate(std::size_t n, void *hint) {
        typename Base::pointer ptr = Base::allocate(n, hint);
        (*bytes_) += n * sizeof(T);
        ++(*count_);
        return ptr;
    }

    std::size_t bytes() const {
        return bytes_->load();
    }

    unsigned count() const {
        return count_->load();
    }
};

TEST(packed_allocation) {
    using namespace unc::robotics::nigh::impl;
    BlockAllocator<1024> alloc;

    std::array<char, 8>* a = alloc.allocate<std::array<char, 8>>();
    std::array<char, 32>* b = alloc.allocate<std::array<char, 32>>();
    std::array<char, 64>* c = alloc.allocate<std::array<char, 64>>();

    EXPECT(a->data()+8 == b->data()) == true;
    EXPECT(b->data()+32) == c->data();
}


TEST(aligned_allocation) {
    using namespace unc::robotics::nigh::impl;
    BlockAllocator<1024> alloc;

    std::array<char, 3>* a = alloc.allocate<std::array<char, 3>>();
    std::array<double, 4>* b = alloc.allocate<std::array<double, 4>>();

    EXPECT(addr(a->data()) < addr(b->data())) == true;
    EXPECT(addr(b->data()) & 8) == 0;
}

TEST(block_alloc) {
    using namespace unc::robotics::nigh::impl;
    BlockAllocator<1024> alloc;
    using Item = std::array<double, 1024/sizeof(double) - 1>;
    
    Item *a = alloc.allocate<Item>();
    Item *b = alloc.allocate<Item>();
    EXPECT(addr(a) + 1024 < addr(b) ||
           addr(b) + 1024 < addr(a)) == true;
}

TEST(mark_rollback_in_block) {
    using namespace unc::robotics::nigh::impl;
    BlockAllocator<1024> alloc;
    using Item = std::array<double, 4>;

    auto markA = alloc.mark();
    Item *a0 = alloc.allocate<Item>();
    Item *b0 = alloc.allocate<Item>();
    
    alloc.rollback(markA);
    Item *a1 = alloc.allocate<Item>();
    EXPECT(a0 == a1) == true;

    auto markB = alloc.mark();
    Item *b1 = alloc.allocate<Item>();
    EXPECT(b0 == b1) == true;
    
    Item *c1 = alloc.allocate<Item>();

    alloc.rollback(markB);
    Item *b2 = alloc.allocate<Item>();
    EXPECT(b1 == b2) == true;
}

TEST(mark_rollback_across_blocks) {
    using namespace unc::robotics::nigh::impl;
    using Alloc = TestAllocator<char>;
    using Item = std::array<double, 1024/sizeof(double) / 3>;

    Alloc counters;
    BlockAllocator<1024, Alloc> alloc(counters);
    
    Item *a0 = alloc.allocate<Item>();
    auto markB = alloc.mark();
    Item *b0 = alloc.allocate<Item>();
    Item *c0 = alloc.allocate<Item>();

    // this should be in the next block
    Item *d0 = alloc.allocate<Item>();

    alloc.rollback(markB);
    Item *b1 = alloc.allocate<Item>();
    Item *c1 = alloc.allocate<Item>();

    // this will be reallocated in second block
    Item *d1 = alloc.allocate<Item>();
    EXPECT(b0 == b1) == true;
    EXPECT(c0 == c1) == true;
    EXPECT(d0 == d1) == true;

    EXPECT(counters.count()) == 2;

    // the linked list allocates (a block of the specified block size)
    // + (a pointer) for each node.
    EXPECT(counters.bytes()) == (1024 + sizeof(void*))*2;
}

