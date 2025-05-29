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
#ifndef NIGH_IMPL_BLOCK_ALLOCATOR_HPP
#define NIGH_IMPL_BLOCK_ALLOCATOR_HPP

#include <array>
#include <cassert>
#include <forward_list>
#include <memory>

namespace unc::robotics::nigh::impl {
    /**
     * A BlockAllocator allocates objects from a pre-allocated list of
     * blocks.  This class implements the ability to quickly rollback
     * to a previous allocation point using the mark()/rollback()
     * methods.  Objects allocated on this pool will not have their
     * destructors called.
     *
     * The default block size is 16k - sizeof a pointer to attempt to
     * have each block fit into a power-of-two allocation.
     */
    template <
        std::size_t blockSize = 16384 - sizeof(void*),
        class Allocator = std::allocator<char>>
    class BlockAllocator {
        using Block = std::array<char, blockSize>;
        using RebindAlloc = typename std::allocator_traits<Allocator>
            ::template rebind_alloc<Block>;
        using BlockList = std::forward_list<Block, RebindAlloc>;
        using BlockIter = typename BlockList::iterator;

        BlockList blockList_;
        BlockIter currentBlock_;
        std::size_t offset_{0};

    public:
        class Marker {
            friend class BlockAllocator;
            
            BlockIter block_;
            std::size_t offset_;

            Marker(BlockIter block, std::size_t offset)
                : block_(block)
                , offset_(offset)
            {
            }
            
        public:
            Marker() = default;
            Marker(const Marker&) = default;
        };
        
        BlockAllocator(Allocator alloc = Allocator())
            : blockList_(alloc)
        {
            blockList_.emplace_front();
            currentBlock_ = blockList_.begin();
        }

        BlockAllocator(const BlockAllocator&) = delete;
        
        BlockAllocator(BlockAllocator&& other)
            : blockList_(std::move(other.blockList_))
            , currentBlock_(std::move(other.currentBlock_))
            , offset_(other.offset_)
        {
        }

        Marker mark() {
            return Marker(currentBlock_, offset_);
        }

        void rollback(Marker marker) {
            currentBlock_ = marker.block_;
            offset_ = marker.offset_;
        }
            
        template <class E, class ... Args>
        E* allocate(Args&& ... args) {
            static_assert(
                std::is_trivially_destructible_v<E>,
                "allocated classes must be trivally destructible, or resource leak could occur");
            
            for ( ;; offset_ = 0) {
                void *ptr = currentBlock_->data() + offset_;
                std::size_t space = blockSize - offset_;

                if (std::align(alignof(E), sizeof(E), ptr, space)) {
                    offset_ = blockSize - space + sizeof(E);
                    return new (reinterpret_cast<E*>(ptr)) E (std::forward<Args>(args)...);
                }

                // not enough space in this block, we have to advance
                // to the next block.  This should also mean that
                // offset_ > 0, otherwise we're trying to allocate
                // something too big for the block size.
                assert(offset_ > 0);

                BlockIter prev = currentBlock_++;
                if (currentBlock_ == blockList_.end())
                    currentBlock_ = blockList_.emplace_after(prev);
            }
        }
    };

    // Specialization to allow BlockAllocator<0, A> to mean the
    // default allocation size.
    template <typename Alloc>
    class BlockAllocator<0, Alloc> : public BlockAllocator<16384 - sizeof(void*), Alloc> {
        using Base = BlockAllocator<16384 - sizeof(void*), Alloc>;
    public:
        using Base::Base;
    };
}

#endif

