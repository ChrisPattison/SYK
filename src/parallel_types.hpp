/* Copyright (c) 2016 C. Pattison
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
 
#pragma once
#include <mpi.h>
#include <type_traits>
#include <vector>
#include <cmath>

namespace parallel {
/** Container for the tree levels required in the heiarchial reductions.
 */
class Heirarchy {
    int levels_;
    int base_;
    std::vector<MPI_Comm> comms_;

    int rank(MPI_Comm comm);
public:
    Heirarchy();
    Heirarchy(const Heirarchy& source) = delete;
/** Builds a tree with FAN-IN base.
 */
    Heirarchy(int base);
/** Deletes frees communicators.
 */
    ~Heirarchy();
/** Const reference to the commuicator list.
 */
    const std::vector<MPI_Comm>& comms();
/** Total number of levels in the tree.
 */
    int global_levels();
/** Levels in the tree seen by current process.
 */
    int local_levels();
/** Gets FAN-IN of tree.
 */
    int base();
};

/** Container for data with a particular source or destination.
 */
template<typename T, typename = std::enable_if_t<std::is_trivially_copyable<T>::value, void>> struct Packet {
    int rank;
    std::vector<T> data;
};

/** Base class for asynchronous MPI operations.
 * The buffer is moved to buffer_ and the pointer passed to MPI.
 * On destruction the associated request is Wait'd so the buffer can be freed.
 */
template<typename T> class AsyncOp {
    friend class Mpi;
protected:
    MPI_Request request_;
    std::vector<T> buffer_;
public:
    AsyncOp();
    virtual ~AsyncOp();
/**
 * Copying would result result in the duplication of requests with only one instance actually holding the data.
 */
    AsyncOp(const AsyncOp&) = delete;
    AsyncOp(AsyncOp&& other);
/**
 * Copying would result result in the duplication of requests with only one instance actually holding the data.
 */
    AsyncOp& operator=(const AsyncOp&) = delete;
    AsyncOp& operator=(AsyncOp&& other);
/** Waits for completion of the aassociated request.
 */
    void Wait();
};

template<typename T> AsyncOp<T>::AsyncOp() {
    request_ = MPI_REQUEST_NULL;
}

template<typename T> AsyncOp<T>::AsyncOp(AsyncOp&& other) {
    buffer_.swap(other.buffer_);
    request_ = other.request_;
    other.request_ = MPI_REQUEST_NULL;
}

template<typename T> AsyncOp<T>::~AsyncOp() {
    Wait();
}

template<typename T> AsyncOp<T>& AsyncOp<T>::operator=(AsyncOp&& other) {
    buffer_.swap(other.buffer_);
    request_ = other.request_;
    other.request_ = MPI_REQUEST_NULL;
    return *this;
}

template<typename T> void AsyncOp<T>::Wait() {
    MPI_Wait(&request_, MPI_STATUS_IGNORE);
}
    
int Heirarchy::rank(MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

const std::vector<MPI_Comm>& Heirarchy::comms() {
    return comms_;
}

int Heirarchy::global_levels() {
    return levels_;
}

int Heirarchy::local_levels() {
    return comms_.size();
}

int Heirarchy::base() {
    return base_;
}

Heirarchy::Heirarchy() {
    levels_ = 0;
    base_ = 0;
    comms_ = { };
}

Heirarchy::Heirarchy(int base) {
    comms_ = { };
    base_ = base;
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    world_rank = rank(MPI_COMM_WORLD);

    MPI_Group mpi_world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &mpi_world_group);

    int level_stride = 1; // level_stride = base ^ level
    levels_ = std::ceil(std::log(world_size)/std::log(base_));
    for(int level = 0; world_rank % level_stride == 0 && level < levels_; ++level, level_stride *= base_) {
        MPI_Comm level_comm;
        MPI_Comm next_comm;
        MPI_Group level_group;
        int kmax = world_size / level_stride;

        // Build group
        std::vector<int> group_indices;
        group_indices.reserve(kmax);
        for(int k = 0; k <= kmax; ++k) {
            int rank = k*level_stride;
            if(rank < world_size) {
                group_indices.push_back(rank);
            }
        }
        MPI_Group_incl(mpi_world_group, group_indices.size(), group_indices.data(), &level_group);
        // Build communicator
        if(level_group != MPI_GROUP_NULL) {
            MPI_Comm_create_group(MPI_COMM_WORLD, level_group, 0, &level_comm);
            MPI_Comm_split(level_comm, rank(level_comm) / base_, world_rank, &next_comm);
            comms_.push_back(next_comm);
            MPI_Comm_free(&level_comm);
            MPI_Group_free(&level_group);
        }
    }
    MPI_Group_free(&mpi_world_group);
}

// fix this at some point
Heirarchy::~Heirarchy() {
    // for(auto comm : comms_) {
    //     MPI_Comm_free(&comm);
    // }
}
}