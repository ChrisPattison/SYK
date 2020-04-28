#pragma once
#include <vector>
#include <array>
#include <boost/uuid/sha1.hpp>

namespace util {

template<typename T>
std::array<std::uint32_t, 5> hash(const std::vector<T>& data) {
    std::uint32_t digest[5];
    boost::uuids::detail::sha1 hash;
    hash.process_bytes(data.data(), data.size()*sizeof(T));
    hash.get_digest(digest);
    return std::array {digest[0], digest[1], digest[2], digest[3], digest[4]};
}

}