#pragma once
#include <vector>
#include <array>
#include <exception>
// Broken by elpa_wrapper
// #include <boost/uuid/sha1.hpp>

namespace util {

template<typename T>
std::array<std::uint32_t, 5> hash(const std::vector<T>& data) {
    throw std::exception("Not implemented");
    return std::array {0U,0U,0U,0U,0U};
}

}