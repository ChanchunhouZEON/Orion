#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace parlayANN {

inline size_t checked_add(size_t a, size_t b) {
  if (b > std::numeric_limits<size_t>::max() - a)
    throw std::overflow_error("allocation size overflow");
  return a + b;
}

inline size_t checked_mul(size_t a, size_t b) {
  if (a && b > std::numeric_limits<size_t>::max() / a)
    throw std::overflow_error("allocation size overflow");
  return a * b;
}

inline size_t align_bytes(size_t bytes, size_t alignment) {
  if (!alignment || (alignment & (alignment - 1)))
    throw std::invalid_argument("alignment must be a power of two");
  return checked_add(bytes, alignment - 1) / alignment * alignment;
}

// posix_memalign accepts sizes that are not multiples of the alignment.
// Huge pages are a performance hint; allocation failure is always fatal.
inline void *allocate_aligned(size_t bytes) {
  void *ptr = nullptr;
#ifdef MADV_HUGEPAGE
  constexpr size_t alignment = 2 * 1024 * 1024;
#else
  constexpr size_t alignment = 64;
#endif
  if (posix_memalign(&ptr, alignment, bytes ? bytes : alignment) != 0)
    throw std::bad_alloc();
#ifdef MADV_HUGEPAGE
  madvise(ptr, bytes, MADV_HUGEPAGE);
#endif
  return ptr;
}

inline void read_exact(std::istream &input, void *destination, size_t bytes) {
  if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()))
    throw std::overflow_error("read size exceeds stream limit");
  if (!input.read(static_cast<char *>(destination), bytes))
    throw std::runtime_error("truncated input or read failure");
}

inline uint32_t read_u32(std::istream &input) {
  unsigned char b[4];
  read_exact(input, b, 4);
  return uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 |
         uint32_t(b[3]) << 24;
}

struct BinaryShape {
  size_t count;
  size_t dimension;
};

inline BinaryShape inspect_binary(const char *path, size_t coordinate_bytes) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    throw std::runtime_error(std::string("cannot open ") + path);
  const auto length = input.tellg();
  if (length < 8)
    throw std::runtime_error("invalid binary header");
  input.seekg(0);
  BinaryShape shape{read_u32(input), read_u32(input)};
  if (!shape.count || !shape.dimension || shape.count == UINT32_MAX)
    throw std::runtime_error("empty input or reserved point count");
  const size_t payload =
      checked_mul(checked_mul(shape.count, shape.dimension), coordinate_bytes);
  if (static_cast<uint64_t>(length) != checked_add(8, payload))
    throw std::runtime_error("binary header/file length mismatch");
  return shape;
}

// Hooks allow deterministic short-write/flush/close fault injection in tests.
// A close hook must consume the FILE*, even when returning an error.
struct OutputHooks {
  decltype(&std::fwrite) write = &std::fwrite;
  decltype(&std::fflush) flush = &std::fflush;
  decltype(&std::fclose) close = &std::fclose;
};

class AtomicOutput {
public:
  explicit AtomicOutput(const std::string &destination, OutputHooks hooks = {})
      : destination_(destination), hooks_(hooks) {
    std::string pattern = destination + ".tmp.XXXXXX";
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    int fd = mkstemp(name.data());
    if (fd < 0)
      throw std::runtime_error("cannot create output temporary file: " +
                               destination);
    temporary_ = name.data();
    file_ = fdopen(fd, "wb");
    if (!file_) {
      ::close(fd);
      std::remove(temporary_.c_str());
      throw std::runtime_error("cannot open output stream");
    }
  }

  AtomicOutput(const AtomicOutput &) = delete;
  AtomicOutput &operator=(const AtomicOutput &) = delete;

  ~AtomicOutput() {
    if (file_)
      std::fclose(file_);
    if (!temporary_.empty())
      std::remove(temporary_.c_str());
  }

  void write(const void *data, size_t bytes) {
    flush_words();
    if (bytes && hooks_.write(data, 1, bytes, file_) != bytes)
      throw std::runtime_error("output short write (possibly disk full)");
  }

  void word(uint32_t value) {
    unsigned char bytes[4] = {static_cast<unsigned char>(value),
                              static_cast<unsigned char>(value >> 8),
                              static_cast<unsigned char>(value >> 16),
                              static_cast<unsigned char>(value >> 24)};
    words_.insert(words_.end(), bytes, bytes + 4);
    if (words_.size() >= 1024 * 1024)
      flush_words();
  }

  void publish() {
    flush_words();
    if (hooks_.flush(file_) != 0 || fsync(fileno(file_)) != 0)
      throw std::runtime_error("output flush/sync failed");
    FILE *closing = file_;
    file_ = nullptr;
    if (hooks_.close(closing) != 0)
      throw std::runtime_error("output close failed");
    if (std::rename(temporary_.c_str(), destination_.c_str()) != 0)
      throw std::runtime_error("output rename failed");
    temporary_.clear();
  }

private:
  void flush_words() {
    if (!words_.empty() &&
        hooks_.write(words_.data(), 1, words_.size(), file_) != words_.size())
      throw std::runtime_error("output short write (possibly disk full)");
    words_.clear();
  }

  std::vector<unsigned char> words_;
  std::string destination_;
  std::string temporary_;
  FILE *file_ = nullptr;
  OutputHooks hooks_;
};

} // namespace parlayANN
