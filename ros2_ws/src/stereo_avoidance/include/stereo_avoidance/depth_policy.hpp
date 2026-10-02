#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace stereo_avoidance {

constexpr int kDepthHeight = 32;
constexpr int kDepthWidth = 48;
constexpr int kStateSize = 11;
constexpr int kActionSize = 4;

class DepthPolicy {
 public:
  explicit DepthPolicy(const std::string& model_path);
  std::array<float, kActionSize> infer(const std::vector<float>& depth,
                                       const std::array<float, kStateSize>& state);

 private:
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo memory_info_;
};

}  // namespace stereo_avoidance
