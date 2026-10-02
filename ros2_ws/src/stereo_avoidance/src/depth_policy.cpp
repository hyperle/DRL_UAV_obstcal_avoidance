#include "stereo_avoidance/depth_policy.hpp"

#include <algorithm>
#include <stdexcept>

namespace stereo_avoidance {

DepthPolicy::DepthPolicy(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "stereo_avoidance"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), options);
}

std::array<float, kActionSize> DepthPolicy::infer(
    const std::vector<float>& depth,
    const std::array<float, kStateSize>& state) {
  if (depth.size() != static_cast<std::size_t>(kDepthHeight * kDepthWidth)) {
    throw std::invalid_argument("Depth tensor must contain 32x48 values");
  }

  std::array<int64_t, 4> depth_shape{1, 1, kDepthHeight, kDepthWidth};
  std::array<int64_t, 2> state_shape{1, kStateSize};
  std::array<int64_t, 2> action_shape{1, kActionSize};
  auto depth_tensor = Ort::Value::CreateTensor<float>(
      memory_info_, const_cast<float*>(depth.data()), depth.size(), depth_shape.data(),
      depth_shape.size());
  auto state_tensor = Ort::Value::CreateTensor<float>(
      memory_info_, const_cast<float*>(state.data()), state.size(), state_shape.data(),
      state_shape.size());

  const char* input_names[] = {"depth", "state"};
  const char* output_names[] = {"action"};
  std::array<Ort::Value, 2> inputs{std::move(depth_tensor), std::move(state_tensor)};
  auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(),
                               output_names, 1);
  const float* output = outputs.front().GetTensorData<float>();
  std::array<float, kActionSize> action{};
  std::copy(output, output + kActionSize, action.begin());
  for (auto& value : action) {
    value = std::clamp(value, -1.0F, 1.0F);
  }
  return action;
}

}  // namespace stereo_avoidance
