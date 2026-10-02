#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "cv_bridge/cv_bridge.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "stereo_avoidance/depth_policy.hpp"
#include "stereo_avoidance/flight_controller_adapter.hpp"

namespace stereo_avoidance {
namespace {

constexpr float kMaxDepthM = 8.0F;
constexpr float kMaxGoalRangeM = 10.0F;
constexpr float kMaxForwardMps = 1.5F;
constexpr float kMaxLateralMps = 0.8F;
constexpr float kMaxVerticalMps = 0.6F;
constexpr float kMaxYawRateRadps = 1.2F;

double yaw_from_quaternion(const geometry_msgs::msg::Quaternion& q) {
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

double age_seconds(const rclcpp::Time& now, const builtin_interfaces::msg::Time& stamp) {
  return (now - rclcpp::Time(stamp)).seconds();
}

}  // namespace

RosTopicFlightControllerAdapter::RosTopicFlightControllerAdapter(
    const rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr& publisher,
    const rclcpp::Clock::SharedPtr& clock,
    const std::string& frame_id)
    : publisher_(publisher), clock_(clock), frame_id_(frame_id) {}

void RosTopicFlightControllerAdapter::publish(const BodyVelocityCommand& command) {
  geometry_msgs::msg::TwistStamped message;
  message.header.stamp = clock_->now();
  message.header.frame_id = frame_id_;
  message.twist.linear.x = command.forward_mps;
  message.twist.linear.y = command.left_mps;
  message.twist.linear.z = command.up_mps;
  message.twist.angular.z = command.yaw_rate_radps;
  publisher_->publish(message);
}

void RosTopicFlightControllerAdapter::hold() { publish(BodyVelocityCommand{}); }

class StereoAvoidanceNode final : public rclcpp::Node {
 public:
  StereoAvoidanceNode() : Node("stereo_avoidance_node") {
    const auto model_path = declare_parameter<std::string>("model_path", "depth_policy.onnx");
    const auto depth_topic = declare_parameter<std::string>("depth_topic", "/stereo/depth");
    const auto odometry_topic = declare_parameter<std::string>("odometry_topic", "/uav/odometry");
    const auto goal_topic = declare_parameter<std::string>("goal_topic", "/uav/goal");
    const auto command_topic = declare_parameter<std::string>("command_topic", "/avoidance/setpoint");
    const auto frame_id = declare_parameter<std::string>("body_frame_id", "base_link");
    const auto policy_hz = declare_parameter<double>("policy_hz", 15.0);
    const auto stale_timeout_ms = declare_parameter<int>("stale_timeout_ms", 250);
    if (policy_hz <= 0.0 || stale_timeout_ms <= 0) {
      throw std::invalid_argument("policy_hz and stale_timeout_ms must be positive");
    }

    policy_ = std::make_unique<DepthPolicy>(model_path);
    command_publisher_ = create_publisher<geometry_msgs::msg::TwistStamped>(command_topic, 10);
    flight_adapter_ = std::make_unique<RosTopicFlightControllerAdapter>(
        command_publisher_, get_clock(), frame_id);

    depth_subscriber_ = create_subscription<sensor_msgs::msg::Image>(
        depth_topic, rclcpp::SensorDataQoS(),
        std::bind(&StereoAvoidanceNode::on_depth, this, std::placeholders::_1));
    odometry_subscriber_ = create_subscription<nav_msgs::msg::Odometry>(
        odometry_topic, rclcpp::SensorDataQoS(),
        std::bind(&StereoAvoidanceNode::on_odometry, this, std::placeholders::_1));
    goal_subscriber_ = create_subscription<geometry_msgs::msg::PointStamped>(
        goal_topic, 10,
        std::bind(&StereoAvoidanceNode::on_goal, this, std::placeholders::_1));

    const auto period = std::chrono::duration<double>(1.0 / policy_hz);
    policy_timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::milliseconds>(period),
        std::bind(&StereoAvoidanceNode::run_policy, this));
    stale_timeout_seconds_ = static_cast<double>(stale_timeout_ms) / 1000.0;
    RCLCPP_INFO(get_logger(), "Depth policy loaded from '%s'", model_path.c_str());
  }

 private:
  void on_depth(const sensor_msgs::msg::Image::ConstSharedPtr message) {
    try {
      auto input = cv_bridge::toCvShare(message);
      cv::Mat depth_m;
      if (message->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        depth_m = input->image;
      } else if (message->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
        input->image.convertTo(depth_m, CV_32FC1, 0.001);
      } else {
        throw std::runtime_error("Depth encoding must be 32FC1 meters or 16UC1 millimeters");
      }
      cv::Mat resized;
      cv::resize(depth_m, resized, cv::Size(kDepthWidth, kDepthHeight), 0.0, 0.0,
                 cv::INTER_AREA);
      std::vector<float> normalized(static_cast<std::size_t>(kDepthHeight * kDepthWidth));
      for (int row = 0; row < kDepthHeight; ++row) {
        const auto* pixels = resized.ptr<float>(row);
        for (int column = 0; column < kDepthWidth; ++column) {
          float value = pixels[column];
          if (!std::isfinite(value) || value <= 0.0F) {
            value = kMaxDepthM;
          }
          value = std::clamp(value, 0.0F, kMaxDepthM);
          normalized[static_cast<std::size_t>(row * kDepthWidth + column)] = value / kMaxDepthM;
        }
      }
      std::lock_guard<std::mutex> lock(data_mutex_);
      depth_ = std::move(normalized);
      depth_stamp_ = message->header.stamp;
      have_depth_ = true;
    } catch (const std::exception& error) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "Depth preprocessing failed: %s",
                           error.what());
    }
  }

  void on_odometry(const nav_msgs::msg::Odometry::ConstSharedPtr message) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    odometry_ = *message;
    have_odometry_ = true;
  }

  void on_goal(const geometry_msgs::msg::PointStamped::ConstSharedPtr message) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    goal_ = *message;
    have_goal_ = true;
  }

  void run_policy() {
    std::vector<float> depth;
    nav_msgs::msg::Odometry odometry;
    geometry_msgs::msg::PointStamped goal;
    builtin_interfaces::msg::Time depth_stamp;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (!have_depth_ || !have_odometry_ || !have_goal_) {
        flight_adapter_->hold();
        return;
      }
      depth = depth_;
      odometry = odometry_;
      goal = goal_;
      depth_stamp = depth_stamp_;
    }

    const auto now = get_clock()->now();
    const double image_age = age_seconds(now, depth_stamp);
    const double odometry_age = age_seconds(now, odometry.header.stamp);
    // Navigation goals may be latched/static messages; unlike sensor data,
    // their age alone does not make the target invalid.
    if (image_age < 0.0 || odometry_age < 0.0 ||
        image_age > stale_timeout_seconds_ || odometry_age > stale_timeout_seconds_) {
      flight_adapter_->hold();
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
                           "Input data stale; publishing hold setpoint");
      return;
    }
    if (!goal.header.frame_id.empty() && !odometry.header.frame_id.empty() &&
        goal.header.frame_id != odometry.header.frame_id) {
      flight_adapter_->hold();
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 3000,
                            "Goal and odometry must use the same world frame");
      return;
    }

    const auto& pose = odometry.pose.pose;
    const double yaw = yaw_from_quaternion(pose.orientation);
    const double dx = goal.point.x - pose.position.x;
    const double dy = goal.point.y - pose.position.y;
    const double dz = goal.point.z - pose.position.z;
    const double goal_forward = std::cos(yaw) * dx + std::sin(yaw) * dy;
    const double goal_left = -std::sin(yaw) * dx + std::cos(yaw) * dy;
    const auto& velocity = odometry.twist.twist;
    std::array<float, kStateSize> state{
        static_cast<float>(std::clamp(goal_forward / kMaxGoalRangeM, -1.0, 1.0)),
        static_cast<float>(std::clamp(goal_left / kMaxGoalRangeM, -1.0, 1.0)),
        static_cast<float>(std::clamp(dz / kMaxGoalRangeM, -1.0, 1.0)),
        static_cast<float>(std::clamp(velocity.linear.x / kMaxForwardMps, -1.0, 1.0)),
        static_cast<float>(std::clamp(velocity.linear.y / kMaxLateralMps, -1.0, 1.0)),
        static_cast<float>(std::clamp(velocity.linear.z / kMaxVerticalMps, -1.0, 1.0)),
        static_cast<float>(std::clamp(velocity.angular.z / kMaxYawRateRadps, -1.0, 1.0)),
        previous_action_[0], previous_action_[1], previous_action_[2], previous_action_[3]};

    try {
      previous_action_ = policy_->infer(depth, state);
      flight_adapter_->publish(BodyVelocityCommand{
          previous_action_[0] * kMaxForwardMps,
          previous_action_[1] * kMaxLateralMps,
          previous_action_[2] * kMaxVerticalMps,
          previous_action_[3] * kMaxYawRateRadps});
    } catch (const std::exception& error) {
      flight_adapter_->hold();
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 3000, "Policy inference failed: %s",
                            error.what());
    }
  }

  std::unique_ptr<DepthPolicy> policy_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr command_publisher_;
  std::unique_ptr<FlightControllerAdapter> flight_adapter_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_subscriber_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_subscriber_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr goal_subscriber_;
  rclcpp::TimerBase::SharedPtr policy_timer_;
  std::mutex data_mutex_;
  std::vector<float> depth_;
  nav_msgs::msg::Odometry odometry_;
  geometry_msgs::msg::PointStamped goal_;
  builtin_interfaces::msg::Time depth_stamp_;
  bool have_depth_{false};
  bool have_odometry_{false};
  bool have_goal_{false};
  double stale_timeout_seconds_{0.25};
  std::array<float, kActionSize> previous_action_{};
};

}  // namespace stereo_avoidance

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<stereo_avoidance::StereoAvoidanceNode>());
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("stereo_avoidance_node"), "Startup failed: %s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
