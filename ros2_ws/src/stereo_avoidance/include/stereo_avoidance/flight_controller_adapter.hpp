#pragma once

#include <memory>
#include <string>

#include "geometry_msgs/msg/twist_stamped.hpp"
#include "rclcpp/rclcpp.hpp"

namespace stereo_avoidance {

struct BodyVelocityCommand {
  double forward_mps{0.0};
  double left_mps{0.0};
  double up_mps{0.0};
  double yaw_rate_radps{0.0};
};

// Hardware-specific flight-control integrations implement this interface.
// Commands are body-frame FLU velocity/yaw-rate setpoints, never motor PWM.
class FlightControllerAdapter {
 public:
  virtual ~FlightControllerAdapter() = default;
  virtual void publish(const BodyVelocityCommand& command) = 0;
  virtual void hold() = 0;
};

// Generic ROS topic adapter. Remap/replace this class for PX4, ArduPilot or a
// proprietary flight stack; each target adapter owns frame conversion and
// the target's required offboard/arming/watchdog protocol.
class RosTopicFlightControllerAdapter final : public FlightControllerAdapter {
 public:
  explicit RosTopicFlightControllerAdapter(
      const rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr& publisher,
      const rclcpp::Clock::SharedPtr& clock,
      const std::string& frame_id);

  void publish(const BodyVelocityCommand& command) override;
  void hold() override;

 private:
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr publisher_;
  rclcpp::Clock::SharedPtr clock_;
  std::string frame_id_;
};

}  // namespace stereo_avoidance
