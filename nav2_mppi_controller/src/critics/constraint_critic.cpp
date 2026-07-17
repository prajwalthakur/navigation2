// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "nav2_mppi_controller/critics/constraint_critic.hpp"
#include "nav2_mppi_controller/motion_models.hpp"

namespace mppi::critics
{

void ConstraintCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  auto getParentParam = parameters_handler_->getParamGetter(parent_name_);

  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 4.0f);
  RCLCPP_INFO(
    logger_, "ConstraintCritic instantiated with %d power and %f weight.",
    power_, weight_);

  getParentParam(vx_max_, "vx_max", 0.5f);
  getParentParam(vx_min_, "vx_min", -0.35f);
  getParentParam(vy_max_, "vy_max", 0.0f);
  getParentParam(ax_max_, "ax_max", 0.5f);
  getParentParam(ax_min_, "ax_min", -0.5f);
  getParentParam(ay_max_, "ay_max", 0.5f);
  getParentParam(ay_min_, "ay_min", -0.5f);
  getParentParam(az_max_, "az_max", 0.5f);
  getParentParam(model_dt_,"model_dt",0.05f);
}

void ConstraintCritic::score(CriticData & data)
{
  if (!enabled_) {
    return;
  }

  float max_delta_vx =  model_dt_ * ax_max_;
  float min_delta_vx =  model_dt_ * ax_min_;
  float max_delta_vy =  model_dt_ * ay_max_;
  float min_delta_vy =  model_dt_ * ay_min_;
  float max_delta_wz =  model_dt_ * az_max_;

  unsigned int num_cols = data.state.vx.cols();
  unsigned int num_rows = data.state.vx.rows();

  // Differential motion model
  auto diff = dynamic_cast<DiffDriveMotionModel *>(data.motion_model.get());
  if (diff != nullptr) {

    auto & vx = data.state.vx;
    Eigen::ArrayXXf vx_shifted(num_rows, num_cols);
    vx_shifted.col(0).setConstant(static_cast<float>(data.state.speed.linear.x));
    vx_shifted.rightCols(num_cols - 1) = vx.leftCols(num_cols - 1);
    // acceleration : a[:, i] = (vx[:, i] - vx[:, i-1]) / dt, i = 1..N-1
    Eigen::ArrayXXf vx_diff = vx - vx_shifted;
    // cost on speed limits
    if (power_ > 1u) {
      data.costs += (((((data.state.vx - vx_max_).max(0.0f) + (vx_min_ - data.state.vx).
        max(0.0f)) * data.model_dt).rowwise().sum().eval()) * weight_).pow(power_).eval();
      
      // cost on acceleration limits
      data.costs += (((((vx_diff - max_delta_vx).max(0.0f) + (min_delta_vx - vx_diff).max(0.0f)) *
        data.model_dt).rowwise().sum().eval()) * weight_).pow(power_).eval();

    } else {
      data.costs += (((((data.state.vx - vx_max_).max(0.0f) + (vx_min_ - data.state.vx).
        max(0.0f)) * data.model_dt).rowwise().sum().eval()) * weight_).eval();

      // cost on acceleration limits
      data.costs += (((((vx_diff - max_delta_vx).max(0.0f) + (min_delta_vx - vx_diff).max(0.0f)) *
        data.model_dt).rowwise().sum().eval()) * weight_).eval();

    }
    return;
  }

  // Omnidirectional motion model
  // Axis wise violation check
  auto omni = dynamic_cast<OmniMotionModel *>(data.motion_model.get());
  if (omni != nullptr) {
    auto & vx = data.state.vx;
    auto & vy = data.state.vy;

    // for acceleration
    // accel_x
    Eigen::ArrayXXf vx_shifted(num_rows, num_cols);
    vx_shifted.col(0).setConstant(static_cast<float>(data.state.speed.linear.x));
    vx_shifted.rightCols(num_cols - 1) = vx.leftCols(num_cols - 1);
    // acceleration : a[:, i] = (vx[:, i] - vx[:, i-1]) / dt, i = 1..N-1
    Eigen::ArrayXXf vx_diff = vx - vx_shifted;

    // accel_y
    Eigen::ArrayXXf vy_shifted(num_rows, num_cols);
    vy_shifted.col(0).setConstant(static_cast<float>(data.state.speed.linear.y));
    vy_shifted.rightCols(num_cols - 1) = vy.leftCols(num_cols - 1);
    // acceleration : a[:, i] = (vx[:, i] - vx[:, i-1]) / dt, i = 1..N-1
    Eigen::ArrayXXf vy_diff = vy - vy_shifted;

    if(power_ > 1u) {
      data.costs += (((((vx - vx_max_).max(0.0f) + (vx_min_ - vx).max(0.0f) +
        (vy.abs() - vy_max_).max(0.0f)) * data.model_dt).rowwise().sum().eval()) *
        weight_).pow(power_).eval();

      data.costs += (((((vx_diff - max_delta_vx).max(0.0f) + (min_delta_vx - vx_diff).max(0.0f) +
        (vy_diff - max_delta_vy).max(0.0f) + (min_delta_vy - vy_diff).max(0.0f)
        ) * data.model_dt).rowwise().sum().eval()) * weight_).pow(power_).eval();


    } else {
      data.costs += (((((vx - vx_max_).max(0.0f) + (vx_min_ - vx).max(0.0f) +
        (vy.abs() - vy_max_).max(0.0f)) * data.model_dt).rowwise().sum().eval()) * weight_).eval();

      data.costs += (((((vx_diff - max_delta_vx).max(0.0f) + (min_delta_vx - vx_diff).max(0.0f) +

        (vy_diff - max_delta_vy).max(0.0f) + (min_delta_vy - vy_diff).max(0.0f)
        ) * data.model_dt).rowwise().sum().eval()) * weight_).eval();
    }
    return;
  }

  // Ackermann motion model
  auto acker = dynamic_cast<AckermannMotionModel *>(data.motion_model.get());
  if (acker != nullptr) {
    auto & vx = data.state.vx;
    auto & wz = data.state.wz;
    const float min_turning_rad = acker->getMinTurningRadius();

    const float epsilon = 1e-6f;
    auto wz_safe = wz.abs().max(epsilon);  // Replace small wz values to avoid division by 0
    auto out_of_turning_rad_motion = (min_turning_rad - (vx.abs() / wz_safe)).max(0.0f);

    Eigen::ArrayXXf vx_shifted(num_rows, num_cols);
    vx_shifted.col(0).setConstant(static_cast<float>(data.state.speed.linear.x));
    vx_shifted.rightCols(num_cols - 1) = vx.leftCols(num_cols - 1);
    // acceleration : a[:, i] = (vx[:, i] - vx[:, i-1]) / dt, i = 1..N-1
    Eigen::ArrayXXf vx_diff = vx - vx_shifted;

    if (power_ > 1u) {
      data.costs += ((((vx - vx_max_).max(0.0f) + (vx_min_ - vx).max(0.0f) +
        out_of_turning_rad_motion) * data.model_dt).rowwise().sum().eval() *
        weight_).pow(power_).eval();

      data.costs += (((((vx_diff - max_delta_vx).max(0.0f) + (min_delta_vx - vx_diff).max(0.0f)) *
        data.model_dt).rowwise().sum().eval()) * weight_).pow(power_).eval();

    } else {
      data.costs += ((((vx - vx_max_).max(0.0f) + (vx_min_ - vx).max(0.0f) +
        out_of_turning_rad_motion) * data.model_dt).rowwise().sum().eval() * weight_).eval();

      data.costs += (((((vx_diff - max_delta_vx).max(0.0f) + (min_delta_vx - vx_diff).max(0.0f)) *
        data.model_dt).rowwise().sum().eval()) * weight_).eval();

    }
    return;
  }
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(mppi::critics::ConstraintCritic, mppi::critics::CriticFunction)
