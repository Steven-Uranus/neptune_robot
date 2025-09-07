#include "neptune_robot/NeptuneRobot.h"

namespace legged {
bool NeptuneRobot::parserObservation(const std::string& name) {
  std::cerr << "NeptuneRobot::parserObservation got called with name: " << name << std::endl;
  if (OnnxController::parserObservation(name)) {
    return true;
  }
  if (name == "my_observation") {
    // observationManager_->addTerm(std::make_shared<MyObservationTerm>(leggedModel));
  } else {
    return false;
  }
  return true;
}

}  // namespace legged

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(legged::NeptuneRobot, controller_interface::ControllerInterface)
