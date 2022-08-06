cmake_minimum_required(VERSION 3.0.2)
project(point_cloud_classifier)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  message_generation
)

catkin_python_setup()

add_message_files(
    FILES
    PointsList.msg
    TF.msg
)

generate_messages(
    DEPENDENCIES
    std_msgs
    sensor_msgs
)

catkin_package(
    CATKIN_DEPENDS message_runtime rospy std_msgs
)

include_directories(
    ${catkin_INCLUDE_DIRS}
    src
)

install(PROGRAMS
        point_cloud_classifier_main.py
        config.json
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

install(DIRECTORY
        launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        )
