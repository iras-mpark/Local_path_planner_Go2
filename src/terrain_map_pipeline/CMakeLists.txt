cmake_minimum_required(VERSION 3.5)
project(terrain_map_pipeline)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
