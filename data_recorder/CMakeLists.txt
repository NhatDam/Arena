cmake_minimum_required(VERSION 3.16)
project(data_recorder)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(ament_cmake_ros REQUIRED)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install scripts
install(PROGRAMS
  scripts/record_control.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install README
install(FILES
  README.md
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
