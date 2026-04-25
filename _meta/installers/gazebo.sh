#!/bin/bash -i

name="gazebo"

install(){
  set -e
  
  # Define Arena workspace directory
  cd "$ARENA_WS_DIR"

  # Install dependencies
  sudo apt-get update
  sudo apt-get install -y \
    python3-pip \
    lsb-release \
    gnupg \
    curl \
    libgps-dev

  # Add OSRF Gazebo packages repository
  sudo curl -sSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
    http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | \
    sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

  # Add ROS 2 repository
  sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > \
    /etc/apt/sources.list.d/ros2-latest.list'
  curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

  # Update package lists
  sudo apt-get update

  # Install Gazebo binaries and ROS-Gazebo bridge
  sudo apt-get install -y \
    "gz-${GAZEBO_VERSION}" \
    libsdformat14-dev


  mkdir -p "$ARENA_WS_DIR/src/gazebo"

  rosdep install -y \
    --from-paths src \
    --ignore-src \
    --rosdistro "$ARENA_ROS_DISTRO" \
    || echo 'rosdep failed to install all dependencies'

  echo "Gazebo ${GAZEBO_VERSION}, ros_gz, sdformat_urdf installed successfully!"


  export USD_PATH="$ARENA_WS_DIR/tools/OpenUSD/install"

  if [ ! -d tools/OpenUSD ]; then
    echo "Installing OpenUSD"
    mkdir -p tools
    pushd tools
      git clone --depth 1 -b v24.08 https://github.com/PixarAnimationStudios/OpenUSD.git
      sudo apt-get install -y libpyside2-dev python3-opengl cmake libglu1-mesa-dev freeglut3-dev mesa-common-dev
      cd OpenUSD
      python3 build_scripts/build_usd.py --build-variant release --no-tests --no-examples --no-imaging --onetbb --no-tutorials --no-docs --no-python "$USD_PATH"
    popd
  fi

  export PATH=$USD_PATH/bin:$PATH
  export LD_LIBRARY_PATH=$USD_PATH/lib:$LD_LIBRARY_PATH
  export CMAKE_PREFIX_PATH=$USD_PATH:$CMAKE_PREFIX_PATH

  if [ ! -d src/tools/gz-usd ]; then
    echo "Installing gz-usd"
    sudo apt-get install -y libgz-cmake4-dev libsdformat15-dev libgz-common6-dev
    mkdir -p src/tools
    pushd src/tools
      git clone -b main https://github.com/gazebosim/gz-usd
    popd
    BASE_PATHS=src/tools/gz-usd arena build
    echo "Successfully installed gz-usd"
  fi

  if ! grep -q "$name" "$INSTALLED" 2>/dev/null; then
    echo "$name" >> "$INSTALLED"
  fi
  arena update
  arena build
  
  echo "Installed $name successfully."
}

uninstall(){
  #TODO
  echo not implemented, but unsetting flag
  sed -i "/$name/d" "$INSTALLED"
}


source_fn(){
  if [[ ! "$ARENA_MODELS_FORMATS" =~ sdf ]]; then
    export ARENA_MODELS_FORMATS="${ARENA_MODELS_FORMATS},sdf"
  fi
}


# === MAIN SCRIPT ===
help(){
  echo "Usage: gazebo.sh <install|uninstall>"
}
if [ $# -ne 1 ]; then
  help
  exit 1
fi
case "$1" in
  install)
    install
    ;;
  uninstall)
    uninstall
    ;;
  source)
    source_fn
    ;;
  *)
    help
    exit 1
    ;;
esac