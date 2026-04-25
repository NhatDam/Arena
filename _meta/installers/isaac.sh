#!/bin/bash -i

name="isaac"

install(){
    set -e
    sudo apt install libfuse2
    
    if [ ! -d ~/isaacsim-4.2.0 ]; then
        wget --content-disposition "https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone%404.2.0-rc.18%2Brelease.16044.3b2ed111.gl.linux-x86_64.release.zip" -O isaac-sim.zip
        mkdir -p ~/isaacsim-4.2.0
        unzip isaac-sim.zip -d ~/isaacsim-4.2.0
        rm isaac-sim.zip
    fi
    
    cd "$ARENA_WS_DIR"
    
    until which nvidia-smi &> /dev/null; do
        echo "Warning: nvidia-smi command not found. Please install nvidia driver using"
        echo "sudo apt-get install nvidia-open"
        read -rp "Confirm installation by pressing [Enter]"
    done
    echo "Successfully detected NVIDIA driver installation"
    
    
    echo "nvidia-driver was installed"
    
    #Optional choice: install a CUDA-enabled PyTorch 2.4.0 build based on the CUDA version available on your system
    python -m pip install torch==2.4.0
    
    #Ensure upgrade the latest pip version
    python -m pip install --upgrade pip
    
    #Install typegaurd dependencies
    python -m pip install typeguard
    
    # zip install for now
    # #Install isaac sim
    # echo "Downloading Isaac sim ..."
    # python -m pip install isaacsim==4.2.0.2 --extra-index-url https://pypi.nvidia.com
    # #Install Isaac sim - python package
    # echo "Downloading Isaac sim - python package ..."
    # python -m pip install isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 --extra-index-url https://pypi.nvidia.com
    # python -m pip install isaacsim-kernel isaacsim-app isaacsim-asset isaacsim-benchmark isaacsim-code-editor isaacsim-core isaacsim-cortex isaacsim-example isaacsim-gui isaacsim-replicator isaacsim-rl isaacsim-robot isaacsim-robot-motion isaacsim-robot-setup isaacsim-ros1 isaacsim-ros2 isaacsim-sensor isaacsim-storage isaacsim-template isaacsim-test isaacsim-utils --extra-index-url https://pypi.nvidia.com
    
    touch ~/.ros/fastdds.xml
    echo '<?xml version="1.0" encoding="UTF-8" ?>

    <license>Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
    NVIDIA CORPORATION and its licensors retain all intellectual property
    and proprietary rights in and to this software, related documentation
    and any modifications thereto.  Any use, reproduction, disclosure or
    distribution of this software and related documentation without an express
    license agreement from NVIDIA CORPORATION is strictly prohibited.</license>


    <profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles" >
        <transport_descriptors>
            <transport_descriptor>
                <transport_id>UdpTransport</transport_id>
                <type>UDPv4</type>
            </transport_descriptor>
        </transport_descriptors>

        <participant profile_name="udp_transport_profile" is_default_profile="true">
            <rtps>
                <userTransports>
                    <transport_id>UdpTransport</transport_id>
                </userTransports>
                <useBuiltinTransports>false</useBuiltinTransports>
            </rtps>
        </participant>
    </profiles>' > ~/.ros/fastdds.xml
    
    #TODO redo this properly
    
    if [ "$(systemd-detect-virt)" = wsl ] ; then
        python -m pip install git+https://github.com/cpbotha/xdg-open-wsl.git
    fi
    # curl "https://install.launcher.omniverse.nvidia.com/installers/omniverse-launcher-linux.AppImage" > omniverse-launcher-linux.AppImage
    # chmod +x omniverse-launcher-linux.AppImage
    # ./omniverse-launcher-linux.AppImage --no-sandbox &
    
    SETUP_FILE=~/isaacsim-4.2.0/setup.sh
    # Write the content to the file
    cat << 'EOF' > "$SETUP_FILE"
#!/bin/sh
MY_DIR="$HOME/isaacsim-4.2.0"
export CARB_APP_PATH="$MY_DIR/kit"
export EXP_PATH="$MY_DIR/apps"
if [ -f "$MY_DIR/setup_python_env.sh" ] ; then
    . "$MY_DIR/setup_python_env.sh"
fi
export ISAAC_PATH="$MY_DIR"
EOF
    
    echo "Completed Isaac Sim download"
    echo 'yes' > "$ARENA_DIR/.venv/lib/python3.10/site-packages/omni/EULA_ACCEPTED" || echo 'failed to auto-accept eula'

    pushd "$ARENA_DIR" > /dev/null
        git submodule update --init arena_isaac
    popd > /dev/null
    
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
    pushd "$ARENA_DIR" > /dev/null
        git submodule deinit -f arena_isaac
    popd > /dev/null
    sed -i "/$name/d" "$INSTALLED"
}

source_fn(){
    if [ -z ${ISAAC_PATH+x} ] ; then
        . "$HOME/isaacsim-4.2.0/setup.sh"
    fi
    if [[ ! "$ARENA_MODELS_FORMATS" =~ usd ]]; then
        export ARENA_MODELS_FORMATS="${ARENA_MODELS_FORMATS},usdz,usd,usda,usdc"
    fi
}

# === MAIN SCRIPT ===
help(){
    echo "Usage: isaac.sh <install|uninstall>"
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
