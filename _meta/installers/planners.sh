#!/bin/bash -i

install(){
    echo 'Enabled optional planners'
}

uninstall(){
    #TODO
    echo not implemented
}

# === MAIN SCRIPT ===
help(){
    echo "Usage: planners.sh <install|uninstall>"
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
    ;;
    *)
        help
        exit 1
    ;;
esac