#!/usr/bin/env python3
"""
Helper script to control data recording.

Usage:
    python3 record_control.py start <run_id> [experiment_tag]
    python3 record_control.py stop
"""

import sys
import time

import rclpy
from hunav_msgs.srv import StartEvaluation
from std_srvs.srv import Empty


def call_start_recording(run_id: int, experiment_tag: str = "default", timeout: int = 10):
    """Start recording."""
    rclpy.init()
    node = rclpy.create_node("record_controller")

    client = node.create_client(StartEvaluation, "/data_recorder_start_recording")

    # Wait for service with timeout
    start_time = time.time()
    while not client.wait_for_service(timeout_sec=1.0):
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"✗ Error: Service /data_recorder_start_recording not available after {timeout}s")
            print("  Make sure data_recorder_node is running: ros2 run data_recorder data_recorder_node")
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(1)
        sys.stdout.write(f"\rWaiting for service ({int(elapsed)}s/{timeout}s)...")
        sys.stdout.flush()

    print("\n✓ Service found!")

    request = StartEvaluation.Request()
    request.run_id = run_id
    request.experiment_tag = experiment_tag

    # Make the service call (blocking)
    future = client.call_async(request)
    
    # Spin until the future completes
    while rclpy.ok() and not future.done():
        rclpy.spin_once(node, timeout_sec=0.1)

    try:
        response = future.result()
        if response.success:
            print(f"✓ Recording started for run_id={run_id}, experiment_tag={experiment_tag}")
        else:
            print("✗ Failed to start recording")
    except Exception as e:
        print(f"✗ Error calling service: {e}")

    node.destroy_node()
    rclpy.shutdown()


def call_stop_recording(timeout: int = 10):
    """Stop recording."""
    rclpy.init()
    node = rclpy.create_node("record_controller")

    client = node.create_client(Empty, "/data_recorder_stop_recording")

    # Wait for service with timeout
    start_time = time.time()
    while not client.wait_for_service(timeout_sec=1.0):
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"✗ Error: Service /data_recorder_stop_recording not available after {timeout}s")
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(1)
        sys.stdout.write(f"\rWaiting for service ({int(elapsed)}s/{timeout}s)...")
        sys.stdout.flush()

    print("\n✓ Service found!")

    request = Empty.Request()
    
    # Make the service call (blocking)
    future = client.call_async(request)
    
    # Spin until the future completes
    while rclpy.ok() and not future.done():
        rclpy.spin_once(node, timeout_sec=0.1)

    try:
        response = future.result()
        print("✓ Recording stopped")
    except Exception as e:
        print(f"✗ Error calling service: {e}")

    node.destroy_node()
    rclpy.shutdown()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 record_control.py start <run_id> [experiment_tag]")
        print("  python3 record_control.py stop")
        sys.exit(1)

    command = sys.argv[1]

    if command == "start":
        if len(sys.argv) < 3:
            print("Error: run_id required")
            print("Usage: python3 record_control.py start <run_id> [experiment_tag]")
            sys.exit(1)

        run_id = int(sys.argv[2])
        experiment_tag = sys.argv[3] if len(sys.argv) > 3 else "default"

        call_start_recording(run_id, experiment_tag)

    elif command == "stop":
        call_stop_recording()

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
