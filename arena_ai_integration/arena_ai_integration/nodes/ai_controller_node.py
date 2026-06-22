#!/usr/bin/env python3
"""Entry point: load agent strategy and run unified AI controller."""

from __future__ import annotations

import sys
import traceback

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from arena_ai_integration.agents import create_agent
from arena_ai_integration.core.base_ai_node import BaseAINode


class AgentTypeResolver(Node):
    """Minimal node used only to read agent_type before spinning the controller."""

    def __init__(self):
        super().__init__('ai_controller_type_resolver')
        self.declare_parameter('agent_type', 'socialnav')


def main(args=None):
    controller = None
    rclpy.init(args=args)
    resolver = None
    try:
        resolver = AgentTypeResolver()
        agent_type = str(resolver.get_parameter('agent_type').value).strip().lower()
        agent = create_agent(agent_type)
        controller = BaseAINode(agent)
        resolver.destroy_node()
        resolver = None
        rclpy.spin(controller)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        print('Fatal error in ai_controller_node:', file=sys.stderr)
        traceback.print_exc()
        raise
    finally:
        if resolver is not None:
            resolver.destroy_node()
        if controller is not None:
            controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
