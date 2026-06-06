#! /usr/bin/env python3
import asyncio
import os
import traceback
import rclpy
import rclpy.executors
from .node import TaskGenerator


def spin_blocking(executor):
    try:
        executor.spin()
    except rclpy.executors.ExternalShutdownException:
        pass


async def app_logic(node):
    node.get_logger().info('Beginning client, shut down with CTRL-C')
    await node.setup()
    stop_event = asyncio.Event()
    await stop_event.wait()


async def main_async(args=None):
    del args
    rclpy.init()
    loop = asyncio.get_running_loop()

    executor = rclpy.executors.MultiThreadedExecutor()

    node = TaskGenerator()
    node.event_loop = loop

    executor.add_node(node)

    spin_future = loop.run_in_executor(None, spin_blocking, executor)
    app_task = asyncio.create_task(app_logic(node))

    async def wait_for_first_completion():
        return await asyncio.wait(
            [spin_future, app_task],
            return_when=asyncio.FIRST_COMPLETED
        )

    try:
        disable_monitor = os.getenv('ARENA_DISABLE_AIOMONITOR', '').strip().lower() in ('1', 'true', 'yes', 'on')
        if disable_monitor:
            node.get_logger().info('aiomonitor disabled by ARENA_DISABLE_AIOMONITOR')
            done, _ = await wait_for_first_completion()
        else:
            import aiomonitor
            with aiomonitor.start_monitor(loop=loop, locals=locals()):
                done, _ = await wait_for_first_completion()

        if spin_future in done:
            spin_future.result()

        if app_task in done:
            app_task.result()

    except asyncio.CancelledError:
        node.get_logger().info('Shutting down.')
    except Exception:
        node.get_logger().error(traceback.format_exc())
        raise
    finally:
        if not app_task.done():
            app_task.cancel()

        executor.shutdown()

        try:
            await spin_future
        except Exception:
            pass

        executor.remove_node(node)
        node.destroy_node()
        rclpy.try_shutdown()


def main(args=None):
    try:
        asyncio.run(main_async(args=args))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    import time
    time.sleep(5)
    main()
