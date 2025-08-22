"""MCP Server runner with memory leak mitigation."""

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time

import psutil

logger = logging.getLogger(__name__)


class MCPServerRunner:
    """Manages MCP server lifecycle with automatic restart for memory leak mitigation."""

    def __init__(self, max_calls: int = 1000, max_memory_mb: int = 1024):
        """Initialize the MCP server runner.

        Args:
            max_calls: Maximum number of tool calls before restart
            max_memory_mb: Maximum memory usage in MB before restart
        """
        self.max_calls = max_calls
        self.max_memory_mb = max_memory_mb
        self.call_count = 0
        self.server_process: subprocess.Popen | None = None
        self.start_time = time.time()
        self.monitor_task: asyncio.Task | None = None
        self.should_stop = False

    async def start_server(self) -> subprocess.Popen:
        """Start the MCP server process."""
        logger.info("Starting MCP SDK server process...")

        # Prepare the command to run the server
        cmd = [sys.executable, "-m", "docsrs_mcp.mcp_sdk_server"]

        # Start the server process
        self.server_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),
        )

        self.start_time = time.time()
        self.call_count = 0

        logger.info(f"MCP SDK server started with PID {self.server_process.pid}")
        return self.server_process

    async def restart_server(self):
        """Restart the MCP server process."""
        logger.info(f"Restarting MCP server after {self.call_count} calls...")

        # Stop the current server
        if self.server_process:
            await self.stop_server()

        # Start a new server
        await self.start_server()

    async def stop_server(self):
        """Stop the MCP server process gracefully."""
        if not self.server_process:
            return

        logger.info(f"Stopping MCP server with PID {self.server_process.pid}...")

        # Try graceful shutdown first
        try:
            self.server_process.terminate()
            await asyncio.sleep(2)  # Give it time to shut down

            if self.server_process.poll() is None:
                # Force kill if still running
                self.server_process.kill()
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error stopping server: {e}")

        self.server_process = None

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage of the server process in MB."""
        if not self.server_process:
            return 0.0

        try:
            process = psutil.Process(self.server_process.pid)
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0

    async def monitor_server(self):
        """Monitor server health and restart if needed."""
        while not self.should_stop:
            try:
                # Check if server is still running
                if self.server_process and self.server_process.poll() is not None:
                    logger.warning(
                        "MCP server process died unexpectedly, restarting..."
                    )
                    await self.restart_server()
                    continue

                # Check memory usage
                memory_mb = self.get_memory_usage_mb()
                if memory_mb > self.max_memory_mb:
                    logger.warning(
                        f"Memory usage ({memory_mb:.1f} MB) exceeded limit "
                        f"({self.max_memory_mb} MB), restarting server..."
                    )
                    await self.restart_server()
                    continue

                # Check call count (would need to be tracked via tool calls)
                if self.call_count >= self.max_calls:
                    logger.info(
                        f"Call count ({self.call_count}) reached limit "
                        f"({self.max_calls}), restarting server..."
                    )
                    await self.restart_server()
                    continue

                # Log periodic status
                uptime = time.time() - self.start_time
                if int(uptime) % 300 == 0:  # Every 5 minutes
                    logger.info(
                        f"MCP server status: PID={self.server_process.pid if self.server_process else 'N/A'}, "
                        f"Memory={memory_mb:.1f}MB, Calls={self.call_count}, "
                        f"Uptime={uptime:.0f}s"
                    )

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Error in server monitor: {e}")
                await asyncio.sleep(10)

    def increment_call_count(self):
        """Increment the tool call counter."""
        self.call_count += 1
        logger.debug(f"Tool call count: {self.call_count}/{self.max_calls}")

    async def run_with_rotation(self):
        """Run the MCP server with automatic rotation."""
        logger.info(
            f"Starting MCP server runner with rotation every {self.max_calls} calls "
            f"or {self.max_memory_mb} MB memory usage"
        )

        # Start the initial server
        await self.start_server()

        # Start the monitor task
        self.monitor_task = asyncio.create_task(self.monitor_server())

        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.should_stop = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # Keep the runner alive
            while not self.should_stop:
                await asyncio.sleep(1)
        finally:
            # Clean shutdown
            logger.info("Shutting down MCP server runner...")
            self.should_stop = True

            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass

            await self.stop_server()
            logger.info("MCP server runner shutdown complete")


async def main():
    """Main entry point for the MCP server runner."""
    # Configure from environment variables
    max_calls = int(os.getenv("MCP_MAX_CALLS", "1000"))
    max_memory_mb = int(os.getenv("MCP_MAX_MEMORY_MB", "1024"))

    # Create and run the server runner
    runner = MCPServerRunner(max_calls=max_calls, max_memory_mb=max_memory_mb)
    await runner.run_with_rotation()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the server runner
    asyncio.run(main())
