


#!/usr/bin/env python3
"""
Standalone script for hosting classmate models using vLLM API server.
This script provides a dedicated service for running classmate models independently.
Includes batching support for handling multiple concurrent requests efficiently.
"""

import argparse
import json
import os
import subprocess
import time
import requests
import socket
import signal
import sys
import threading
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from multiprocessing import Queue, Process, Manager
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from queue import Queue as ThreadQueue, Empty


class ClassmateModelHost:
    """Host for a single classmate model using vLLM API server."""

    def __init__(
        self,
        model_name_or_path: str,
        model_index: int = 0,
        api_host: str = "127.0.0.1",
        api_port: int = 8000,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        use_v0: bool = False,
    ):
        """Initialize classmate model host.
        
        Args:
            model_name_or_path: Path or name of the model to host
            model_index: Index identifier for this model instance
            api_host: Host address for the API server
            api_port: Port for the API server
            gpu_memory_utilization: GPU memory utilization ratio
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            served_model_name: Name to use when serving the model (defaults to model_name_or_path)
            trust_remote_code: Whether to trust remote code
            dtype: Data type for model weights
        """
        self.model_name_or_path = model_name_or_path
        self.model_index = model_index
        self.api_host = api_host
        self.api_port = api_port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.served_model_name = model_name_or_path
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.use_v0 = use_v0

        self.api_base_url = None
        self.vllm_server_process = None

    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available for binding."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(('localhost', port))
                return True
            except OSError:
                return False

    def _find_available_port(self, start_port: int) -> int:
        """Find an available port starting from start_port."""
        port = start_port
        while not self._is_port_available(port):
            port += 1
            if port > start_port + 100:  # Avoid infinite loop
                raise RuntimeError(f"Could not find available port starting from {start_port}")
        return port

    def _wait_for_server_ready(self, max_wait_time: int = 300):
        """Wait for vLLM server to be ready to accept requests."""
        start_time = time.time()
        print(f"Waiting for vLLM server to be ready...")

        while time.time() - start_time < max_wait_time:
            try:
                # Check if server is ready by making a simple request
                response = requests.get(f"{self.api_base_url}/v{'0' if self.use_v0 else '1'}/models", timeout=5)
                if response.status_code == 200:
                    print(f"✅ vLLM server is ready!")
                    return
            except (requests.ConnectionError, requests.Timeout):
                pass

            # Check if process is still running
            if self.vllm_server_process and self.vllm_server_process.poll() is not None:
                stdout, stderr = self.vllm_server_process.communicate()
                raise RuntimeError(f"vLLM server process died. stdout: {stdout.decode()}, stderr: {stderr.decode()}")

            print(".", end="", flush=True)
            time.sleep(5)

        raise RuntimeError(f"vLLM server did not start within {max_wait_time} seconds")

    def start_server(self):
        """Start vLLM API server process for this classmate model."""
        if self.vllm_server_process is not None:
            print(f"⚠️  vLLM server is already running for model {self.model_index}")
            return

        # Find an available port
        self.api_port = self._find_available_port(self.api_port)
        self.api_base_url = f"http://{self.api_host}:{self.api_port}"

        # Prepare vLLM server command
        cmd = [
            "vllm", "serve", self.model_name_or_path,
            "--served-model-name", self.model_name_or_path,
            "--host", self.api_host,
            "--port", str(self.api_port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
        ]

        if self.trust_remote_code:
            cmd.append("--trust-remote-code")

        # print(f"🚀 Starting vLLM server for classmate model {self.model_index}")
        # print(f"📍 Model: {self.model_name_or_path}")
        # print(f"🌐 URL: {self.api_base_url}")
        # print(f"💻 Command: {' '.join(cmd)}")

        # Prepare environment with vLLM settings
        env = os.environ.copy()
        if self.use_v0:
            env["VLLM_USE_V1"] = "0"  # Use legacy API endpoints for compatibility
        
        # Start the vLLM server process
        self.vllm_server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )

        # Wait for server to be ready
        self._wait_for_server_ready()
        print(f"✅ vLLM server for classmate model {self.model_index} started successfully!")
        print(f"🔗 Access at: {self.api_base_url}")

    def stop_server(self):
        """Stop the vLLM API server process."""
        if self.vllm_server_process is None:
            return
            
        print(f"🛑 Stopping vLLM server for classmate model {self.model_index}")
        
        try:
            # First try graceful termination
            self.vllm_server_process.terminate()
            self.vllm_server_process.wait(timeout=15)
            print(f"✅ Model {self.model_index} server stopped gracefully")
        except subprocess.TimeoutExpired:
            print(f"⚠️  Force killing server for model {self.model_index}")
            self.vllm_server_process.kill()
            try:
                self.vllm_server_process.wait(timeout=5)
                print(f"✅ Model {self.model_index} server killed")
            except subprocess.TimeoutExpired:
                print(f"⚠️  Server for model {self.model_index} may still be running")
        except Exception as e:
            print(f"❌ Error stopping server for model {self.model_index}: {e}")

        self.vllm_server_process = None

    def get_server_info(self) -> Dict[str, Any]:
        """Get information about the running server."""
        return {
            "model_name_or_path": self.model_name_or_path,
            "served_model_name": self.model_name_or_path,
            "model_index": self.model_index,
            "api_base_url": self.api_base_url,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "is_running": self.vllm_server_process is not None and self.vllm_server_process.poll() is None,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
        }

class MultiClassmateModelHost:
    """Host multiple classmate models using separate vLLM API servers."""
    
    def __init__(
        self,
        model_paths: List[str],
        output_dir: str,
        outward_hostname: str,
        api_host: str = "127.0.0.1",
        start_port: int = 8000,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        use_v0: bool = False,
    ):
        """Initialize multi-model host.
        
        Args:
            model_paths: List of model paths or names to host
            output_dir: Directory to save the model mapping JSON file
            api_host: Host address for API servers
            start_port: Starting port number (each model gets consecutive ports)
            gpu_memory_utilization: GPU memory utilization ratio
            tensor_parallel_size: Number of GPUs to use for tensor parallelism per model
            trust_remote_code: Whether to trust remote code
            dtype: Data type for model weights
        """
        self.model_paths = model_paths
        self.output_dir = Path(output_dir)
        self.outward_hostname = outward_hostname
        self.api_host = api_host
        self.start_port = start_port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.use_v0 = use_v0

        self.hosts: Dict[str, ClassmateModelHost] = {}
        self.model_mapping: Dict[str, Dict[str, Any]] = {}

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\nReceived signal {signum}. Shutting down all servers gracefully...")
        self.stop_all_servers()
        sys.exit(0)

    # def get_outward_facing_address(self) -> str:
    #     import socket
    #     # get your outward-facing IP
    #     s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #     s.connect(("8.8.8.8", 80))
    #     ip = s.getsockname()[0]
    #     s.close()
    #
    #     return socket.gethostbyaddr(ip)[0]

    def start_all_servers(self):
        """Start vLLM servers for all models."""
        print(f"🚀 Starting {len(self.model_paths)} classmate models...")

        current_port = self.start_port

        # outward_host = self.get_outward_facing_address()

        for i, model_path in enumerate(self.model_paths):
            print(f"\n📦 Starting model {i+1}/{len(self.model_paths)}: {model_path}")
            print(f"   Path: {model_path}")

            # Create host for this model
            host = ClassmateModelHost(
                model_name_or_path=model_path,
                model_index=i,
                api_host=self.api_host,
                api_port=current_port,
                gpu_memory_utilization=self.gpu_memory_utilization,
                tensor_parallel_size=self.tensor_parallel_size,
                trust_remote_code=self.trust_remote_code,
                dtype=self.dtype,
                use_v0=self.use_v0
            )

            try:
                host.start_server()
                self.hosts[model_path] = host

                # Store mapping information
                self.model_mapping[model_path] = {
                    "outward_host": self.outward_hostname,
                    "local_host": self.api_host,
                    "port": host.api_port,
                    # "url": host.api_base_url,
                    "served_model_name": model_path,
                    "model_index": i,
                    "status": "running"
                }

                print(f"   ✅ Started successfully on port {host.api_port}")
                current_port = host.api_port + 1  # Use next available port for next model

            except KeyboardInterrupt:
                print(f"\n⚠️  Received Ctrl+C during startup. Stopping all servers...")
                self.stop_all_servers()
                raise
            except Exception as e:
                print(f"   ❌ Failed to start {model_path}: {e}")
                current_port += 1  # Still increment port to avoid conflicts

        # Save mapping to JSON file
        self.save_model_mapping()

        print(f"\n🎉 Successfully started {len(self.hosts)} out of {len(self.model_paths)} models")
        print(f"💡 Press Ctrl+C to stop all servers and exit")

    def save_model_mapping(self):
        """Save model mapping to JSON file."""
        mapping_file = self.output_dir / "classmate_model_mapping.json"

        with open(mapping_file, 'w') as f:
            json.dump(self.model_mapping, f, indent=2)

        print(f"💾 Model mapping saved to: {mapping_file}")

    def stop_all_servers(self):
        """Stop all running vLLM servers."""
        print("🛑 Stopping all servers...")

        for model_name, host in self.hosts.items():
            print(f"   Stopping {model_name}...")
            host.stop_server()

        # Update mapping to reflect stopped status
        for model_name in self.model_mapping:
            if self.model_mapping[model_name]["status"] == "running":
                self.model_mapping[model_name]["status"] = "stopped"

        self.save_model_mapping()
        print("✅ All servers stopped")

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about all hosted models."""
        return {
            "total_models": len(self.model_paths),
            "running_models": len(self.hosts),
            "model_mapping": self.model_mapping,
            "output_dir": str(self.output_dir),
            "start_port": self.start_port,
            "api_host": self.api_host
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Host multiple classmate models using vLLM API servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model specification
    parser.add_argument(
        "model_paths",
        nargs="+",
        help="Paths or names of models to host"
    )
    
    # Required arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save model mapping JSON file"
    )
    
    # Server configuration
    parser.add_argument(
        "--outward_hostname",
        type=str,
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1",
        help="Host address for all API servers (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--start-port", 
        type=int, 
        default=8000,
        help="Starting port number (models will use consecutive ports) (default: 8000)"
    )
    
    # vLLM configuration
    parser.add_argument(
        "--gpu-memory-utilization", 
        type=float, 
        default=0.9,
        help="GPU memory utilization ratio (default: 0.9)"
    )
    parser.add_argument(
        "--tensor-parallel-size", 
        type=int, 
        default=1,
        help="Number of GPUs to use for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights (default: auto)"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Whether to trust remote code (default: True)"
    )
    parser.add_argument(
        "--use_v0",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Get model paths from command line arguments
    model_paths = args.model_paths
    print(f"📦 Hosting {len(model_paths)} models:")
    for i, model_path in enumerate(model_paths, 1):
        print(f"   {i}. {model_path}")

    # Create multi-host instance
    multi_host = MultiClassmateModelHost(
        model_paths=model_paths,
        output_dir=args.output_dir,
        outward_hostname=args.outward_hostname,
        api_host=args.host,
        start_port=args.start_port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        use_v0=args.use_v0,
    )
    
    try:
        # Start all servers
        multi_host.start_all_servers()
        
        # Print comprehensive info
        info = multi_host.get_model_info()
        print(f"\n📊 Multi-Host Information:")
        print(f"   Total models: {info['total_models']}")
        print(f"   Successfully started: {info['running_models']}")
        print(f"   Host: local->{info['api_host']} | outward->{args.outward_hostname}")
        # print(f"   Start port: {info['start_port']}")
        print(f"   Output directory: {info['output_dir']}")

        # Keep the script running until interrupted
        print(f"\n🔄 All servers are running. Monitoring...")
        print(f"💡 Press Ctrl+C to stop all servers and exit gracefully")

        try:
            # Keep running indefinitely until Ctrl+C
            while True:
                time.sleep(1)
                # Optional: Check if any servers have died and restart them
                # This could be added as a feature later

        except KeyboardInterrupt:
            print(f"\n🛑 Received Ctrl+C. Initiating graceful shutdown...")

    except KeyboardInterrupt:
        print(f"\n🛑 Received Ctrl+C during startup. Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Always clean up servers before exiting
        multi_host.stop_all_servers()
        print(f"👋 All servers stopped. Goodbye!")


if __name__ == "__main__":
    main()