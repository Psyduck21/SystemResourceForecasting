#!/usr/bin/env python3
"""
Script to create artificial anomalies for testing the Neural System Monitor

This creates various types of stress patterns that should trigger anomaly detection:
- High CPU usage
- High memory usage
- Sudden spikes in resource usage

CAUTION: This script intentionally stresses the system. Use on test machines only.
"""

import time
import psutil
import subprocess
import sys
import os
import signal
from datetime import datetime


class AnomalyStressGenerator:
    """Generates different types of system stress for testing anomaly detection"""

    def __init__(self):
        self.processes = []

    def _start_subprocess(self, args, text=False):
        """Start subprocess safely and record it; return Popen or None."""
        try:
            # Use close_fds to avoid leaking fds; start_new_session so signals can be sent to the group
            p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 close_fds=True, start_new_session=True, text=text)
            self.processes.append(p)
            return p
        except Exception as e:
            print(f"   Failed to start process {args[:3]}...: {e}")
            return None

    def create_cpu_stress(self, duration_seconds=30, intensity=4):
        """Create high CPU usage by spawning stress processes"""
        print(f"🔥 Creating CPU stress for {duration_seconds} seconds (intensity: {intensity})")

        for i in range(intensity):
            # Try stress-ng first
            p = None
            try:
                p = self._start_subprocess([
                    'stress-ng', '--cpu', '1', '--cpu-method', 'fft',
                    '--timeout', str(duration_seconds)
                ])
                if p:
                    print(f"   Started stress-ng CPU process {i+1}")
                    continue
            except FileNotFoundError:
                pass

            # Fallback: small Python busy loop that runs for duration_seconds
            fallback_cmd = (
                "import time\n"
                "end = time.time() + %d\n"
                "while time.time() < end:\n"
                "    for _ in range(100000):\n"
                "        pass\n"
            ) % duration_seconds
            p = self._start_subprocess(['python3', '-u', '-c', fallback_cmd], text=True)
            if p:
                print(f"   Started Python CPU stress process {i+1}")

    def create_memory_stress(self, duration_seconds=30, memory_gb=2):
        """Create high memory usage (approximate)"""
        print(f"💾 Creating memory stress for {duration_seconds} seconds (~{memory_gb}GB)")

        p = None
        try:
            p = self._start_subprocess([
                'stress-ng', '--vm', '1', '--vm-bytes', f'{memory_gb}G',
                '--timeout', str(duration_seconds)
            ])
            if p:
                print("   Started stress-ng memory stress")
                return
        except FileNotFoundError:
            pass

        # Fallback memory hog in Python: allocate chunks until target or time reached.
        # Note: This is approximate and may raise MemoryError (caught in subprocess).
        chunk_count = max(1, int(memory_gb * 8))  # ~8 x 1MB chunks per 1GB approx (coarse)
        fallback_cmd = (
            "import time, sys\n"
            "chunks = []\n"
            "end = time.time() + %d\n"
            "try:\n"
            "    while time.time() < end:\n"
            "        # allocate ~8MB chunk (list of zeros)\n"
            "        chunks.append([0]*1000000)\n"
            "        time.sleep(0.1)\n"
            "except MemoryError:\n"
            "    pass\n"
            "finally:\n"
            "    time.sleep(0.5)\n"
        ) % duration_seconds
        p = self._start_subprocess(['python3', '-u', '-c', fallback_cmd], text=True)
        if p:
            print("   Started Python memory stress process")

    def create_sudden_spike(self, processes_to_create=3):
        """Create sudden spike by launching multiple busy-loop processes"""
        print(f"⚡ Creating sudden CPU spike with {processes_to_create} processes")

        spike_processes = []
        for i in range(processes_to_create):
            spike_cmd = (
                "import time\n"
                "end = time.time() + 50\n"
                "while time.time() < end:\n"
                "    s = 0\n"
                "    for x in range(10000):\n"
                "        s += x*x\n"
                "    time.sleep(1)\n"
            )
            p = self._start_subprocess(['python3', '-u', '-c', spike_cmd], text=True)
            if p:
                spike_processes.append(p)
                print(f"   Started spike process {i+1}")

        # Let them run for a short period, then terminate some of them to simulate a partial drop
        time.sleep(10)
        print("   Stopping half of spike processes...")
        # terminate all but one (if >1)
        for i, p in enumerate(spike_processes[:-1]):
            try:
                p.terminate()
                p.wait(timeout=2)
                print(f"   Terminated spike process {i+1}")
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

    def create_combined_stress(self, duration_seconds=45):
        """Create combined CPU + memory stress"""
        print(f"🚀 Creating combined CPU+memory stress for {duration_seconds} seconds")
        self.create_cpu_stress(duration_seconds, intensity=2)
        # Stagger the start a bit
        time.sleep(5)
        self.create_memory_stress(duration_seconds, memory_gb=1)

    def create_high_stress_general(self, duration_seconds=60):
        """Create extreme multi-system stress that triggers CRITICAL anomaly detection"""
        print(f"🚨 HIGH STRESS GENERAL MODE ACTIVATED - CRITICAL ANOMALIES EXPECTED")
        print(f"   This will create extreme system load for {duration_seconds} seconds")
        print(f"   Expected anomalies: CRITICAL system overload")

        # Stage 1: Massive CPU load waves
        for wave in range(3):
            print(f"   🚀 Stage 1: Wave {wave+1} launching CPU workers")
            cpu_processes = []
            for i in range(6):
                # Try stress-ng
                p = self._start_subprocess([
                    'stress-ng', '--cpu', '1', '--cpu-method', 'fft',
                    '--timeout', '20'
                ])
                if p:
                    cpu_processes.append(p)
                    continue

                # Fallback Python heavy calc limited to ~20s
                fallback_cmd = (
                    "import time, math\n"
                    "def intensive_calc():\n"
                    "    result = 0.0\n"
                    "    for i in range(200000):\n"
                    "        result += math.sin(i) * math.cos(i)\n"
                    "    return result\n"
                    "end = time.time() + 20\n"
                    "while time.time() < end:\n"
                    "    intensive_calc()\n"
                )
                p = self._start_subprocess(['python3', '-u', '-c', fallback_cmd], text=True)
                if p:
                    cpu_processes.append(p)

            time.sleep(8)
            # terminate half of them to simulate partial drop
            for p in cpu_processes[:3]:
                try:
                    p.terminate()
                    p.wait(timeout=1)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass

        # Stage 2: Massive memory allocation
        print("   💾 Stage 2: Extreme memory allocation")
        mem_procs = []
        for i in range(4):
            p = None
            try:
                p = self._start_subprocess([
                    'stress-ng', '--vm', '1', '--vm-bytes', '1G',
                    '--timeout', '25'
                ])
                if p:
                    mem_procs.append(p)
                    print(f"      Memory stress process {i+1} started (1G)")
                    continue
            except Exception:
                pass

            # fallback python memory hog
            fallback_cmd = (
                "import time\n"
                "chunks = []\n"
                "end = time.time() + 25\n"
                "try:\n"
                "    while time.time() < end:\n"
                "        chunks.append([0]*1000000)\n"
                "        time.sleep(0.2)\n"
                "except MemoryError:\n"
                "    pass\n"
            )
            p = self._start_subprocess(['python3', '-u', '-c', fallback_cmd], text=True)
            if p:
                mem_procs.append(p)
                print(f"      Python memory hog process {i+1} started")

        # Stage 3: I/O and process spawn stress if available
        print("   🔄 Stage 3: I/O and spawn stress (if available)")
        try:
            p_io = self._start_subprocess(['stress-ng', '--io', '8', '--timeout', '25'])
            if p_io:
                print("      I/O stress process started")
        except Exception:
            print("      I/O stress not started")

        try:
            p_fork = self._start_subprocess(['stress-ng', '--fork', '4', '--timeout', '25'])
            if p_fork:
                print("      Process spawn stress started")
        except Exception:
            print("      Process spawn stress not started")

        # Stage 4: Final CPU escalation
        print("   🚨 Stage 4: Final CPU escalation")
        for i in range(4):
            fallback_cmd = (
                "import time\n"
                "end = time.time() + 15\n"
                "while time.time() < end:\n"
                "    [x*x for x in range(10000)]\n"
            )
            p = self._start_subprocess(['python3', '-u', '-c', fallback_cmd], text=True)
            if p:
                print(f"      Additional CPU worker {i+1} started")

        print(f"   🔥 HIGH STRESS GENERAL: All systems under extreme load for {duration_seconds} seconds")
        time.sleep(duration_seconds)

        print("   🧹 Cleaning up high stress processes...")
        self.cleanup()

    def cleanup(self):
        """Clean up all running stress processes"""
        print("\n🧹 Cleaning up stress processes...")
        for p in list(self.processes):
            try:
                # try terminate group if possible
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception:
                    pass
                p.terminate()
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

            # wait briefly
            try:
                p.wait(timeout=2)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

        self.processes.clear()
        print("✅ Cleanup complete")

    def print_system_status(self):
        """Print current system resource usage"""
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        print("📊 Current system status:")
        print(f"   CPU Usage: {cpu:.1f}%")
        print(f"   Memory Usage: {memory.percent:.1f}%")
        print(f"   Processes: {len(psutil.pids())} active")


def print_usage():
    print("""
Usage:
    python3 anomaly_stress.py [command] [args...]

Commands:
    comprehensive                Run the full comprehensive test suite
    high-stress-general         Run an extreme stress sequence (asks for confirmation)
    cpu-stress [intensity] [duration]
                                Run CPU stress. intensity (default 4), duration seconds (default 30)
    memory-stress [memory_gb] [duration]
                                Run memory stress. memory_gb (default 2.0), duration seconds (default 30)
    combined-stress [duration]  Run combined CPU+Memory stress (default duration 45)
    sudden-spike [processes]    Create a sudden CPU spike (default 5)
    help                        Show this help
""")


def run_comprehensive_test():
    """Run a comprehensive anomaly detection test suite"""
    print("🧪 COMPREHENSIVE ANOMALY DETECTION TEST SUITE")
    print("=" * 50)

    stress_gen = AnomalyStressGenerator()

    try:
        print("⏳ Starting test suite - ensure Neural System Monitor is running!")
        print("💡 Keep monitoring the GUI for anomaly detections\n")

        # Test 1: Baseline (wait for normal data collection)
        print("📈 TEST 1: Baseline monitoring (20 seconds)")
        for i in range(4):
            stress_gen.print_system_status()
            print("   Waiting 5 seconds...")
            time.sleep(5)

        print("\n" + "=" * 50)
        # Test 2: CPU Stress
        print("🔥 TEST 2: CPU Stress Test")
        stress_gen.create_cpu_stress(duration_seconds=30, intensity=3)
        print("   CPU stress active - check GUI for anomalies!")
        time.sleep(30)
        stress_gen.cleanup()

        print("\n" + "=" * 50)
        # Test 3: Memory Stress
        stress_gen.print_system_status()
        print("💾 TEST 3: Memory Stress Test")
        stress_gen.create_memory_stress(duration_seconds=30)
        print("   Memory stress active - check GUI for anomalies!")
        time.sleep(30)
        stress_gen.cleanup()

        print("\n" + "=" * 50)
        # Test 4: Sudden Spike
        stress_gen.print_system_status()
        print("⚡ TEST 4: Sudden Spike Test")
        stress_gen.create_sudden_spike(5)
        print("   Sudden spike created - check GUI for anomalies!")
        time.sleep(20)
        stress_gen.cleanup()

        print("\n" + "=" * 50)
        # Test 5: Combined Stress
        stress_gen.print_system_status()
        print("🚀 TEST 5: Combined CPU+Memory Stress")
        stress_gen.create_combined_stress(45)
        print("   Combined stress active - check GUI for anomalies!")
        time.sleep(45)
        stress_gen.cleanup()

        print("\n" + "=" * 50)
        print("✅ Test suite complete!")
        print("📋 Summary:")
        print("   - CPU stress: High CPU processes should trigger 'Elevated CPU Activity'")
        print("   - Memory stress: High memory usage should trigger 'High Memory Consumption'")
        print("   - Sudden spikes: Rapid CPU jumps should trigger 'Sudden CPU Spike'")
        print("   - Combined: Multiple simultaneous anomalies possible")

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}")
    finally:
        stress_gen.cleanup()


def main():
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd in ("help", "-h", "--help"):
            print_usage()
            return

        if cmd == "comprehensive":
            run_comprehensive_test()
            return

        if cmd == "highstress":
            print("🚨 HIGH STRESS GENERAL MODE")
            print("🔴 This will create CRITICAL system stress for anomaly testing!")
            print("💡 Make sure Neural System Monitor GUI is running to see CRITICAL anomalies\n")

            confirm = input("⚠️  Are you sure you want to create extreme system stress? (y/N): ").strip().lower()
            if confirm in ('y', 'yes'):
                stress_gen = AnomalyStressGenerator()
                try:
                    stress_gen.print_system_status()
                    stress_gen.create_high_stress_general(duration_seconds=60)
                except KeyboardInterrupt:
                    print("\n🛑 High stress test interrupted by user")
                finally:
                    stress_gen.cleanup()
            else:
                print("❌ High stress test cancelled")
            return

        if cmd == "cpu-stress":
            intensity = int(sys.argv[2]) if len(sys.argv) > 2 else 4
            duration = int(sys.argv[3]) if len(sys.argv) > 3 else 30
            stress_gen = AnomalyStressGenerator()
            try:
                stress_gen.print_system_status()
                stress_gen.create_cpu_stress(duration_seconds=duration, intensity=intensity)
                print(f"   💡 CPU stress active for {duration} seconds - check GUI for anomalies!")
                time.sleep(duration)
            except KeyboardInterrupt:
                print("\n🛑 CPU stress test interrupted by user")
            finally:
                stress_gen.cleanup()
            return

        if cmd == "memory-stress":
            memory_gb = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
            duration = int(sys.argv[3]) if len(sys.argv) > 3 else 30
            stress_gen = AnomalyStressGenerator()
            try:
                stress_gen.print_system_status()
                stress_gen.create_memory_stress(duration_seconds=duration, memory_gb=memory_gb)
                print(f"   💡 Memory stress active for {duration} seconds - check GUI for anomalies!")
                time.sleep(duration)
            except KeyboardInterrupt:
                print("\n🛑 Memory stress test interrupted by user")
            finally:
                stress_gen.cleanup()
            return

        if cmd == "combined-stress":
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 45
            stress_gen = AnomalyStressGenerator()
            try:
                stress_gen.print_system_status()
                stress_gen.create_combined_stress(duration_seconds=duration)
                print(f"   💡 Combined stress active for {duration} seconds - check GUI for anomalies!")
                time.sleep(duration)
            except KeyboardInterrupt:
                print("\n🛑 Combined stress test interrupted by user")
            finally:
                stress_gen.cleanup()
            return

        if cmd == "sudden-spike":
            processes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            stress_gen = AnomalyStressGenerator()
            try:
                stress_gen.print_system_status()
                stress_gen.create_sudden_spike(processes_to_create=processes)
                print("   💡 Sudden spike created - check GUI for anomalies!")
                time.sleep(25)
            except KeyboardInterrupt:
                print("\n🛑 Sudden spike test interrupted by user")
            finally:
                stress_gen.cleanup()
            return

        print("Unknown command.")
        print_usage()
    else:
        # Simple stress test (default quick run)
        print("🔥 Quick CPU + Memory Stress Test")
        print("💡 Keep Neural System Monitor GUI open to see anomalies!\n")

        stress_gen = AnomalyStressGenerator()

        try:
            # 30 seconds of high CPU
            stress_gen.print_system_status()
            stress_gen.create_cpu_stress(duration_seconds=30, intensity=3)
            time.sleep(30)

            # 30 seconds of high memory
            stress_gen.print_system_status()
            stress_gen.create_memory_stress(duration_seconds=30, memory_gb=1.5)

            print("\n✅ Quick test complete! Check GUI for anomaly detections")

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        finally:
            stress_gen.cleanup()


if __name__ == "__main__":
    main()
