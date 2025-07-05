import subprocess
import time
import os
import signal
import random

# CONFIGURATION
SERVER_IP = "127.0.0.1"
INTERFACE = "Ethernet 2"
DURATION = 10  # per mode test cycle
TOTAL_CYCLES = 3  # how many times to rotate through the 3 modes


def get_iperf_command(mode, client_id=None):
    if mode == 'heavy':
        return [r"C:\iperf\iperf3.exe", "-c", SERVER_IP, "-u", "-b", "100M", "-t", str(DURATION), "-p", "5201"]
    elif mode == 'normal':
        return [r"C:\iperf\iperf3.exe", "-c", SERVER_IP, "-u", "-b", "20M", "-t", str(DURATION), "-p", "5202"]
    elif mode == 'limited':
        return [r"C:\iperf\iperf3.exe", "-c", SERVER_IP, "-u", "-b", "5M", "-t", str(DURATION), "-l", "1200", "-p",
                "5203"]
    else:
        raise ValueError("Invalid mode")


def close_server(port):
    """Closes any running iperf3 server on the specified port"""
    print(f"[+] Checking for existing iperf3 server on port {port}...")
    # Find and kill the process using the given port
    try:
        result = subprocess.run(f"netstat -ano | findstr :{port}", shell=True, capture_output=True, text=True)
        if result.stdout:
            pid = result.stdout.split()[-1]
            print(f"[+] Killing iperf3 server with PID {pid}...")
            os.kill(int(pid), signal.SIGTERM)
            print(f"[+] Successfully killed the iperf3 server on port {port}.")
        else:
            print(f"[+] No iperf3 server found on port {port}.")
    except Exception as e:
        print(f"[!] Error killing iperf3 server: {e}")


def start_server(port):
    """Start an iperf3 server on the specified port"""
    print(f"[+] Starting iperf3 server on port {port}...")
    server_cmd = [r"C:\iperf\iperf3.exe", "-s", "-p", str(port)]
    return subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def add_delay(mode):
    """Add artificial delay and jitter based on the mode"""
    if mode == 'heavy':
        delay = 0.1  # 100ms for heavy mode
        jitter = random.uniform(0, 0.05)  # 0-50ms jitter
    elif mode == 'normal':
        delay = 0.05  # 50ms for normal mode
        jitter = random.uniform(0, 0.02)  # 0-20ms jitter
    elif mode == 'limited':
        delay = 0.02  # 20ms for limited mode
        jitter = random.uniform(0, 0.01)  # 0-10ms jitter
    else:
        delay = 0
        jitter = 0

    total_delay = delay + jitter
    print(f"[+] Added {total_delay:.3f}s delay and jitter ({jitter:.3f}s) for {mode} mode.")
    time.sleep(total_delay)  # Apply the delay + jitter


def run_test():
    total_test_duration = (DURATION + 2) * TOTAL_CYCLES * 3
    pcap_file = f"combined_capture.pcap"

    # Start packet capture with tshark
    tshark_cmd = [
        r"C:\Program Files\Wireshark\tshark.exe", "-i", INTERFACE,
        "-a", f"duration:{total_test_duration}", "-w", pcap_file
    ]
    print(f"[+] Starting Wireshark capture on {INTERFACE} for {total_test_duration} seconds...")
    tshark_proc = subprocess.Popen(tshark_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    time.sleep(5)  # Allow capture to initialize

    modes = ['heavy', 'normal', 'limited']
    # Close any existing servers before starting new ones
    close_server(5201)  # Close heavy mode server
    close_server(5202)  # Close normal mode server
    close_server(5203)  # Close limited mode server

    # Start the iperf3 servers for all modes
    server_procs = [
        start_server(5201),  # Heavy mode server
        start_server(5202),  # Normal mode server
        start_server(5203)  # Limited mode server
    ]

    # Ensure all servers are up and listening
    time.sleep(3)  # Wait for the servers to initialize

    for cycle in range(TOTAL_CYCLES):
        for mode in modes:
            print(f"[+] Starting {mode} mode cycle {cycle + 1}")
            iperf_procs = []

            # Add delay based on the mode
            add_delay(mode)

            if mode == 'heavy':
                for client_id in range(10):  # Multiple UDP clients for heavy
                    iperf_cmd = get_iperf_command(mode, client_id)
                    proc = subprocess.Popen(iperf_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    iperf_procs.append(proc)
            else:
                iperf_cmd = get_iperf_command(mode)
                proc = subprocess.Popen(iperf_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                iperf_procs.append(proc)

            for proc in iperf_procs:
                proc.wait()

            print(f"[✓] Finished {mode} mode cycle {cycle + 1}\n")
            time.sleep(2)  # Short pause before switching mode

    stdout, stderr = tshark_proc.communicate()
    print(f"tshark output: {stdout.decode()}")
    print(f"tshark error: {stderr.decode()}")
    print(f"[✓] All mode tests completed. Results saved to {pcap_file}")


if __name__ == "__main__":
    run_test()
