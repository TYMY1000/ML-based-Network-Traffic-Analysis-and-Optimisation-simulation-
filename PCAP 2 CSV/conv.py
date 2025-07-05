import pyshark
import csv
import time


def extract_pcap_to_csv(pcap_file, output_csv):
    print(f"[+] Opening PCAP file {pcap_file}...")
    capture = pyshark.FileCapture(pcap_file, display_filter="ip")

    print(f"[+] Writing data to {output_csv}...")
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # CSV header
        writer.writerow([
            "Timestamp", "Source IP", "Destination IP", "Protocol",
            "Frame Length", "TCP Port", "UDP Port", "Jitter", "Target"
        ])

        previous_time = None

        for packet in capture:
            try:
                # Timestamp
                timestamp = packet.sniff_time.strftime('%Y-%m-%d %H:%M:%S.%f')
                packet_time = packet.sniff_time.timestamp()  # float seconds since epoch

                # IP and protocol
                src_ip = packet.ip.src
                dst_ip = packet.ip.dst
                protocol = packet.transport_layer

                # Frame length
                frame_length = int(packet.length)

                # Ports
                tcp_port = packet.tcp.port if hasattr(packet, 'tcp') else ""
                udp_port = packet.udp.port if hasattr(packet, 'udp') else ""

                # Continuous float jitter in milliseconds
                jitter = 0.0
                if previous_time is not None:
                    jitter = (packet_time - previous_time) * 1000.0  # ms
                previous_time = packet_time

                # Target (congestion label)
                if frame_length < 100:
                    target = 2  # High congestion
                elif 100 <= frame_length < 500:
                    target = 1  # Moderate congestion
                else:
                    target = 0  # No congestion

                # Write row
                writer.writerow([
                    timestamp, src_ip, dst_ip, protocol,
                    frame_length, tcp_port, udp_port, round(jitter, 3), target
                ])

            except AttributeError:
                continue  # Skip packets missing required fields

    print(f"[✓] Data extraction complete! Results saved to {output_csv}")


if __name__ == "__main__":
    pcap_file = "combined_capture.pcap"
    output_csv = "packet_data.csv"
    extract_pcap_to_csv(pcap_file, output_csv)
