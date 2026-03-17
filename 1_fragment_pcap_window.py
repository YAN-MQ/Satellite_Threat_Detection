#!/usr/bin/env python3
"""
Module 1b: Attack-window extraction + IP fragmentation.

This version is adapted to the local workspace and uses packet-level windows
that are consistent with the CIC-IDS2017 Friday afternoon attack schedule.

Windows used in the current pipeline:
  - benign: Monday working hours
  - portscan: Friday 12:30 PM - 3:40 PM local capture time
  - ddos: Friday 3:40 PM - 4:30 PM local capture time

Those Friday windows come from the CICIDS2017 extended attack schedule that
refines the coarse "Friday afternoon" description from the ICISSP 2018 paper.
All timestamps below are converted to Asia/Shanghai to match this host.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Iterable, List

from scapy.all import Ether, IP, PcapReader, fragment, wrpcap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/home/lithic/final/data"
OUTPUT_DIR = os.path.join(BASE_DIR, "fragments_window")
MTU = 1500


@dataclass(frozen=True)
class WindowSpec:
    name: str
    source_file: str
    start_local: str
    end_local: str


WINDOWS = [
    WindowSpec("benign", "Monday-WorkingHours.pcap", "2017-07-03 20:00:00", "2017-07-04 03:59:59"),
    WindowSpec("portscan", "Friday-WorkingHours.pcap", "2017-07-07 23:30:00", "2017-07-08 02:39:59"),
    WindowSpec("ddos", "Friday-WorkingHours.pcap", "2017-07-08 02:40:00", "2017-07-08 03:29:59"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CIC-IDS2017 attack windows and fragment jumbo packets")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory containing CIC-IDS2017 PCAPs")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory for extracted PCAPs")
    parser.add_argument("--max-packets", type=int, default=50000, help="Maximum packets to keep per class after windowing")
    return parser.parse_args()


def extract_time_window(src: str, start_local: str, end_local: str, dst: str) -> None:
    subprocess.run(
        ["editcap", "-A", start_local, "-B", end_local, src, dst],
        check=True,
    )


def fragment_packet(pkt, mtu: int = MTU) -> List:
    if IP not in pkt:
        return [pkt]

    ip = pkt[IP]
    ip_header_len = int(ip.ihl) * 4
    declared_ip_len = int(ip.len) if ip.len is not None else len(bytes(ip))
    raw_ip = bytes(ip)
    if declared_ip_len <= mtu:
        return [pkt]

    payload = raw_ip[ip_header_len:declared_ip_len]
    frag_base = IP(
        version=ip.version,
        ihl=ip.ihl,
        tos=ip.tos,
        id=ip.id,
        flags=0,
        frag=0,
        ttl=ip.ttl,
        proto=ip.proto,
        src=ip.src,
        dst=ip.dst,
        options=ip.options,
    ) / payload

    ether = None
    if Ether in pkt:
        ether = Ether(src=pkt[Ether].src, dst=pkt[Ether].dst, type=pkt[Ether].type)

    frag_payload_size = mtu - ip_header_len
    frags = fragment(frag_base, fragsize=frag_payload_size)
    return [ether / frag if ether is not None else frag for frag in frags]


def iter_fragmented_packets(window_pcap: str, max_packets: int) -> Iterable:
    kept = 0
    with PcapReader(window_pcap) as reader:
        for pkt in reader:
            if IP not in pkt:
                continue
            for frag in fragment_packet(pkt):
                yield frag
            kept += 1
            if kept >= max_packets:
                break


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Module 1b: Attack Window Extraction + IP Fragmentation")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max packets per class: {args.max_packets}")

    with tempfile.TemporaryDirectory(prefix="attack_windows_") as temp_dir:
        for spec in WINDOWS:
            src = os.path.join(args.data_dir, spec.source_file)
            if not os.path.exists(src):
                raise FileNotFoundError(f"Missing source PCAP: {src}")

            temp_pcap = os.path.join(temp_dir, f"{spec.name}_window.pcap")
            dst = os.path.join(args.output_dir, f"{spec.name}.pcap")

            print(f"\n[{spec.name}] {spec.source_file}")
            print(f"  Window: {spec.start_local} -> {spec.end_local} (Asia/Shanghai)")
            extract_time_window(src, spec.start_local, spec.end_local, temp_pcap)

            packets = list(iter_fragmented_packets(temp_pcap, args.max_packets))
            wrpcap(dst, packets)
            print(f"  Saved {len(packets)} packets to {dst}")

    print("\nDone.")


if __name__ == "__main__":
    main()
