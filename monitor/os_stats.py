import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import os
import argparse
import platform
import shutil
from tabulate import tabulate
import socket
import asyncio

# Alfer 07/2024
# This script will capture system metrics and generate a report over a given interval.
# The script will capture CPU utilization, IOPS, Bandwidth, Device Utilization, and IO latency.
# The script will generate a system report with the following information:
# - Hostname and IP
# - CPU Information
# - Memory Information
# - Disk Information
# - Block Device Information
# - NVMe Volumes Information
# - RAID Information
# - LVM RAID Information
#
# Usage: python3 OS_stats.py --duration <duration> --split_wal --results_dir <results_dir> --report_only

def check_and_install_iostat():
    """Check if iostat is available, and install it if missing."""
    if shutil.which("iostat") is None:
        print("iostat is not installed. Attempting to install...")
        os_name = platform.system().lower()

        try:
            if os_name == "linux":
                subprocess.run(["sudo", "apt-get", "update"], check=True)
                subprocess.run(["sudo", "apt-get", "install", "-y", "sysstat"], check=True)
            elif os_name == "darwin":  # macOS
                subprocess.run(["brew", "install", "sysstat"], check=True)
            elif os_name == "windows":
                print("iostat is not natively supported on Windows. Please install it manually.")
            else:
                print(f"Unsupported OS: {os_name}. Please install iostat manually.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install iostat: {e}")
    else:
        print("iostat is already installed.")

def generate_system_report(results_dir):
    # Get Hostname and IP
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)

    host_data = [["Hostname", hostname], ["IP Address", ip_address]]
    # Get CPU info
    n_sockets = int(
        subprocess.check_output("lscpu | grep -i Socket | awk -F ':' '/Socket/{print $2}'", shell=True).decode(
            'utf-8').strip())
    n_cores = psutil.cpu_count(logical=False)
    n_vcpu = psutil.cpu_count(logical=True)
    cpu_model = subprocess.check_output(
        "lscpu | grep -v BIOS | awk -F ':' '/Model name/ {print $2}'", shell=True).decode('utf-8').strip()

    cpu_data = [["Number of CPU Socket(s)", n_sockets], ["Number of cores", n_cores], [
        "Number of logical cores(vCPU)", n_vcpu], ["CPU Model", cpu_model]]

    # Get Memory info
    mem_info = psutil.virtual_memory()

    mem_data = [["Total memory", f"{mem_info.total / (1024.0 ** 3):.2f} GB"]]

    # Get lsblk info
    try:
        lsblk_info = subprocess.check_output(
            ['lsblk', '-o', 'NAME,SIZE,FSTYPE,TYPE,MOUNTPOINT']).decode('utf-8').split('\n')
        lsblk_info = tabulate(
            [line.split() for line in lsblk_info[1:] if line], headers=lsblk_info[0].split())
    except FileNotFoundError:
        lsblk_info = "lsblk command not found. Please install util-linux to retrieve block device information."

    # Get NVMe volumes info
    try:
        nvme_info_output = subprocess.check_output(
            ['nvme', 'list']).decode('utf-8')
    except FileNotFoundError:
        nvme_info_output = "nvme command not found. Please install nvme-cli to retrieve NVMe volumes information."

    # Get RAID info via MDADM
    if os.path.exists('/proc/mdstat'):
        raid_info = subprocess.check_output(
            "cat /proc/mdstat", shell=True).decode('utf-8')
    else:
        raid_info = "No RAID devices found. /proc/mdstat does not exist."

    # Get LVM RAID info
    try:
        lvm_raid_info = subprocess.check_output(
            ['lvs', '-o', '+devices,segtype']).decode('utf-8').split('\n')
        if len(lvm_raid_info) <= 1:  # If the output is empty
            lvm_raid_info = "No LVM RAID devices found."
        else:
            lvm_raid_info = tabulate([line.split() for line in lvm_raid_info[1:] if line],
                                     headers=lvm_raid_info[0].split())
    except FileNotFoundError:
        lvm_raid_info = "lvs command not found. Please install lvm2 to retrieve LVM RAID information."
    except subprocess.CalledProcessError:
        lvm_raid_info = "No LVM RAID devices found."

    # Get Disk info
    disk_info = psutil.disk_partitions()
    disk_data = []
    for partition in disk_info:
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
        except PermissionError:
            continue
        disk_data.append([partition.device, partition.mountpoint, partition.fstype,
                          f"{partition_usage.total / (1024.0 ** 3):.2f} GB",
                          f"{partition_usage.used / (1024.0 ** 3):.2f} GB",
                          f"{partition_usage.free / (1024.0 ** 3):.2f} GB"])

    with open(os.path.join(results_dir, 'system_report.txt'), 'w') as f:
        f.write("Host Information:\n")
        f.write(tabulate(host_data, headers=[
                "Info", "Value"], tablefmt="pretty"))
        f.write("\n\nCPU Information:\n")
        f.write(tabulate(cpu_data, headers=[
                "Info", "Value"], tablefmt="pretty"))
        f.write("\n\nMemory Information:\n")
        f.write(tabulate(mem_data, headers=[
                "Info", "Value"], tablefmt="pretty"))
        f.write("\n\nDisk Information:\n")
        f.write(tabulate(disk_data, headers=[
                "Device", "Mountpoint", "Type", "Total Size", "Used", "Free"], tablefmt="pretty"))
        f.write("\n\nBlock Device Information:\n\n")
        f.write(lsblk_info)
        f.write("\n\nNVMe Volumes Information:\n\n")
        f.write(nvme_info_output)
        f.write("\n\nRAID Information:\n\n")
        f.write(raid_info)
        f.write("\n\nLVM RAID Information:\n\n")
        f.write(lvm_raid_info)


def do_capture_metrics(devices_to_monitor, cpu_data, io_data):
    cpu_times = psutil.cpu_times_percent()
    cpu_data.append({
        'CPU Utilization': psutil.cpu_percent(),
        'IO Wait': cpu_times.iowait,
        'User': cpu_times.user,
        'System': cpu_times.system
    })
    iostat_output = subprocess.check_output(
        ['iostat', '-dxy', '1', '1']).decode('utf-8').split('\n')

    for line in iostat_output[4:-2]:
        fields = line.split()
        if len(fields) > 0:
            for device in devices_to_monitor:
                read_iops = float(fields[1]) if fields[1].replace(
                    '.', '', 1).isdigit() else None
                write_iops = float(fields[7]) if fields[7].replace(
                    '.', '', 1).isdigit() else None
                read_bandwidth = float(fields[2]) if fields[2].replace(
                    '.', '', 1).isdigit() else None
                write_bandwidth = float(fields[8]) if fields[8].replace(
                    '.', '', 1).isdigit() else None
                device_util = float(fields[22]) if fields[22].replace(
                    '.', '', 1).isdigit() else None
                r_await = float(fields[5]) if fields[5].replace(
                    '.', '', 1).isdigit() else None
                w_await = float(fields[11]) if fields[11].replace(
                    '.', '', 1).isdigit() else None

                # Convert KB/sec to MB/sec
                read_bandwidth = read_bandwidth / 1024
                write_bandwidth = write_bandwidth / 1024

                if device not in io_data:
                    io_data[device] = {'Read IOPS': [], 'Write IOPS': [], 'Read Bandwidth': [],
                                       'Write Bandwidth': [], 'Device Utilization': [], 'r_await': [],
                                       'w_await': []}
                io_data[device]['Read IOPS'].append(read_iops)
                io_data[device]['Write IOPS'].append(write_iops)
                io_data[device]['Read Bandwidth'].append(read_bandwidth)
                io_data[device]['Write Bandwidth'].append(write_bandwidth)
                io_data[device]['Device Utilization'].append(device_util)
                io_data[device]['r_await'].append(r_await)
                io_data[device]['w_await'].append(w_await)

    return cpu_data, io_data


def capture_metrics_by_duration(duration, devices_to_monitor):
    cpu_data = []
    io_data = {}

    start_time = time.time()
    while time.time() - start_time < duration:
        cpu_data, io_data = do_capture_metrics(
            devices_to_monitor, cpu_data, io_data)

    return cpu_data, io_data


def capture_metrics_by_event(finish: asyncio.Event, devices_to_monitor):
    cpu_data = []
    io_data = {}

    while not finish.is_set():
        cpu_data, io_data = do_capture_metrics(
            devices_to_monitor, cpu_data, io_data)

    return cpu_data, io_data


def plot_metrics(results_dir, cpu_data, io_data):
    image_paths = []  # List to store paths of generated images

    # Plot for CPU Utilization
    cpu_df = pd.DataFrame(cpu_data)
    cpu_df['%CpuAvg'] = cpu_df['CPU Utilization'].rolling(window=5).mean()
    cpu_df['%iowait'] = cpu_df['IO Wait'].rolling(window=5).mean()
    cpu_df['%user'] = cpu_df['User'].rolling(window=5).mean()
    cpu_df['%sys'] = cpu_df['System'].rolling(window=5).mean()

    cpu_plot = cpu_df[['%CpuAvg', '%iowait', '%user', '%sys']].plot(
        grid=True, title='CPU Utilization')
    cpu_fig = cpu_plot.get_figure()
    cpu_plot.set_xlabel('Time')
    cpu_plot.set_ylabel('CPU Utilization (%)')
    cpu_image_path = os.path.join(results_dir, 'cpu_utilization.png')
    plt.savefig(cpu_image_path)
    image_paths.append(cpu_image_path)
    cpu_df.to_csv(os.path.join(results_dir, 'cpu_utilization.csv'))
    plt.clf()

    # Plot for IOPS
    fig, ax = plt.subplots(figsize=(15, 10))  # Adjust the size here
    for device_name, device_data in io_data.items():
        io_df = pd.DataFrame(device_data)
        io_df['Read IOPS'] = io_df['Read IOPS'].rolling(window=5).mean()
        io_df['Write IOPS'] = io_df['Write IOPS'].rolling(window=5).mean()
        io_df['Read IOPS'].plot(
            ax=ax, grid=True, label=f'{device_name} Read IOPS')
        io_df['Write IOPS'].plot(
            ax=ax, grid=True, label=f'{device_name} Write IOPS')
        io_df[['Read IOPS', 'Write IOPS']].to_csv(
            # Save DataFrame to CSV
            os.path.join(results_dir, f'{device_name}_io_iops.csv'))
    ax.legend(title='Device', loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlabel('Time')
    ax.set_ylabel('IOPS')
    plt.tight_layout()
    iops_image_path = os.path.join(results_dir, 'io_iops.png')
    plt.savefig(iops_image_path)
    image_paths.append(iops_image_path)
    plt.clf()

    # Plot for Bandwidth
    fig, ax = plt.subplots(figsize=(15, 10))  # Adjust the size here
    for device_name, device_data in io_data.items():
        io_df = pd.DataFrame(device_data)
        io_df['Read Bandwidth'] = io_df['Read Bandwidth'].rolling(
            window=5).mean()
        io_df['Write Bandwidth'] = io_df['Write Bandwidth'].rolling(
            window=5).mean()
        io_df['Read Bandwidth'].plot(
            ax=ax, grid=True, label=f'{device_name} Read Bandwidth', linewidth=2, alpha=0.7)
        io_df['Write Bandwidth'].plot(
            ax=ax, grid=True, label=f'{device_name} Write Bandwidth', linewidth=2, alpha=0.7)
        io_df[['Read Bandwidth', 'Write Bandwidth']].to_csv(
            # Save DataFrame to CSV
            os.path.join(results_dir, f'{device_name}_io_bandwidth.csv'))
    ax.legend(title='Device', loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlabel('Time')
    ax.set_ylabel('Bandwidth (MB/sec)')
    plt.tight_layout()
    bandwidth_image_path = os.path.join(results_dir, 'io_bandwidth.png')
    plt.savefig(bandwidth_image_path)
    image_paths.append(bandwidth_image_path)
    plt.clf()

    # Plot for Device Utilization
    fig, ax = plt.subplots(figsize=(15, 10))  # Adjust the size here
    for device_name, device_data in io_data.items():
        io_df = pd.DataFrame(device_data)
        io_df['Device Utilization'] = io_df['Device Utilization'].rolling(
            window=5).mean()
        io_df['Device Utilization'].plot(
            ax=ax, grid=True, label=f'{device_name} Device Utilization')
        io_df[['Device Utilization']].to_csv(
            # Save DataFrame to CSV
            os.path.join(results_dir, f'{device_name}_device_utilization.csv'))
    ax.legend(title='Device', loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlabel('Time')
    ax.set_ylabel('Device Utilization (%)')
    plt.tight_layout()
    devutil_image_path = os.path.join(results_dir, 'io_devutil.png')
    plt.savefig(devutil_image_path)
    image_paths.append(devutil_image_path)
    plt.clf()

    # Plot for IO latency
    fig, ax = plt.subplots(figsize=(15, 10))
    for device_name, device_data in io_data.items():
        io_df = pd.DataFrame(device_data)
        io_df['r_await'] = io_df['r_await'].rolling(window=5).mean()
        io_df['w_await'] = io_df['w_await'].rolling(window=5).mean()
        io_df['r_await'].plot(
            ax=ax, grid=True, label=f'{device_name} Read Latency')
        io_df['w_await'].plot(
            ax=ax, grid=True, label=f'{device_name} Write Latency')
        io_df[['r_await', 'w_await']].to_csv(
            # Save DataFrame to CSV
            os.path.join(results_dir, f'{device_name}_io_latency.csv'))
    ax.legend(title='Device', loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlabel('Time')
    ax.set_ylabel('Latency (ms)')
    plt.tight_layout()
    latency_image_path = os.path.join(results_dir, 'io_latency.png')
    plt.savefig(latency_image_path)
    image_paths.append(latency_image_path)
    plt.clf()

    plt.close('all')  # Close all figures to free memory

    # Generate an HTML file to embed all images
    html_file_path = os.path.join(results_dir, 'metrics_summary.html')
    with open(html_file_path, 'w') as html_file:
        html_file.write(
            '<html><head><title>Metrics Summary</title></head><body>\n')
        html_file.write('<h1>Metrics Summary</h1>\n')
        for image_path in image_paths:
            html_file.write(f'<h2>{os.path.basename(image_path)}</h2>\n')
            html_file.write(
                f'<img src="{image_path}" alt="{os.path.basename(image_path)}" style="max-width:100%;"><br>\n')
        html_file.write('</body></html>\n')

    print(f"HTML summary generated at: {html_file_path}")


def monitor_and_generate_report(results_dir, devices_to_monitor, finish):
    cpu_data, io_data = capture_metrics_by_event(finish, devices_to_monitor)
    plot_metrics(results_dir, cpu_data, io_data)
    generate_system_report(results_dir)
    #print("System report generated successfully.")

    return


# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OS Statistics parser")
    parser.add_argument("--duration", help="Sampling duration",
                        dest="duration", default=60, type=int)
    parser.add_argument("--results_dir", help="Folder to store the results",
                        dest="results_dir", default=os.getcwd())
    parser.add_argument("--report_only", help="Create system report only",
                        dest="report_only", action="store_true")
    parser.add_argument("--devices", nargs='+',
                        help='Devices to be monitored', required=True)
    args = parser.parse_args()

    if args.report_only:
        generate_system_report(args.results_dir)
        print("System report generated successfully.")
        exit(0)

    if not args.results_dir:
        print("Results directory must be specified.")
        parser.print_help()
        exit(1)
    check_and_install_iostat()
    cpu_data, io_data = capture_metrics_by_duration(
        args.duration, args.devices)
    plot_metrics(args.results_dir, cpu_data, io_data)
    print("Statistics captured successfully.")
    generate_system_report(args.results_dir)