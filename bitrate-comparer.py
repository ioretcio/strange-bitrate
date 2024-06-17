import subprocess
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import sys

def get_bitrate_per_second(video_path):
    command = [
        'ffprobe', '-show_entries', 'packet=pts_time,stream_index', '-select_streams', 'v', 
        '-show_entries', 'frame=pkt_size', '-of', 'csv=p=0', video_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    times = []
    sizes = []
    for line in lines:
        parts = line.split(',')
        if len(parts) == 2:
            if(  len(parts[1])>0):
                times.append(float(parts[1]))
            else:
                sizes.append(float(parts[0]))
        if(len(parts) ==1):
            sizes.append(float(parts[0]))
    bitrates = {}
    for time, size in zip(times, sizes):
        second = int(time)
        if second not in bitrates:
            bitrates[second] = 0
        bitrates[second] += size * 8

    return bitrates

def plot_bitrate(bitrates1, bitrates2):
    seconds1 = sorted(bitrates1.keys())
    bitrate_values1 = [bitrates1[second] / 1000 for second in seconds1]
    seconds2 = sorted(bitrates2.keys())
    bitrate_values2 = [bitrates2[second] / 1000 for second in seconds2]
    smoothed_bitrate_values1 = gaussian_filter1d(bitrate_values1, sigma=1)
    smoothed_bitrate_values2 = gaussian_filter1d(bitrate_values2, sigma=1)
    plt.figure(figsize=(12, 6))
    plt.plot(seconds1, smoothed_bitrate_values1, label='Bitrate Video original (kbps)')
    plt.plot(seconds2, smoothed_bitrate_values2, label='Bitrate Video processed (kbps)')
    plt.xlabel('Time (s)')
    plt.ylabel('Bitrate (kbps)')
    plt.title('Smoothed Bitrate per Second Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    if len(sys.argv) != 3:
        print("Usage: python bitrate-analyser.py <video_path1> <video_path2>")
        sys.exit(1)
    video_path1 = sys.argv[1]
    video_path2 = sys.argv[2]
    bitrates1 = get_bitrate_per_second(video_path1)
    bitrates2 = get_bitrate_per_second(video_path2)
    plot_bitrate(bitrates1, bitrates2)

if __name__ == "__main__":
    main()
