import csv
import time
from collections import deque
from datetime import datetime

import numpy as np
import serial
from serial.tools import list_ports


# ============================================================
# SETTINGS
# ============================================================

# Serial port of the ESP32.
# Windows  → "COM3", "COM4", etc.
# Linux    → "/dev/ttyUSB0" or "/dev/ttyACM0"
# macOS    → "/dev/cu.usbserial-XXXX"
SERIAL_PORT = "COM3"

# Must match Serial.begin() in csi_recorder_v4.ino (default 115200).
BAUD_RATE = 115200

# Number of consecutive CSI frames per ML sample.
WINDOW_SIZE = 5

# Recording duration for each labeled session.
RECORD_SECONDS = 300

# Each ML sample uses a separate group of frames.
# This avoids highly overlapping samples.
NON_OVERLAPPING_WINDOWS = True

# Median filter across time.
USE_MEDIAN_FILTER = False


# Discard a window if too many frames are missing in time.
MAX_FRAME_GAP_MS = 2000

# Show frame-rate diagnostics every N seconds.
DIAG_INTERVAL = 30

# ============================================================


FILENAME = (
    "raw_csi_gesture_data_"
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)


CLASS_LIST = ["still", "walk", "wave"]


FEATURE_NAMES = [
    "mean_signal",
    "std_signal",
    "range_signal",
    "energy_signal",
    "temporal_change",
    "temporal_change_max",

    "mean_subcarrier_mean",
    "mean_subcarrier_std",

    "std_subcarrier_mean",
    "std_subcarrier_max",
    "std_subcarrier_std",

    "range_subcarrier_mean",
    "range_subcarrier_max",

    "mean_change_mean",
    "mean_change_max",

    "max_change_mean",
    "max_change_max",

    "normalized_change_mean",
    "normalized_change_max",

    "active_ratio",
    "fraction_active",
    "mean_frame_corr",

    # New temporal waveform features.
    "motion_mean",
    "motion_std",
    "motion_range",
    "motion_peak",

    # Session and frame metadata.
    "session_id",
    "label",
]


def show_ports():
    ports = list(list_ports.comports())

    if not ports:
        print("No serial ports detected.")
        return

    print("Available serial ports:")

    for port in ports:
        print(f"  {port.device}: {port.description}")


def parse_esp32_line(line):
    """
    Expected ESP32 data line:

    timestamp,frame_number,sample_count,amp_0,amp_1,...,amp_n
    """

    if not line or line.startswith("#"):
        return None

    parts = line.split(",")

    if len(parts) < 4:
        return None

    try:
        timestamp = int(parts[0])
        frame_number = int(parts[1])
        sample_count = int(parts[2])

        amplitudes = np.asarray(
            [float(value) for value in parts[3:]],
            dtype=np.float32,
        )

    except ValueError:
        return None

    if sample_count != len(amplitudes):
        return None

    if sample_count <= 0:
        return None

    if not np.all(np.isfinite(amplitudes)):
        return None

    return timestamp, frame_number, amplitudes


def median_filter_window(window, kernel_size=3):
    """
    Apply a temporal median filter independently to every subcarrier.
    """

    arr = np.asarray(window, dtype=np.float32)

    if arr.ndim != 2:
        return arr

    if kernel_size < 3 or kernel_size % 2 == 0:
        return arr

    radius = kernel_size // 2
    filtered = arr.copy()

    for frame_index in range(arr.shape[0]):
        start = max(0, frame_index - radius)
        end = min(arr.shape[0], frame_index + radius + 1)

        filtered[frame_index] = np.median(
            arr[start:end],
            axis=0,
        )

    return filtered


def safe_zscore(values):
    values = np.asarray(values, dtype=np.float32)

    mean_value = np.mean(values)
    standard_deviation = np.std(values)

    if standard_deviation < 1e-8:
        return np.zeros_like(values)

    return (values - mean_value) / standard_deviation


def calculate_motion_signal(arr):
    """
    Create one temporal motion signal from the CSI window.

    arr shape:
        frames x subcarriers

    The mean absolute subcarrier change is used as a compact
    movement signal.
    """

    differences = np.diff(arr, axis=0)

    if differences.shape[0] == 0:
        return np.zeros(1, dtype=np.float32)

    motion_signal = np.mean(
        np.abs(differences),
        axis=1,
    )

    return motion_signal.astype(np.float32)


def extract_temporal_features(window):
    """
    Extract features from a window.

    window contains tuples:
        (timestamp, frame_number, amplitudes)
    """

    if len(window) < WINDOW_SIZE:
        return None

    counts = [
        len(item[2])
        for item in window
    ]

    if len(set(counts)) != 1:
        return None

    timestamps = np.asarray(
        [item[0] for item in window],
        dtype=np.int64,
    )



    arr = np.asarray(
        [item[2] for item in window],
        dtype=np.float32,
    )

    if arr.ndim != 2:
        return None

    if arr.shape[0] != WINDOW_SIZE:
        return None

    if not np.all(np.isfinite(arr)):
        return None

    # Reject windows containing a large time gap.
    timestamp_gaps = np.diff(timestamps)

    if len(timestamp_gaps) > 0:
        if np.max(timestamp_gaps) > MAX_FRAME_GAP_MS:
            return None

    if USE_MEDIAN_FILTER:
        arr = median_filter_window(
            arr,
            3,
        )

    mean_subcarrier = np.mean(arr, axis=0)
    std_subcarrier = np.std(arr, axis=0)
    max_subcarrier = np.max(arr, axis=0)
    min_subcarrier = np.min(arr, axis=0)
    range_subcarrier = (
        max_subcarrier - min_subcarrier
    )

    differences = np.diff(arr, axis=0)

    if differences.shape[0] > 0:
        absolute_differences = np.abs(differences)

        mean_change_subcarrier = np.mean(
            absolute_differences,
            axis=0,
        )

        max_change_subcarrier = np.max(
            absolute_differences,
            axis=0,
        )

    else:
        mean_change_subcarrier = np.zeros(arr.shape[1])
        max_change_subcarrier = np.zeros(arr.shape[1])

    normalized = np.asarray(
        [safe_zscore(frame) for frame in arr],
        dtype=np.float32,
    )

    normalized_differences = np.diff(
        normalized,
        axis=0,
    )

    activity = np.ptp(arr, axis=0)
    activity_threshold = np.mean(activity)

    correlations = []

    for index in range(len(arr) - 1):
        first_frame = arr[index]
        second_frame = arr[index + 1]

        if (
            np.std(first_frame) < 1e-8
            or np.std(second_frame) < 1e-8
        ):
            continue

        correlation = np.corrcoef(
            first_frame,
            second_frame,
        )[0, 1]

        if np.isfinite(correlation):
            correlations.append(correlation)

    motion_signal = calculate_motion_signal(arr)

    features = {
        "mean_signal": float(np.mean(arr)),
        "std_signal": float(np.std(arr)),
        "range_signal": float(np.ptp(arr)),
        "energy_signal": float(np.mean(arr ** 2)),

        "temporal_change": float(
            np.mean(np.abs(differences))
        ),

        "temporal_change_max": float(
            np.max(np.abs(differences))
        ),

        "mean_subcarrier_mean": float(
            np.mean(mean_subcarrier)
        ),

        "mean_subcarrier_std": float(
            np.std(mean_subcarrier)
        ),

        "std_subcarrier_mean": float(
            np.mean(std_subcarrier)
        ),

        "std_subcarrier_max": float(
            np.max(std_subcarrier)
        ),

        "std_subcarrier_std": float(
            np.std(std_subcarrier)
        ),

        "range_subcarrier_mean": float(
            np.mean(range_subcarrier)
        ),

        "range_subcarrier_max": float(
            np.max(range_subcarrier)
        ),

        "mean_change_mean": float(
            np.mean(mean_change_subcarrier)
        ),

        "mean_change_max": float(
            np.max(mean_change_subcarrier)
        ),

        "max_change_mean": float(
            np.mean(max_change_subcarrier)
        ),

        "max_change_max": float(
            np.max(max_change_subcarrier)
        ),

        "normalized_change_mean": float(
            np.mean(np.abs(normalized_differences))
        ),

        "normalized_change_max": float(
            np.max(np.abs(normalized_differences))
        ),

        "active_ratio": float(
            np.max(activity)
            / (np.mean(activity) + 1e-6)
        ),

        "fraction_active": float(
            np.sum(activity > activity_threshold)
            / len(activity)
        ),

        "mean_frame_corr": float(
            np.mean(correlations)
            if correlations
            else 1.0
        ),

        "motion_mean": float(
            np.mean(motion_signal)
        ),

        "motion_std": float(
            np.std(motion_signal)
        ),

        "motion_range": float(
            np.ptp(motion_signal)
        ),

        "motion_peak": float(
            np.max(motion_signal)
        ),
    }

    return features


def wait_for_valid_frame(
    ser,
    expected_count=None,
):
    """
    Read serial lines until a valid CSI frame is found.
    """

    while True:
        raw = ser.readline()

        if not raw:
            return None

        line = raw.decode(
            "utf-8",
            errors="ignore",
        ).strip()

        parsed = parse_esp32_line(line)

        if parsed is None:
            continue

        _, _, amplitudes = parsed

        if (
            expected_count is not None
            and len(amplitudes) != expected_count
        ):
            continue

        return parsed


def make_session_id(label):
    return (
        f"{label}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


def record_class(
    ser,
    label,
    feature_names,
    writer,
    csvfile,
):
    print()
    print(f"Class selected: [{label}]")
    print(
        "Keep the ESP32 and router in their fixed positions."
    )

    input("Press Enter when ready...")

    for number in range(3, 0, -1):
        print(f"Starting in {number}...")
        time.sleep(1)

    print("Recording now.")

    # Discard old frames from before the preparation period.
    ser.reset_input_buffer()

    session_id = make_session_id(label)

    start_time = time.time()

    frame_buffer = deque(
        maxlen=WINDOW_SIZE
    )

    expected_count = None

    saved = 0
    received = 0
    rejected_windows = 0

    diagnostic_start = time.time()
    diagnostic_frames = 0

    while time.time() - start_time < RECORD_SECONDS:
        result = wait_for_valid_frame(
            ser,
            expected_count,
        )

        if result is None:
            continue

        timestamp, frame_number, amplitudes = result

        if expected_count is None:
            expected_count = len(amplitudes)

        frame_buffer.append(
            (
                timestamp,
                frame_number,
                amplitudes,
            )
        )

        received += 1
        diagnostic_frames += 1

        now = time.time()

        if now - diagnostic_start >= DIAG_INTERVAL:
            fps = diagnostic_frames / (
                now - diagnostic_start
            )

            elapsed = now - start_time
            remaining = max(
                0,
                RECORD_SECONDS - elapsed,
            )

            print()
            print(
                f"  Frame rate: {fps:.2f} frames/s | "
                f"Frames: {received} | "
                f"Windows: {saved} | "
                f"Time left: {remaining:.0f}s"
            )

            if fps < 2.0:
                print(
                    "  Warning: CSI rate is approximately 1 FPS. "
                    "This recorder is configured for that rate."
                )

            diagnostic_start = now
            diagnostic_frames = 0

        if len(frame_buffer) < WINDOW_SIZE:
            continue

        features = extract_temporal_features(
            list(frame_buffer)
        )

        if features is None:
            rejected_windows += 1

            if NON_OVERLAPPING_WINDOWS:
                frame_buffer.clear()

            continue

        row = [
            features[name]
            for name in feature_names
            if name not in {
                "session_id",
                "label",
            }
        ]

        row.extend(
            [
                session_id,
                label,
            ]
        )

        writer.writerow(row)
        csvfile.flush()

        saved += 1

        elapsed = max(
            time.time() - start_time,
            1e-6,
        )

        current_fps = received / elapsed

        print(
            f"  [{RECORD_SECONDS - elapsed:4.0f}s left] "
            f"fps={current_fps:.2f} | "
            f"frames={received:4d} | "
            f"windows={saved:3d}",
            end="\r",
        )

        if NON_OVERLAPPING_WINDOWS:
            frame_buffer.clear()

    print()
    print(
        f"Finished [{label}] | "
        f"Session: {session_id} | "
        f"Frames: {received} | "
        f"Windows: {saved} | "
        f"Rejected: {rejected_windows}"
    )

    if saved < 5:
        print(
            "Warning: very few valid windows were saved. "
            "At the current frame rate, use a smaller window "
            "or record for a longer duration."
        )

    return saved


def count_classes(
    filename,
    class_list,
):
    counts = {
        class_name: 0
        for class_name in class_list
    }

    try:
        with open(
            filename,
            "r",
            newline="",
        ) as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                label = row.get("label", "")

                if label in counts:
                    counts[label] += 1

    except FileNotFoundError:
        pass

    return counts


def print_balance(counts):
    print()
    print("Samples collected so far:")

    for class_name, count in counts.items():
        bar = "█" * (count // 2)
        warning = "  low" if count < 10 else ""

        print(
            f"  {class_name:<8} "
            f"{count:>4}  "
            f"{bar}{warning}"
        )

    positive_counts = [
        value
        for value in counts.values()
        if value > 0
    ]

    if len(positive_counts) >= 2:
        ratio = max(positive_counts) / min(
            positive_counts
        )

        if ratio > 2.0:
            print(
                f"Warning: class imbalance "
                f"({ratio:.1f}x)."
            )


def main():
    show_ports()

    print(
        f"\nOpening {SERIAL_PORT} "
        f"at {BAUD_RATE} baud..."
    )

    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUD_RATE,
        timeout=1,
    )

    # Opening the port may reset the ESP32.
    time.sleep(2)

    print("Waiting for the first valid CSI frame...")

    first_frame = None

    while first_frame is None:
        raw = ser.readline()

        if not raw:
            continue

        line = raw.decode(
            "utf-8",
            errors="ignore",
        ).strip()

        first_frame = parse_esp32_line(line)

    _, _, first_amplitudes = first_frame

    print(
        "CSI stream detected: "
        f"{len(first_amplitudes)} amplitudes/frame"
    )

    ser.reset_input_buffer()

    csv_feature_names = [
        name
        for name in FEATURE_NAMES
        if name not in {
            "session_id",
            "label",
        }
    ]

    print()
    print("=" * 62)
    print("              Raw CSI Gesture Recorder")
    print("=" * 62)
    print(f"Window size       : {WINDOW_SIZE} frames")
    print(f"Record duration   : {RECORD_SECONDS} seconds")
    print(
        "Window overlap    : "
        f"{'No' if NON_OVERLAPPING_WINDOWS else 'Yes'}"
    )
    print(
        "Median filter     : "
        f"{'On' if USE_MEDIAN_FILTER else 'Off'}"
    )
    print(f"Output file       : {FILENAME}")
    print("1 -> still | 2 -> walk | 3 -> wave | q -> quit")
    print("=" * 62)

    try:
        with open(
            FILENAME,
            "w",
            newline="",
        ) as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(
                csv_feature_names
                + [
                    "session_id",
                    "label",
                ]
            )

            csvfile.flush()

            while True:
                counts = count_classes(
                    FILENAME,
                    CLASS_LIST,
                )

                print_balance(counts)

                command = input(
                    "\nEnter label (1/2/3/q): "
                ).strip().lower()

                if command == "q":
                    break

                if command == "1":
                    label = "still"
                elif command == "2":
                    label = "walk"
                elif command == "3":
                    label = "wave"
                else:
                    print("Invalid input.")
                    continue

                record_class(
                    ser,
                    label,
                    FEATURE_NAMES,
                    writer,
                    csvfile,
                )

    finally:
        ser.close()

    final_counts = count_classes(
        FILENAME,
        CLASS_LIST,
    )

    print()
    print(f"Saved: {FILENAME}")
    print_balance(final_counts)


if __name__ == "__main__":
    main()