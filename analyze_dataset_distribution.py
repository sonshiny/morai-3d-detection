#!/usr/bin/env python3
"""
analyze_dataset_distribution.py
================================
dataset_root 아래 scen 폴더들의 labels_3d/*.csv를 읽어 데이터 분포를 집계한다.
(신규 파일, 기존 파일 무수정. csv 표준 라이브러리로 읽기)

출력:
  1) 클래스별 박스 개수 (vehicle/pedestrian)
  2) x(전방거리) 히스토그램: 0-10, 10-20, 20-30, 30-40, 40-50, 50-60m
  3) y(좌우) 히스토그램: -30~-20, -20~-10, -10~0, 0~10, 10~20, 20~30m
  4) 씬별 프레임 수 / 평균 박스 수
  5) x·y 히스토그램 PNG 저장 (matplotlib)

CSV 컬럼(morai_3d_live.py CSV_HEADER 기준): class_name, x, y, ... 를 그대로 사용.
"""

import os
import csv
import argparse
import collections

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

X_BINS = [0, 10, 20, 30, 40, 50, 60]
Y_BINS = [-30, -20, -10, 0, 10, 20, 30]

_HERE = os.path.dirname(os.path.abspath(__file__))


def bin_label(edges, i):
    return f"{edges[i]}~{edges[i + 1]}"


def bin_index(value, edges):
    """value가 속하는 [edges[i], edges[i+1]) 구간 인덱스. 범위 밖이면 None."""
    if value < edges[0] or value >= edges[-1]:
        return None
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return None


def find_scen_dirs(dataset_root, scenarios=None):
    if scenarios:
        dirs = []
        for name in scenarios:
            path = os.path.join(dataset_root, name)
            if not os.path.isdir(path):
                print(f"[WARN] 시나리오 폴더 없음, 건너뜀: {path}")
                continue
            dirs.append((name, path))
        return dirs

    dirs = []
    for name in sorted(os.listdir(dataset_root)):
        path = os.path.join(dataset_root, name)
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, "labels_3d")):
            dirs.append((name, path))
    return dirs


def analyze(dataset_root, scenarios=None):
    scen_dirs = find_scen_dirs(dataset_root, scenarios)
    if not scen_dirs:
        raise FileNotFoundError(f"[ERROR] {dataset_root} 아래에 labels_3d 폴더를 가진 시나리오가 없습니다.")

    class_counts = collections.Counter()
    x_hist = [0] * (len(X_BINS) - 1)
    y_hist = [0] * (len(Y_BINS) - 1)
    x_values = []
    y_values = []

    frames_per_scen = {}
    boxes_per_scen = {}

    for scen_name, scen_dir in scen_dirs:
        lbl_dir = os.path.join(scen_dir, "labels_3d")
        csv_files = sorted(f for f in os.listdir(lbl_dir) if f.endswith(".csv"))
        frames_per_scen[scen_name] = len(csv_files)
        n_boxes_this_scen = 0

        for fname in csv_files:
            csv_path = os.path.join(lbl_dir, fname)
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    class_name = row["class_name"]
                    x = float(row["x"])
                    y = float(row["y"])

                    class_counts[class_name] += 1
                    n_boxes_this_scen += 1
                    x_values.append(x)
                    y_values.append(y)

                    xi = bin_index(x, X_BINS)
                    if xi is not None:
                        x_hist[xi] += 1

                    yi = bin_index(y, Y_BINS)
                    if yi is not None:
                        y_hist[yi] += 1

        boxes_per_scen[scen_name] = n_boxes_this_scen

    return {
        "scen_dirs": scen_dirs,
        "class_counts": class_counts,
        "x_hist": x_hist,
        "y_hist": y_hist,
        "x_values": x_values,
        "y_values": y_values,
        "frames_per_scen": frames_per_scen,
        "boxes_per_scen": boxes_per_scen,
    }


def print_report(stats):
    class_counts = stats["class_counts"]
    x_hist = stats["x_hist"]
    y_hist = stats["y_hist"]
    frames_per_scen = stats["frames_per_scen"]
    boxes_per_scen = stats["boxes_per_scen"]

    total_boxes = sum(class_counts.values())
    total_frames = sum(frames_per_scen.values())

    print("=" * 66)
    print("  데이터셋 분포 분석")
    print("=" * 66)
    print(f"시나리오 수: {len(stats['scen_dirs'])}   전체 프레임: {total_frames}   전체 박스: {total_boxes}")

    print("\n[1] 클래스별 박스 개수")
    print("-" * 66)
    for cls_name, count in sorted(class_counts.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * count / total_boxes if total_boxes else 0.0
        print(f"  {cls_name:15s}: {count:8d}  ({pct:5.1f}%)")

    print("\n[2] x(전방거리) 히스토그램 (m)")
    print("-" * 66)
    for i, count in enumerate(x_hist):
        pct = 100.0 * count / total_boxes if total_boxes else 0.0
        bar = "#" * int(pct / 2)
        print(f"  {bin_label(X_BINS, i):>10s}: {count:8d}  ({pct:5.1f}%) {bar}")

    print("\n[3] y(좌우) 히스토그램 (m)")
    print("-" * 66)
    for i, count in enumerate(y_hist):
        pct = 100.0 * count / total_boxes if total_boxes else 0.0
        bar = "#" * int(pct / 2)
        print(f"  {bin_label(Y_BINS, i):>10s}: {count:8d}  ({pct:5.1f}%) {bar}")

    print("\n[4] 씬별 프레임 수 / 평균 박스 수")
    print("-" * 66)
    print(f"  {'scenario':12s} {'frames':>8s} {'boxes':>8s} {'avg_boxes/frame':>16s}")
    for scen_name, _ in stats["scen_dirs"]:
        n_frames = frames_per_scen[scen_name]
        n_boxes = boxes_per_scen[scen_name]
        avg = n_boxes / n_frames if n_frames else 0.0
        print(f"  {scen_name:12s} {n_frames:8d} {n_boxes:8d} {avg:16.2f}")
    print("=" * 66)


def save_histograms(stats, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x_labels = [bin_label(X_BINS, i) for i in range(len(X_BINS) - 1)]
    y_labels = [bin_label(Y_BINS, i) for i in range(len(Y_BINS) - 1)]

    axes[0].bar(x_labels, stats["x_hist"], color="steelblue")
    axes[0].set_title("x (forward distance) distribution")
    axes[0].set_xlabel("x range (m)")
    axes[0].set_ylabel("box count")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(y_labels, stats["y_hist"], color="indianred")
    axes[1].set_title("y (lateral) distribution")
    axes[1].set_xlabel("y range (m)")
    axes[1].set_ylabel("box count")
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="MORAI 데이터셋 라벨 분포 분석")
    ap.add_argument("--dataset_root", default=os.path.join(_HERE, "dataset"),
                    help="scen 폴더들이 있는 상위 dataset 디렉토리")
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="분석할 시나리오 이름 목록 (예: scen01 scen02). 미지정 시 전체")
    ap.add_argument("--out", default=None,
                    help="히스토그램 PNG 저장 경로 (기본: verify_output/dataset_distribution.png)")
    args = ap.parse_args()

    if not os.path.isdir(args.dataset_root):
        print(f"[ERROR] dataset_root 없음: {args.dataset_root}")
        return

    stats = analyze(args.dataset_root, args.scenarios)
    print_report(stats)

    out_path = args.out
    if out_path is None:
        preferred = "/mnt/user-data/outputs"
        out_dir = preferred if os.path.isdir(preferred) else os.path.join(_HERE, "verify_output")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "dataset_distribution.png")

    save_histograms(stats, out_path)
    print(f"\n히스토그램 PNG 저장: {out_path}")


if __name__ == "__main__":
    main()
