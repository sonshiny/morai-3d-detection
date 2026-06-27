#!/usr/bin/env python3
import argparse
import contextlib
import csv
import io
import os
import shutil
import tempfile
from datetime import datetime

import cv2


DEFAULT_DATASET_ROOT = "/data/dataset"
DEFAULT_REPORT = "bad_jpegs_report.csv"
DEFAULT_BACKUP_ROOT = "/data/dataset_jpeg_backup"


@contextlib.contextmanager
def capture_fd2():
    old_fd = os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        os.dup2(tmp.fileno(), 2)
        yield tmp
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)


def read_with_libjpeg_warnings(path):
    with capture_fd2() as err_file:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        err_file.flush()
        err_file.seek(0)
        stderr = err_file.read().decode("utf-8", errors="replace").strip()
    return img, stderr


def iter_jpegs(dataset_root):
    for root, _, files in os.walk(dataset_root):
        if "/images/" not in root.replace(os.sep, "/"):
            continue
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg")):
                yield os.path.join(root, name)


def scenario_from_path(path, dataset_root):
    rel = os.path.relpath(path, dataset_root)
    parts = rel.split(os.sep)
    return parts[0] if parts else ""


def camera_from_path(path):
    parts = path.split(os.sep)
    try:
        idx = parts.index("images")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return ""


def frame_stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def write_report(rows, report_path):
    os.makedirs(os.path.dirname(os.path.abspath(report_path)) or ".", exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "scenario",
                "camera",
                "stem",
                "status",
                "height",
                "width",
                "message",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def backup_path_for(path, dataset_root, backup_root):
    rel = os.path.relpath(path, dataset_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(backup_root, stamp, rel)


def repair_jpeg(path, img, dataset_root, backup_root, quality):
    backup_path = backup_path_for(path, dataset_root, backup_root)
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    shutil.copy2(path, backup_path)

    tmp_path = f"{path}.tmp_reencode.jpg"
    ok = cv2.imwrite(tmp_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"cv2.imwrite failed: {path}")
    os.replace(tmp_path, path)
    return backup_path


def load_report(report_path):
    with open(report_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def scan(args):
    rows = []
    total = 0
    bad = 0
    unreadable = 0
    for path in iter_jpegs(args.dataset_root):
        total += 1
        img, stderr = read_with_libjpeg_warnings(path)
        if img is None:
            unreadable += 1
            bad += 1
            rows.append({
                "path": path,
                "scenario": scenario_from_path(path, args.dataset_root),
                "camera": camera_from_path(path),
                "stem": frame_stem(path),
                "status": "unreadable",
                "height": "",
                "width": "",
                "message": stderr or "cv2.imread returned None",
            })
        elif stderr:
            bad += 1
            h, w = img.shape[:2]
            rows.append({
                "path": path,
                "scenario": scenario_from_path(path, args.dataset_root),
                "camera": camera_from_path(path),
                "stem": frame_stem(path),
                "status": "warning",
                "height": h,
                "width": w,
                "message": " | ".join(stderr.splitlines()),
            })

        if args.progress and total % args.progress == 0:
            print(f"[scan] checked={total:,} bad={bad:,} unreadable={unreadable:,}", flush=True)

    write_report(rows, args.report)
    print(f"[scan] checked={total:,} bad={bad:,} unreadable={unreadable:,}", flush=True)
    print(f"[scan] report={args.report}", flush=True)

    by_scenario = {}
    by_camera = {}
    by_message = {}
    for row in rows:
        by_scenario[row["scenario"]] = by_scenario.get(row["scenario"], 0) + 1
        by_camera[row["camera"]] = by_camera.get(row["camera"], 0) + 1
        key = row["message"].split("|")[0].strip()[:80]
        by_message[key] = by_message.get(key, 0) + 1

    print("[scan] by scenario:", flush=True)
    for key, val in sorted(by_scenario.items()):
        print(f"  {key}: {val}", flush=True)
    print("[scan] by camera:", flush=True)
    for key, val in sorted(by_camera.items()):
        print(f"  {key}: {val}", flush=True)
    print("[scan] by first warning:", flush=True)
    for key, val in sorted(by_message.items(), key=lambda kv: (-kv[1], kv[0]))[:20]:
        print(f"  {val:5d}  {key}", flush=True)


def repair(args):
    rows = load_report(args.report)
    repaired = 0
    skipped = 0
    failed = 0
    failures = []

    for idx, row in enumerate(rows, start=1):
        path = row["path"]
        if row["status"] != "warning":
            skipped += 1
            continue
        img, stderr = read_with_libjpeg_warnings(path)
        if img is None:
            skipped += 1
            continue
        try:
            backup = repair_jpeg(
                path,
                img,
                dataset_root=args.dataset_root,
                backup_root=args.backup_root,
                quality=args.quality,
            )
            repaired += 1
            if args.progress and repaired % args.progress == 0:
                print(f"[repair] repaired={repaired:,} latest={path} backup={backup}", flush=True)
        except Exception as exc:
            failed += 1
            failures.append((path, str(exc)))

    print(f"[repair] report rows={len(rows):,} repaired={repaired:,} skipped={skipped:,} failed={failed:,}", flush=True)
    print(f"[repair] backups under {args.backup_root}", flush=True)
    if failures:
        print("[repair] failures:", flush=True)
        for path, msg in failures[:20]:
            print(f"  {path}: {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Scan and optionally repair JPEGs that emit libjpeg warnings.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    scan_p = sub.add_parser("scan")
    scan_p.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    scan_p.add_argument("--report", default=DEFAULT_REPORT)
    scan_p.add_argument("--progress", type=int, default=5000)
    scan_p.set_defaults(func=scan)

    repair_p = sub.add_parser("repair")
    repair_p.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    repair_p.add_argument("--report", default=DEFAULT_REPORT)
    repair_p.add_argument("--backup-root", default=DEFAULT_BACKUP_ROOT)
    repair_p.add_argument("--quality", type=int, default=95)
    repair_p.add_argument("--progress", type=int, default=100)
    repair_p.set_defaults(func=repair)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
