#!/usr/bin/env python3
"""
mgeo_to_static_labels.py (2D Polyline 수정본 - PDF 원리 완벽 반영)
- 출력 텍스트 포맷: class_id pt1_x pt1_y pt2_x pt2_y ... pt20_x pt20_y (총 41개 값)
"""

import os
import json
import numpy as np
import rosbag
from scipy.interpolate import interp1d

# 설정
MGEO_DIR = './mgeo_data'
BAG_FILE = 'o.bag'     # 실제 bag 파일 이름으로 꼭 변경하세요!
DATASET_DIR = './dataset'
OUT_DIR = os.path.join(DATASET_DIR, 'labels_static')
os.makedirs(OUT_DIR, exist_ok=True)

MAX_RANGE = 50.0
POINTS_PER_LINE = 20

def resample_polyline_2d(points_2d, num_points=20):
    """(x,y) 2D 선을 20개의 점으로 일정하게 쪼갭니다."""
    points_2d = np.array(points_2d)
    if len(points_2d) < 2: return None
    
    distances = np.cumsum(np.sqrt(np.sum(np.diff(points_2d, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)
    
    if distances[-1] == 0: return None
    
    alpha = np.linspace(0, distances[-1], num_points)
    resampled = np.zeros((num_points, 2))
    for i in range(2): # x, y 2개만 보간
        interpolator = interp1d(distances, points_2d[:, i], kind='linear')
        resampled[:, i] = interpolator(alpha)
    return resampled

def load_mgeo_polylines(mgeo_dir):
    global_lines = []
    
    files_to_load = [
        ('lane_boundary_set.json', 0),
        ('crosswalk_set.json', 1)
    ]
    
    for file_name, class_id in files_to_load:
        path = os.path.join(mgeo_dir, file_name)
        if not os.path.exists(path):
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"❌ {file_name} 읽기 실패: {e}")
                continue

            # 데이터가 리스트가 아니면 리스트로 강제 변환
            items = data if isinstance(data, list) else data.get('features', [data])

            for item in items:
                points = None
                
                # 1. item이 사전인 경우: 'points' 키를 먼저 찾고 없으면 'geometry' 확인
                if isinstance(item, dict):
                    points = item.get('points')
                    if points is None:
                        points = item.get('geometry', {}).get('coordinates')
                
                # 2. item 자체가 리스트인 경우: 그 자체가 좌표 리스트라고 간주
                elif isinstance(item, list):
                    points = item

                # 좌표 데이터가 유효한지 검사 (리스트이고, 내부에 또 리스트가 있어야 함)
                if points and isinstance(points, list) and len(points) >= 2:
                    try:
                        # 첫 번째 요소가 숫자가 아니라면(즉, [x,y,z] 형태라면) 넘파이 변환
                        pts_np = np.array(points)
                        if pts_np.ndim >= 2:
                            pts_2d = pts_np[:, :2]
                            global_lines.append({'class': class_id, 'points': pts_2d})
                    except:
                        continue
                        
    return global_lines

def transform_to_ego_centric_2d(global_lines, ego_pos, ego_heading, max_range=50.0):
    ego_centric_lines = []
    cos_t = np.cos(-ego_heading)
    sin_t = np.sin(-ego_heading)
    rot_mat = np.array([[cos_t, -sin_t], [sin_t,  cos_t]])

    for line in global_lines:
        pts = line['points']
        shifted_x = pts[:, 0] - ego_pos[0]
        shifted_y = pts[:, 1] - ego_pos[1]
        
        if np.min(np.sqrt(shifted_x**2 + shifted_y**2)) > max_range:
            continue
            
        xy = np.stack([shifted_x, shifted_y], axis=1)
        rotated_xy = xy @ rot_mat.T
        
        resampled_pts = resample_polyline_2d(rotated_xy, POINTS_PER_LINE)
        if resampled_pts is not None:
            ego_centric_lines.append({'class': line['class'], 'points': resampled_pts})
            
    return ego_centric_lines

def main():
    print("🚀 정적 맵 라벨 생성 시작 (2D Polyline 적용)...")
    global_lines = load_mgeo_polylines(MGEO_DIR)
    
    with open(os.path.join(DATASET_DIR, 'frame_groups.json'), 'r') as f:
        groups = json.load(f)
        
    bag = rosbag.Bag(BAG_FILE)
    ego_msgs = []
    for topic, msg, t in bag.read_messages(topics=['/Ego_topic']):
        ego_msgs.append({'ts': t.to_sec(), 'pos': [msg.position.x, msg.position.y], 'heading': msg.heading})
    bag.close()
    
    for group in groups:
        best_ego = min(ego_msgs, key=lambda e: abs(e['ts'] - group['ts']))
        ego_lines = transform_to_ego_centric_2d(global_lines, best_ego['pos'], best_ego['heading'], MAX_RANGE)
        
        out_file = os.path.join(OUT_DIR, f"{group['label_stem']}.txt")
        with open(out_file, 'w') as f:
            for line in ego_lines:
                # x, y 2개만 기록
                pts_str = " ".join([f"{p[0]:.2f} {p[1]:.2f}" for p in line['points']])
                f.write(f"{line['class']} {pts_str}\n")
                
    print(f"✅ 완료! {OUT_DIR} 에 2D 라벨 저장 완료.")

if __name__ == '__main__':
    main()