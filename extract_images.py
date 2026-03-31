# extract_images.py
import os
import cv2
import rosbag
from cv_bridge import CvBridge

bridge = CvBridge()

bags = [
    'scenario1.bag', 'scenario1_1.bag', 'scenario1_1.2.bag', 'scenario1_1.3.bag',
    'scenario3.bag', 'scenario322.bag', 'scenario4.bag', 'scenario6.bag',
    'scenario7.bag', 'scenario7.bag_2.bag', 'scenario7.2_5.bag'
]

cam_topics = {
    '/morai/cam_front': 'cam_front',
    '/morai/cam_front_left': 'cam_front_left',
    '/morai/cam_front_right': 'cam_front_right',
    '/morai/cam_back': 'cam_back',
    '/morai/cam_back_left': 'cam_back_left',
    '/morai/cam_back_right': 'cam_back_right',
}

os.makedirs('dataset/images', exist_ok=True)

counters = {v: 0 for v in cam_topics.values()}

for bag_path in bags:
    print(f'처리 중: {bag_path}')
    bag = rosbag.Bag(bag_path)

    for topic, msg, t in bag.read_messages(topics=list(cam_topics.keys())):
        cam_name = cam_topics[topic]
        idx = counters[cam_name]
        fname = f'dataset/images/{cam_name}_{idx:05d}.jpg'

        try:
            # sensor_msgs/Image
            if hasattr(msg, 'encoding'):
                img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # sensor_msgs/CompressedImage
            elif hasattr(msg, 'format'):
                img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                print(f'[WARN] 알 수 없는 메시지 타입: {topic}, skip')
                continue

            ok = cv2.imwrite(fname, img)
            if not ok:
                print(f'[WARN] 저장 실패: {fname}')
                continue

            counters[cam_name] += 1

        except Exception as e:
            print(f'[ERROR] {bag_path} | {topic} | frame={idx} | {e}')

    bag.close()
    print(f'  완료: {counters}')

print('전체 완료!')
print(f'최종 카운터: {counters}')