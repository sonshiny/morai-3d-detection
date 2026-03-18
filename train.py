import torch
import torch.nn.functional as F
import torch.optim as optim

from morai_dataset import MoraiDataset, morai_collate_fn
from resnet_fpn import ResNet50_FPN, Bottleneck
from anchor_generator import generate_anchors
from decoder import FFNDecoder
from torch.utils.data import DataLoader
from loss_calculator import CustomLoss

CAM_ORDER = ['cam_front', 'cam_front_left', 'cam_front_right',
             'cam_back',  'cam_back_left',  'cam_back_right']

class AutoNavModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet50_FPN(Bottleneck)
        self.decoder  = FFNDecoder()
        self.anchors  = generate_anchors()   # [900, 3]

    def forward(self, images, intrinsics, extrinsics):
        """
        images     : [1, 6, 3, 224, 224]
        intrinsics : [1, 6, 3, 3]
        extrinsics : [1, 6, 4, 4]
        """
        device = images.device
        self.anchors = self.anchors.to(device)
        N = self.anchors.shape[0]  # 900

        anchors_homo = torch.cat(
            [self.anchors, torch.ones(N, 1, device=device)], dim=-1
        )  # [900, 4]

        agg_features = torch.zeros(N, 256, device=device)
        valid_cams   = 0

        for cam_idx in range(6):
            cam_img = images[0, cam_idx]          # [3, 224, 224]
            if cam_img.abs().sum() < 1e-6:        # 빈 슬롯 스킵
                continue

            # 1. 특징맵 추출
            features   = self.backbone(cam_img.unsqueeze(0))  # [1,256,56,56]
            p2_feature = features[0]

            # 2. 앵커 → 이미지 투영
            E = extrinsics[0, cam_idx]   # [4, 4]
            K = intrinsics[0, cam_idx]   # [3, 3]

            points_cam = (E @ anchors_homo.T).T          # [900, 4]
            points_2d  = (K @ points_cam[:, :3].T).T     # [900, 3]
            depth = points_2d[:, 2]
            u = points_2d[:, 0] / (depth + 1e-6)
            v = points_2d[:, 1] / (depth + 1e-6)

            # depth > 0 인 것만 유효
            valid_mask = depth > 0.1   # [900] bool

            # 3. 정규화 좌표 [-1, 1] (원본 해상도 기준!)
            u_norm = (u / 640.0) * 2.0 - 1.0
            v_norm = (v / 480.0) * 2.0 - 1.0
            grid   = torch.stack([u_norm, v_norm], dim=-1).view(1, 1, N, 2)

            # 4. 특징 샘플링
            sampled = F.grid_sample(p2_feature, grid,
                                    align_corners=False)   # [1,256,1,900]
            sampled = sampled.view(256, N).T               # [900, 256]

            # ✅ 버그1 수정: in-place 대신 mask 곱하기 (gradient 안전!)
            mask    = valid_mask.float().unsqueeze(1)      # [900, 1]
            sampled = sampled * mask                       # [900, 256]

            agg_features += sampled
            valid_cams   += 1

        if valid_cams > 0:
            agg_features = agg_features / valid_cams

        pred_classes, pred_boxes = self.decoder(agg_features)
        return pred_classes, pred_boxes


if __name__ == "__main__":
    print("🚗 MORAI 자율주행 3D Detection 학습을 시작합니다! 🚗\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[디바이스] {device}\n")

    model      = AutoNavModel().to(device)
    dataset    = MoraiDataset(dataset_dir='./dataset', split='train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True,
                            collate_fn=morai_collate_fn, num_workers=2)
    criterion  = CustomLoss().to(device)
    optimizer  = optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs   = 50
    best_loss    = float('inf')
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        print(f"========== [Epoch {epoch+1}/{num_epochs}] ==========")
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader):
            images     = batch['images'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)

            batch_loss     = 0.0
            batch_cls_loss = 0.0
            batch_box_loss = 0.0
            n = len(batch['dynamic_gt_boxes'])

            for i in range(n):
                gt_boxes   = batch['dynamic_gt_boxes'][i].to(device)
                gt_classes = batch['dynamic_gt_labels'][i].to(device)

                pred_classes, pred_boxes = model(
                    images[i:i+1],
                    intrinsics[i:i+1],
                    extrinsics[i:i+1]
                )

                loss, cls_loss, box_loss = criterion(
                    pred_classes, pred_boxes, gt_classes, gt_boxes
                )

                batch_loss     += loss
                batch_cls_loss += cls_loss.item()
                batch_box_loss += box_loss.item()

            batch_loss = batch_loss / n

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += batch_loss.item()

            if step % 10 == 0:
                print(f"  Step {step:03d} | Loss: {batch_loss.item():.4f} "
                      f"(분류: {batch_cls_loss/n:.4f}, "
                      f"박스: {batch_box_loss/n:.4f})")

        scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"\n🚀 Epoch {epoch+1} 완료! 평균 Loss: {avg_loss:.4f} "
              f"| LR: {scheduler.get_last_lr()[0]:.2e}\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  💾 Best 모델 저장! Loss: {best_loss:.4f}\n")

    print("🎉 학습 완료!")
    torch.save(model.state_dict(), "morai_autonav_weights.pth")
    print("💾 최종 모델 저장 완료: morai_autonav_weights.pth")