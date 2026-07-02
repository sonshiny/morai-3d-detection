import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from loss_calculator import CustomLoss
from make_kmeans import DEFAULT_K, ensure_kmeans_files
from morai_dataset import MoraiDataset, morai_collate_fn
from train import AutoNavModel, compute_auxiliary_detection_loss


def find_batch_with_gt(dataset, batch_size):
    indices = []
    for idx in range(len(dataset)):
        _, stem = dataset.items[idx]
        sample = dataset[idx]
        num_gt = int(sample['dynamic_gt_boxes'].shape[0])
        if num_gt > 0:
            indices.append(idx)
            if len(indices) >= batch_size:
                break

    if not indices:
        raise RuntimeError("GT가 있는 train sample을 찾지 못했습니다.")

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=len(indices),
        shuffle=False,
        collate_fn=morai_collate_fn,
        num_workers=0,
    )
    return next(iter(loader)), indices


def score_stats(model_out):
    cls = model_out['det_cls'].detach().float()
    quality = model_out['det_quality'].detach().float()

    raw = cls.sigmoid().max(dim=-1).values
    calibrated = raw * torch.sigmoid(quality[..., 0])

    parts = []
    for name, scores in (("raw", raw), ("cal", calibrated)):
        parts.append(
            f"{name}: mean={scores.mean().item():.4f} "
            f"max={scores.max().item():.4f} "
            f">0.05={(scores > 0.05).sum().item()} "
            f">0.25={(scores > 0.25).sum().item()}"
        )
    return " | ".join(parts)


def move_batch_to_device(batch, device):
    return {
        **batch,
        'images': batch['images'].to(device),
        'intrinsics': batch['intrinsics'].to(device),
        'extrinsics': batch['extrinsics'].to(device),
        'ego_pose': batch['ego_pose'].to(device),
    }


def main():
    parser = argparse.ArgumentParser(
        description="한 배치를 반복 학습해서 loss/matcher/model 연결이 정상인지 확인합니다."
    )
    parser.add_argument("--dataset-root", default="/data/dataset")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--load", default=None, help="예: best_model.pth 또는 last_checkpoint.pth")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-freeze-bn", action="store_true")
    parser.add_argument("--eval-mode", action="store_true", help="Dropout까지 끄고 overfit 가능성만 확인")
    parser.add_argument("--no-amp", action="store_true", help="AMP/autocast 없이 FP32로 테스트")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[debug] device={device}")

    ensure_kmeans_files(
        dataset_root=args.dataset_root,
        val_scenarios=None,
        k=DEFAULT_K,
        force=False,
    )

    dataset = MoraiDataset(dataset_root=args.dataset_root, split="train", val_scenarios=None)
    batch, indices = find_batch_with_gt(dataset, args.batch_size)
    stems = batch['stem']
    gt_counts = [int(x.shape[0]) for x in batch['dynamic_gt_boxes']]
    print(f"[debug] selected indices={indices}")
    print(f"[debug] stems={stems}")
    print(f"[debug] gt_counts={gt_counts}")

    batch = move_batch_to_device(batch, device)

    model = AutoNavModel(
        num_decoder_layers=6,
        pretrained_backbone=not args.no_pretrained,
        use_temporal_memory=False,
    ).to(device)
    if not args.no_freeze_bn:
        model.freeze_backbone_bn()

    if args.load is not None:
        ckpt = torch.load(args.load, map_location=device)
        state = ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt
        model.load_state_dict(state)
        print(f"[debug] loaded weights: {args.load}")

    criterion = CustomLoss(num_classes=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    first_loss = None
    model.train()
    if not args.no_freeze_bn:
        model.freeze_backbone_bn()
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, args.steps + 1):
        if args.eval_mode:
            model.eval()
        else:
            model.train()
            if not args.no_freeze_bn:
                model.freeze_backbone_bn()
        if hasattr(model, "reset_temporal_memory"):
            model.reset_temporal_memory()

        with torch.cuda.amp.autocast(enabled=use_amp):
            model_out = model(
                batch['images'],
                batch['intrinsics'],
                batch['extrinsics'],
                stems=batch['stem'],
                ego_poses=batch['ego_pose'],
                return_intermediate=True,
            )
            loss, cls_loss, box_loss, quality_loss = compute_auxiliary_detection_loss(
                model_out,
                batch,
                criterion,
                device,
                aux_weight=args.aux_weight,
            )

        if first_loss is None:
            first_loss = float(loss.detach().item())

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 25.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % 10 == 0 or step == args.steps:
            print(
                f"[overfit] step {step:04d}/{args.steps} "
                f"loss={loss.item():.4f} cls={cls_loss.item():.4f} "
                f"box={box_loss.item():.4f} q={quality_loss.item():.4f} | "
                f"{score_stats(model_out)}"
            )

    last_loss = float(loss.detach().item())
    ratio = last_loss / max(first_loss, 1e-8)
    print(f"[debug] first_loss={first_loss:.4f} last_loss={last_loss:.4f} ratio={ratio:.3f}")
    if ratio < 0.7:
        print("[debug] 판정: 한 배치 overfit은 됩니다. 전체 학습 병목/score calibration 쪽을 봐야 합니다.")
    else:
        print("[debug] 판정: 한 배치도 잘 안 내려갑니다. loss/matcher/gradient/model 연결을 우선 봐야 합니다.")


if __name__ == "__main__":
    main()
