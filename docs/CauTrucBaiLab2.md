# Mô tả chi tiết cấu trúc bài Lab 2 — Transfer Learning với Pre-trained Models trên CIFAR-10

> **Môn học:** Deep Learning — Thực hành 2
> **Bài toán:** Phân loại ảnh CIFAR-10 (60.000 ảnh màu 32×32, 10 lớp) bằng các kiến trúc đã huấn luyện sẵn trên ImageNet.
> **File này:** Tổng hợp cấu trúc thư mục, vai trò từng thành phần, luồng chạy và các quy ước của dự án để người đọc nắm toàn cảnh bài lab.

---

## 1. Mục tiêu bài tập (theo đề)

1. Thử nghiệm với các mô hình pre-trained khác nhau (VGG, DenseNet, v.v.).
2. Điều chỉnh các siêu tham số (learning rate, batch size, v.v.).
3. Sử dụng TensorBoard để theo dõi quá trình huấn luyện.

**Mục tiêu học tập:** làm quen tải/sử dụng mô hình pre-trained từ `torchvision.models`; hiểu cách **adapt** (đổi head, đóng băng backbone) và **fine-tune** mô hình cho tác vụ cụ thể.

## 2. Kiến trúc mô hình thực nghiệm

| Mô hình | Run ID | Backbone | Chiến lược | Ghi chú |
|---|---|---|---|---|
| VGG16 | `v0_vgg16` | `vgg16(weights='IMAGENET1K_V1')` | Feature Extraction (freeze feature extractor, chỉ train classifier mới) | CNN cổ điển, nhiều tham số (~528 MB weights) |
| ResNet-18 | `v1_resnet18` | `resnet18(weights='IMAGENET1K_V1')` | Feature Extraction | Baseline; residual nhẹ, hội tụ nhanh |
| ResNet-18 | `v1_resnet18_finetune` | Nạp `best_model.pth` của baseline rồi unfreeze `layer4` | Fine-Tuning (lr backbone 1e-4) | Run fine-tune duy nhất; phải chạy sau `v1_resnet18` |

Mỗi run `v1_resnet18_finetune` là **run fine-tune duy nhất** của bài — các run còn lại đều là feature extraction.

## 3. Cây thư mục

```
lab2/
├── notebooks/
│   └── pretrained_models.ipynb   # NOTEBOOK CHÍNH — entry point duy nhất, chứa toàn bộ luồng thực nghiệm
│
├── going_modular/                # Package thư viện tái sử dụng (logic được tách khỏi notebook)
│   ├── __init__.py               # Export API của package (~99 dòng)
│   ├── data_setup.py             # (~205 dòng) Chia train/val/test, transforms ImageNet, tạo DataLoader
│   ├── model_builder.py          # (~281 dòng) Dựng mô hình pre-trained, wrapper backbone+head, freeze/unfreeze, đếm tham số
│   ├── engine.py                 # (~259 dòng) Vòng lặp train/eval, chọn best checkpoint theo val acc, TensorBoard, checkpoint/resume
│   ├── evaluate.py               # (~476 dòng) Suy luận trên Test: metrics, confusion matrix, per-class accuracy, biểu đồ
│   └── utils.py                  # (~438 dòng) set_seed, chọn device, lưu/nạp artifacts, history schemas, atomic write
│
├── tests/                        # Bộ kiểm thử pytest (chạy CPU, không cần mạng/GPU cho phần lớn)
│   ├── conftest.py               # Fixture chung
│   ├── test_smoke.py             # Smoke test: import package
│   ├── test_data_setup.py        # (~388 dòng)
│   ├── test_model_builder.py     # (~359 dòng)
│   ├── test_engine.py            # (~530 dòng)
│   ├── test_utils.py             # (~335 dòng)
│   └── test_evaluate.py          # (~507 dòng)
│
├── docs/                         # Hệ thống tài liệu đặc tả + kế hoạch + phân tích
│   ├── yeucaugoc.md              # Đề bài gốc của giảng viên
│   ├── thinh_notes.md            # Ghi chú tìm hiểu chức năng going_modular (Huỳnh Quang Thịnh)
│   ├── define&designt/           # ĐẶC TẢ THIẾT KẾ (spec)
│   │   ├── 00_lab2_description.md        # Spec tổng quan: mục tiêu, FR/NFR, ràng buộc, danh mục thí nghiệm
│   │   ├── 01_data.md                   # Spec data_setup
│   │   ├── 02_model.md                  # Spec model_builder
│   │   ├── 03_engine.md                 # Spec engine
│   │   ├── 04_utils.md                  # Spec utils
│   │   ├── 05_evaluation_results.md     # Spec evaluate
│   │   └── tests/                       # Spec kiểm thử 00_test_plan → 05_test_evaluate
│   ├── plans/                     # KẾ HOẠCH TRIỂN KHAI theo phase
│   │   ├── implementation_plan.md       # Kế hoạch tổng
│   │   ├── phase0.md … phase8.md        # Checklist từng phase (0: chốt spec → 8: chuẩn hóa config/Colab/Drive)
│   │   └── run-orchestration-checklist-plan.md  # Kế hoạch điều phối multi-run trong notebook
│   └── analysts/                  # BÁO CÁO PHÂN TÍCH / rà soát
│       ├── spec_vs_code_analyst.md       # Đối chiếu spec vs code (2026-11-04)
│       ├── Spec_Review_Report.md         # Nhật ký chốt đặc tả
│       ├── notebook_status_checklist.md  # Trạng thái notebook (cập nhật 2026-11-11)
│       ├── problems.md                   # Vấn đề đã phát hiện (transform preset, DI)
│       ├── transfer_learning_experiments_analyst.md  # Kế hoạch/tiến độ thực nghiệm
│       ├── dated-outputs-logging-plan.md # Kế hoạch tách artifacts/outputs/logs
│       ├── 05_pytorch_going_modular.md   # Tài liệu tham khảo (mrdbourke, going modular)
│       └── 06_pytorch_transfer_learning.ipynb  # Notebook tham khảo transfer learning
│
├── requirements.txt               # Dependencies: torch, torchvision, torchinfo, tensorboard,
│                                  # matplotlib, pandas, numpy, pillow, pytest, tqdm, ipykernel
├── README.md                      # Tài liệu hướng dẫn chính của dự án
└── (runtime) data/, artifacts/, outputs/, logs/, runs/ — thư mục sinh ra khi chạy, không commit
```

## 4. Vai trò từng module `going_modular`

Luồng tổng thể: `Notebook → data_setup + model_builder → engine.train → artifacts → evaluate`

| Module | Trách nhiệm chính |
|---|---|
| `data_setup.py` | `get_pretrained_transform(model_name)` chọn preset transform theo weights (VGG16/ResNet18 `IMAGENET1K_V1.transforms()`); `create_train_val_test_dataloaders()` chia 45.000 train / 5.000 val / 10.000 test; augmentation (RandomHorizontalFlip) chỉ cho train |
| `model_builder.py` | `build_model()` nạp backbone torchvision + thay head 10 lớp; hỗ trợ `freeze` toàn bộ hoặc `unfreeze_last_n` khối cuối; `count_params()` thống kê total/trainable/frozen |
| `engine.py` | `train()`/`train_step`/`evaluate_step` (loss tích lũy có trọng số theo mẫu), chọn best model theo **validation** accuracy, lưu `config.json`/`history.json`/`best_model.pth`/`last_checkpoint.pth`, ghi TensorBoard, hỗ trợ `resume`; `set_train_mode` giữ BatchNorm frozen ở eval mode |
| `utils.py` | `set_seed(42)` (tái lập), chọn `device` (cuda/cpu), quản lý artifacts atomic write, history key schemas, dated outputs |
| `evaluate.py` | Suy luận tập Test độc lập: test accuracy, confusion matrix, per-class accuracy, top confused pairs; xuất metrics JSON + hình vào `outputs/YYYY-MM-DD/<RUN_ID>/<EVAL_ID>/` |

## 5. Notebook — 8 Section tương ứng 7 bước đề bài

`notebooks/pretrained_models.ipynb` là entry point duy nhất; toàn bộ tham số cấu hình inline, **không có scripts CLI rời rạc**.

1. **Section 0 — Setup & Config:** nhận diện `ROOT`, import, hyperparameters (`SEED=42`, `BATCH_SIZE=32`, `LR=1e-3`, `EPOCHS`, `VAL_SIZE=5000`).
2. **Section 1 — Load & Visualize Data:** tải CIFAR-10, chia tập, vẽ lưới ảnh kèm nhãn.
3. **Section 2 — Khám phá kiến trúc Pre-trained:** `torchinfo.summary()` VGG16/ResNet-18; phân tích tham số tổng vs trainable.
4. **Section 3 — Kiểm tra model đã adapt:** xác nhận output shape `[batch, 10]`, giải thích chiến lược freeze.
5. **Section 4 — Train & Monitor TensorBoard:** `engine.train()` cho các run; chọn checkpoint theo val acc; hỗ trợ `RUN_MODE="reuse"`.
6. **Section 5 — So sánh Models:** đường cong loss/accuracy, thời gian huấn luyện, tốc độ hội tụ VGG16 vs ResNet-18.
7. **Section 6 — Thử nghiệm Fine-tuning:** unfreeze `layer4` ResNet-18, lr nhỏ 1e-4, so sánh Frozen vs Fine-tuned.
8. **Section 7 — Đánh giá trên Test:** accuracy, confusion matrix, per-class accuracy, các cặp nhãn nhầm lẫn nhiều nhất.

### Điều phối multi-run trong notebook

- Chọn run bằng `RUN_ID` (đơn) hoặc `RUN_IDS` (list/chuỗi phân tách phẩy) — chạy **tuần tự**.
- Người dùng tự chọn hành động qua `RUN_PLAN` / `DEFAULT_RUN_ACTION`:
  - `train`: huấn luyện từ đầu; artifacts cũ không tự xóa, chỉ train lại khi `FORCE_RETRAIN=True`.
  - `resume`: tiếp tục từ `last_checkpoint.pth`, yêu cầu `done_epochs < target_epochs`.
  - `reuse`: nạp `config/history/best_model.pth`, không train; thiếu `test_metrics.json` thì đánh giá Test một lần.
- Fine-tune train mới nạp `v1_resnet18/best_model.pth`; resume nạp checkpoint của chính run fine-tune.

## 6. Quy ước dữ liệu & artifacts

| Nhánh | Đường dẫn | Nội dung | Commit Git |
|---|---|---|---|
| `artifacts/` | `artifacts/<RUN_ID>/` | Trạng thái train: `config.json`, `history.json`, `best_model.pth`, `last_checkpoint.pth` | Có (trừ `.pth`) |
| `outputs/` | `outputs/YYYY-MM-DD/<RUN_ID>/<EVAL_ID>/` | `test_metrics.json`, ảnh confusion matrix, per-class, dự đoán sai | Có (JSON/PNG) |
| `logs/` (a.k.a `runs/`) | `runs/<RUN_ID>/<SESSION_ID>/` | TensorBoard events mỗi phiên train/resume | Không |
| `data/` | `data/` | CIFAR-10 (~170 MB) tự tải lần đầu | Không |

- `YYYY-MM-DD` theo múi giờ `Asia/Ho_Chi_Minh`; `EVAL_ID`/`SESSION_ID` là timestamp `YYYYMMDD-HHMMSS+07:00` tránh ghi đè.
- `.pth` lớn (VGG16 ≈ 512 MiB) phát hành ngoài Git (Release/Drive), README ghi link.

## 7. Cấu hình mặc định & ràng buộc

- **Defaults:** `seed=42`, `val_size=5000` (10% train gốc), `batch_size=32`, `lr_head=1e-3` (Adam), `epochs=3` (có thể tăng 5 khi chạy chính thức), fine-tune `lr_backbone=1e-4`, `unfreeze_last_n=1` (`layer4`).
- **Transforms:** lấy từ `VGG16_Weights/ResNet18_Weights.IMAGENET1K_V1.transforms()` — Resize cạnh ngắn 256 → CenterCrop 224×224 → ToTensor → Normalize mean/std ImageNet; augmentation chỉ áp cho train.
- **Ràng buộc:** CIFAR-10 chỉ có Train 50k/Test 10k → val tách từ train; dùng `weights=` tường minh (không `pretrained=True` đã deprecated); ảnh 32×32 upscale lên 224×224; giả định có GPU; Test chỉ dùng đánh giá cuối (không rò rỉ dữ liệu); cùng seed cho tái lập.
- **Yêu cầu phi chức năng:** NFR-1 tái lập, NFR-2 không rò rỉ test, NFR-3 chạy được Colab + máy cá nhân, NFR-4 unit test nhanh không cần mạng/GPU.

## 8. Quy trình làm việc theo tài liệu (docs/)

Dự án được phát triển theo quy trình spec-driven:

1. **`docs/yeucaugoc.md`** — đề bài gốc (7 bước).
2. **`docs/define&designt/00–05`** — đặc tả từng module + spec test (`tests/00–05`), `00` là nguồn sự thật duy nhất cho mục tiêu/FR/NFR/danh mục thí nghiệm.
3. **`docs/plans/`** — `implementation_plan.md` + checklist `phase0`–`phase8` triển khai từng module (0: chốt spec → 1: hạ tầng test → 2–6: model_builder/data_setup/utils/engine/evaluate → 7: tích hợp notebook → 8: chuẩn hóa config Colab/Drive).
4. **`docs/analysts/`** — các báo cáo rà soát: đối chiếu spec vs code, trạng thái notebook, vấn đề phát hiện, kế hoạch artifacts/outputs/logs.

## 9. Cách chạy

- **Môi trường:** Python ≥3.10; `python -m venv .venv` → `pip install -r requirements.txt`. GPU NVIDIA thì cài torch theo CUDA phù hợp; Colab chọn GPU runtime.
- **Test:** `pytest tests/test_smoke.py -v` (nhanh); `pytest tests/ -m "not slow" -q`; `pytest tests/ -v` (gồm nhóm `slow` cần mạng để tải weights/CIFAR-10).
- **Thực nghiệm:** mở `notebooks/pretrained_models.ipynb`, chọn kernel `.venv`, chạy tuần tự Section 0→7; giám sát `tensorboard --logdir=runs` tại `localhost:6006`. Trên Colab: data đặt ở `MyDrive/lab2/data/`, code clone từ GitHub.

## 10. Lưu ý khi đọc code (điểm cần chú ý)

- `data_setup` **không import** `model_builder`; transform được inject qua `transform_model`/tham số — tránh coupling.
- Notebook từng có vấn đề transform hardcode `"vgg16"` cho cả ResNet — đã được ghi nhận trong `docs/analysts/problems.md`; kiểm tra notebook dùng đúng preset theo `RUN_ID`.
- `engine.train()` tích lũy batch-loss có trọng số theo mẫu; chọn best checkpoint theo validation, không dùng test.
- Không commit `data/`, `runs/`, `outputs/` logs, file `.pth`.
