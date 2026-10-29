# CIFAR-10 Image Classification — Lab 2 (PyTorch)

> **Môn học:** Deep Learning  
> **Chủ đề:** Hands-on practice with pre-trained neural network architectures (Transfer Learning & Fine-tuning)  
> **Tập dữ liệu:** CIFAR-10 (60.000 ảnh màu 32×32, 10 lớp)

Dự án này tập trung vào thực hành ứng dụng các kiến trúc mạng nơ-ron học sâu đã được huấn luyện trước (**Pre-trained Models** trên ImageNet) bằng PyTorch. Toàn bộ luồng thực nghiệm, huấn luyện, phân tích và đánh giá được thiết kế tập trung duy nhất tại **Jupyter Notebook**, kết hợp cùng gói thư viện tái sử dụng tinh gọn **`going_modular/`**.

---

## 🎯 Kiến trúc và Mô hình Thực nghiệm

| Mô hình | ID | Backbone Pretrained | Cơ chế huấn luyện | Ghi chú |
|---|---|---|---|---|
| **VGG16** | `v0` | `vgg16(weights='DEFAULT')` | Feature Extraction (Đóng băng feature extractor, chỉ train classifier mới) | Kiến trúc CNN cổ điển sâu, nhiều tham số |
| **ResNet-18** | `v1` | `resnet18(weights='DEFAULT')` | Feature Extraction & Fine-tuning | Mạng Residual hiện đại, nhẹ, hội tụ nhanh |
| **DenseNet-121** | — | `densenet121(weights='DEFAULT')` | Khám phá trực tiếp (Inline Demo) | Khám phá kết nối dày đặc (Dense connectivity) |

---

## 📁 Cấu trúc Thư mục Dự án

```
lab2/
│
├── notebooks/
│   └── lab2_pretrained_models.ipynb   # 🎯 NOTEBOOK CHÍNH (Entry point duy nhất)
│
├── going_modular/                     # Thư viện hỗ trợ mô-đun hoá (5 modules)
│   ├── __init__.py                    # Export các hàm và lớp tiện ích
│   ├── data_setup.py                  # Chia train/val/test & DataLoader với ImageNet transform
│   ├── model_builder.py               # Xây dựng CIFAR10ModelV0, CIFAR10ModelV1
│   ├── engine.py                      # train(), eval_model() — hỗ trợ ghi log TensorBoard
│   ├── evaluate.py                    # Confusion matrix, per-class accuracy, biểu đồ trực quan
│   └── utils.py                       # seed, device, lưu/nạp checkpoint (.pth)
│
├── data/                              # Tự động tải CIFAR-10 (~170 MB) khi chạy lần đầu
├── artifacts/                         # Checkpoint + history/config của các lần train (không đưa lên Git)
├── runs/                              # Thư mục log TensorBoard (tự sinh khi huấn luyện)
├── requirements.txt                   # Danh sách thư viện phụ thuộc
├── lab2_plan.md                       # Kế hoạch và checklist triển khai chi tiết
└── README.md                          # Tài liệu hướng dẫn dự án
```

> [!NOTE]
> Dự án được tối giản hóa cho môi trường học tập/thực hành: **không dùng scripts CLI rời rạc**, toàn bộ luồng chạy và tham số được cấu hình inline trong notebook giúp dễ quan sát, đối chiếu và báo cáo kết quả.

---

## 🚀 Hướng dẫn Cài đặt & Chạy

### 1. Chuẩn bị môi trường Python (Python 3.10+)

Mở terminal tại thư mục gốc `lab2`:

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo:
# Trên Windows PowerShell:
.venv\Scripts\Activate.ps1
# Trên Linux / macOS:
# source .venv/bin/activate

# Nâng cấp pip và cài đặt thư viện phụ thuộc
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu dùng GPU NVIDIA, hãy chọn lệnh cài PyTorch phù hợp với hệ điều hành và CUDA tại [trang cài đặt chính thức](https://pytorch.org/get-started/locally/), chạy lệnh đó sau khi cài requirements, rồi kiểm tra `torch.cuda.is_available()`.

### 2. Chạy Notebook

1. Mở VS Code / Jupyter Lab và mở file [notebooks/lab2_pretrained_models.ipynb](notebooks/lab2_pretrained_models.ipynb).
2. Chọn Kernel là môi trường ảo `.venv` vừa tạo.
3. Chạy các cell tuần tự từ trên xuống dưới.

### 3. Lưu và dùng lại kết quả huấn luyện

Trong Section 0, đặt `RUN_MODE = "train"` để huấn luyện các mô hình và lưu checkpoint tốt nhất, lịch sử loss/accuracy cùng cấu hình vào `artifacts/`. Sau khi chạy thành công một lần, đổi thành `RUN_MODE = "reuse"` rồi chạy lại notebook để bỏ qua huấn luyện và tiếp tục các phần so sánh/đánh giá. Chế độ reuse báo lỗi rõ ràng nếu thiếu artifact; không tự âm thầm train lại. Artifact được lưu riêng cho VGG16, ResNet-18 và ResNet-18 fine-tuned.

Trên Colab, notebook tự mount Drive và dùng `MyDrive/Lab2/artifacts/` để giữ kết quả qua các phiên runtime. Đặt `going_modular/` tại `MyDrive/Lab2/`; CIFAR-10 được tải vào vùng tạm `/content/cifar10_data` để đọc nhanh. Trên máy local, code và artifact nằm trong thư mục dự án.

Không commit `artifacts/`, dataset hay TensorBoard logs lên GitHub. Để đổi hyperparameter hoặc tạo một lần chạy mới, chọn `RUN_MODE = "train"`; artifact cùng tên sẽ được cập nhật.

### 4. Giám sát bằng TensorBoard

Trong quá trình huấn luyện ở Section 4, quá trình học (Loss & Accuracy trên tập Train và Validation) sẽ được tự động ghi log vào thư mục `runs/`. Để mở giao diện trực quan:

```bash
# Trong terminal (đã kích hoạt .venv):
tensorboard --logdir=runs
```

Mở trình duyệt truy cập: [http://localhost:6006](http://localhost:6006)

---

## 📖 Nội dung Chi tiết trong Notebook (`lab2_pretrained_models.ipynb`)

Notebook được cấu trúc chặt chẽ thành **8 Section** tương ứng đầy đủ các bước của bài lab:

1. **Section 0 — Setup & Config:** Tự động nhận diện đường dẫn `ROOT`, import thư viện và thiết lập inline hyperparameters (`SEED=42`, `BATCH_SIZE=32`, `LR=1e-3`, `EPOCHS=5`, `VAL_SIZE=5000`).
2. **Section 1 — Load & Visualize Data:** Tải CIFAR-10, phân chia tập (45.000 train / 5.000 val / 10.000 test), kiểm tra kích thước và vẽ lưới ảnh minh họa kèm nhãn.
3. **Section 2 — Khám phá kiến trúc Pre-trained:** Soi chi tiết cấu trúc mạng VGG16, ResNet-18 qua `torchinfo.summary()`; phân tích số lượng tham số tổng vs tham số trainable; demo inline kiến trúc DenseNet-121.
4. **Section 3 — Kiểm tra model đã adapt:** Classifier 10 lớp được cấu hình trong `model_builder.py`; notebook xác nhận output shape `[2, 10]` và giải thích chiến lược đóng băng (freezing).
5. **Section 4 — Train & Monitor với TensorBoard:** Huấn luyện `v0` (VGG16) và `v1` (ResNet-18) với `engine.train()`; chọn checkpoint tốt nhất theo validation accuracy, lưu weights/history/config vào `artifacts/`, log vào TensorBoard; hỗ trợ nạp lại bằng `RUN_MODE="reuse"`.
6. **Section 5 — So sánh Models:** So sánh đường cong Loss/Accuracy qua các epoch, so sánh thời gian huấn luyện và hiệu quả hội tụ giữa VGG16 và ResNet-18.
7. **Section 6 — Thử nghiệm Fine-tuning:** Mở đóng băng (unfreeze) một số block cuối của ResNet-18, huấn luyện bổ sung với learning rate nhỏ (`1e-4`) và so sánh đối chứng giữa Frozen vs Fine-tuned.
8. **Section 7 — Đánh giá trên tập Test:** Đánh giá khách quan mô hình tốt nhất trên tập Test (10.000 ảnh chưa từng thấy); vẽ ma trận nhầm lẫn (**Confusion Matrix**), thống kê độ chính xác theo từng lớp (**Per-class Accuracy**) và tìm ra các cặp nhãn hay nhầm lẫn nhất.

---

## ⚙️ Quy ước Tiền xử lý Dữ liệu (Transforms)

Tiền xử lý được lấy trực tiếp từ weights tương ứng trong torchvision: `VGG16_Weights.IMAGENET1K_V1.transforms()` cho VGG16 và `ResNet18_Weights.IMAGENET1K_V1.transforms()` cho ResNet-18. Với hai weights hiện đang dùng, cấu hình là:

| Thành phần | Thông số | Mô tả |
|---|---|---|
| **Resize** | Cạnh ngắn lên `256` | Phóng to ảnh, giữ nguyên tỷ lệ khung hình |
| **Center crop** | `(224, 224)` | Cắt vùng giữa ảnh theo pipeline của weights |
| **ToTensor** | Chuyển đổi | Chuyển pixel về tensor và dải `[0.0, 1.0]` |
| **Normalize** | Mean/std lấy từ weights | Chuẩn hóa theo cấu hình ImageNet-1K đi kèm weights |

`get_pretrained_transform(model_name="vgg16")` và `get_pretrained_transform(model_name="resnet18")` chọn preset riêng cho từng kiến trúc. `create_train_val_test_dataloaders(transform_model=...)` dùng lựa chọn này nếu không truyền transform tùy chỉnh. Hai preset hiện tại có cùng cấu hình, nên notebook dùng chung DataLoader cho cả hai model. Khi bật augmentation cho tập train, `RandomHorizontalFlip` được áp dụng trước pipeline của weights; validation và test không augmentation.

---

## 🛠️ Xử lý Sự cố Thường gặp

| Hiện tượng | Nguyên nhân | Hướng khắc phục |
|---|---|---|
| `No module named 'going_modular'` | Kernel không nhận đúng thư mục cha | Section 0 trong notebook đã có cell tự động thêm `ROOT` vào `sys.path`. Đảm bảo đã chạy Section 0. |
| Tải Pretrained Weights rất chậm | Trọng số mạng (đặc biệt là VGG16 ~528MB) lớn | Quá trình tải chỉ diễn ra một lần duy nhất (được cache tại thư mục `~/.cache/torch/hub/checkpoints`). Hãy kiên nhẫn đợi tải xong. |
| `CUDA out of memory` | Bộ nhớ GPU không đủ khi xử lý batch 224×224 | Giảm `BATCH_SIZE` xuống `16` trong cell cấu hình ở Section 0. |
| Cảnh báo BatchNorm khi backbone bị freeze | PyTorch giữ eval mode cho BatchNorm | `engine.py` đã hỗ trợ tham số `frozen_backbone` để giữ trạng thái `eval()` ổn định cho BatchNorm khi freeze. |
