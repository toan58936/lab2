# CIFAR-10 Image Classification — Lab 2 (PyTorch)

> **Môn học:** Deep Learning  
> **Chủ đề:** Hands-on practice with pre-trained neural network architectures (Transfer Learning & Fine-tuning)  
> **Tập dữ liệu:** CIFAR-10 (60.000 ảnh màu 32×32, 10 lớp)

Dự án này tập trung vào thực hành ứng dụng các kiến trúc mạng nơ-ron học sâu đã được huấn luyện trước (**Pre-trained Models** trên ImageNet) bằng PyTorch. Toàn bộ luồng thực nghiệm, huấn luyện, phân tích và đánh giá được thiết kế tập trung duy nhất tại **Jupyter Notebook**, kết hợp cùng gói thư viện tái sử dụng tinh gọn **`going_modular/`**.

---

## 🎯 Kiến trúc và Mô hình Thực nghiệm

| Mô hình | ID | Backbone Pretrained | Cơ chế huấn luyện | Ghi chú |
|---|---|---|---|---|
| **VGG16** | `v0` | `vgg16(weights='IMAGENET1K_V1')` | Feature Extraction (native torchvision, G1: đóng băng backbone, chỉ train `classifier[6]` 10 lớp) | Kiến trúc CNN cổ điển sâu; head 4096→10 (FE trainable 40,970) |
| **ResNet-18** | `v1` | `resnet18(weights='IMAGENET1K_V1')` | Feature Extraction & Fine-tuning (native: `fc` 512→10) | Mạng Residual hiện đại, nhẹ, hội tụ nhanh |

> **Hướng B (native, hiện hành):** `build_model` trả model torchvision đã adapt trực tiếp (không wrapper `MyModel`), `state_dict` dùng key gốc. Checkpoint/artifacts format cũ (`backbone.*`/`head.*`) **không tương thích** (không migrate). Khi chuyển sang format mới cần `FORCE_RETRAIN=True` hoặc dọn `artifacts/v0_vgg16,v1_resnet18,v1_resnet18_finetune` và train lại theo thứ tự `v0_vgg16` → `v1_resnet18` → `v1_resnet18_finetune`. `utils.load_model`/`load_checkpoint` sẽ báo lỗi rõ nếu phát hiện format cũ.

---

## 📁 Cấu trúc Thư mục Dự án

```
lab2/
│
├── notebooks/
│   └── pretrained_models.ipynb   # 🎯 NOTEBOOK CHÍNH (Entry point duy nhất)
│
├── going_modular/                     # Thư viện hỗ trợ mô-đun hoá (modules)
│   ├── __init__.py                    # Export các hàm và lớp tiện ích
│   ├── data_setup.py                  # Chia train/val/test & DataLoader với ImageNet transform
│   ├── model_builder.py               # Xây dựng và tùy biến mô hình Pre-trained
│   ├── engine.py                      # Vòng lặp huấn luyện và kiểm định
│   ├── evaluate.py                    # Confusion matrix, per-class accuracy, biểu đồ trực quan
│   └── utils.py                       # seed, device, lưu/nạp checkpoint (.pth), guard format native
│
├── data/                              # Tự động tải CIFAR-10 (~170 MB) khi chạy lần đầu (không commit)
├── artifacts/                         # Kết quả sinh ra khi chạy (config/history/metrics commit, *.pth không commit)
│   ├── figures/                       # Biểu đồ xuất ra từ quá trình đánh giá
│   └── <tên thí nghiệm>/              # Mỗi thí nghiệm một thư mục
├── runs/                              # Thư mục log TensorBoard (không commit)
├── tests/                             # Bộ kiểm thử tự động (conftest.py + test module)
│   ├── test_data_setup.py
│   ├── test_model_builder.py
│   ├── test_engine.py
│   ├── test_utils.py
│   ├── test_evaluate.py
│   └── test_smoke.py
├── docs/                              # Hệ thống tài liệu đặc tả thiết kế
│   ├── define&designt/               # Define & Design Specs (00–05)
│   │   └── tests/                    # Bộ đặc tả kiểm thử (00_test_plan.md – 05_test_evaluate.md)
│   ├── plans/                        # Kế hoạch triển khai theo phase (phase0.md, phase1.md, implementation_plan.md)
│   └── analysts/                     # Báo cáo phân tích (spec_vs_code_analyst.md, Spec_Review_Report.md)
├── requirements.txt                   # Danh sách thư viện phụ thuộc
└── README.md                          # Tài liệu hướng dẫn dự án
```

> [!NOTE]
> Dự án được tối giản hóa cho môi trường học tập/thực hành: **không dùng scripts CLI rời rạc**, toàn bộ luồng chạy và tham số được cấu hình inline trong notebook giúp dễ quan sát, đối chiếu và báo cáo kết quả.

---

## 🚀 Khởi tạo môi trường

Các lệnh dưới đây dành cho Windows PowerShell, chạy từ thư mục gốc của repository. Dự án yêu cầu Python 3.10 trở lên.

### Windows PowerShell

```powershell
# Tạo môi trường ảo (chỉ cần chạy một lần)
python -m venv .venv

# Kích hoạt môi trường ảo trong PowerShell
.venv\Scripts\Activate.ps1

# Nếu PowerShell chặn script kích hoạt, có thể bỏ qua kích hoạt và dùng
# trực tiếp .venv\Scripts\python.exe trong các lệnh bên dưới.

python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu cần dùng phiên bản Python cụ thể đã cài đặt, có thể tạo môi trường bằng Python Launcher, ví dụ:

```powershell
py -3.10 -m venv .venv
```

> Nếu sử dụng GPU NVIDIA trên máy cá nhân, hãy chọn lệnh cài PyTorch phù hợp với hệ điều hành/CUDA từ [trang cài đặt PyTorch](https://pytorch.org/get-started/locally/). Sau đó xác nhận `torch.cuda.is_available()` trả về `True`. Trên Google Colab, chọn GPU trong **Runtime → Change runtime type**.

## 🧪 Chạy kiểm thử

Chạy từ thư mục gốc repository. Trên Windows, có thể dùng trực tiếp Python trong `.venv`, không phụ thuộc việc môi trường đã được activate hay chưa.

```powershell
# Smoke test nhanh để xác nhận package import được
.venv\Scripts\python.exe -m pytest tests\test_smoke.py -v

# Bộ test nhanh, bỏ qua các test được đánh dấu slow
.venv\Scripts\python.exe -m pytest tests\ -m "not slow" -q

# Toàn bộ test, bao gồm cả các test slow
.venv\Scripts\python.exe -m pytest tests\ -v
```

Lệnh cuối bao gồm các test `slow`: một số test tải CIFAR-10 hoặc ImageNet weights và cần kết nối mạng. Nếu mạng/dữ liệu không sẵn sàng, một số test dữ liệu có thể được skip; hãy đọc phần tổng kết của pytest để phân biệt `passed`, `skipped`, `failed` và `deselected`.

Có thể chạy riêng nhóm test chậm hoặc một module:

```powershell
.venv\Scripts\python.exe -m pytest tests\ -m slow -v
.venv\Scripts\python.exe -m pytest tests\test_model_builder.py -m "not slow" -q
.venv\Scripts\python.exe -m pytest tests\test_engine.py -q
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest tests/ -m "not slow" -q
python -m pytest tests/ -v  # toàn bộ test, có cả nhóm slow
```

## 📓 Mở Notebook và chạy thực nghiệm

Notebook chính là [notebooks/pretrained_models.ipynb](notebooks/pretrained_models.ipynb).

### Chạy trên máy cá nhân

1. Mở notebook trong VS Code (cần extension Jupyter) hoặc JupyterLab.
2. Chọn kernel `.venv` của repository.
3. Chuẩn bị CIFAR-10 theo cấu trúc torchvision nhận tại `data/`. Notebook hiện dùng `download=False`; pretrained weights có thể được tải khi khởi tạo model ở luồng train.

### Chạy trên Google Colab

1. Đặt dữ liệu CIFAR-10 trong `MyDrive/lab2/data/` (tên thư mục phân biệt chữ hoa/thường). Không cần chép `going_modular/` lên Drive; notebook clone code từ GitHub vào Colab runtime.
2. Mở notebook `notebooks/pretrained_models.ipynb` từ checkout GitHub trong Colab và chọn GPU nếu có.
3. Chạy cell setup để mount Drive. Notebook dùng `MyDrive/lab2` làm data/artifact root, `/content/lab2` cho code checkout và `/content/cifar10_data` làm bản dữ liệu runtime.
4. Với code hiện tại, sau khi cấu hình `RUN_ID`/`RUN_IDS` và kiểm tra setup, chạy tuần tự các cell data → model → train/reuse → so sánh → đánh giá để thực thi (các) run đã chọn. Có thể chạy từng run riêng lẻ hoặc truyền danh sách run để chạy liên tiếp.

### Trạng thái và hành động các run

Notebook hỗ trợ chọn **1 hoặc nhiều run chạy tuần tự** qua `RUN_ID` hoặc `RUN_IDS`, và **người dùng tự chọn hành động** cho từng run qua `RUN_PLAN` / `DEFAULT_RUN_ACTION` (code không tự suy luận):

- `train`: huấn luyện đầy đủ từ đầu. Nếu artifacts đã tồn tại, không tự xóa; chỉ train lại khi `FORCE_RETRAIN=True`.
- `resume`: tiếp tục từ `last_checkpoint.pth` khi dang dở (phù hợp Colab ngắt kết nối), yêu cầu `done_epochs < target_epochs`.
- `reuse`: nạp artifacts có sẵn (`config/history/best_model.pth`), không train.

Ví dụ cấu hình:

```python
RUN_IDS = ["v0_vgg16", "v1_resnet18"]
RUN_PLAN = {"v0_vgg16": "reuse", "v1_resnet18": "resume"}
DEFAULT_RUN_ACTION = "train"
FORCE_RETRAIN = False
```

Mỗi run build/train/resume/reuse/đánh giá theo spec riêng, lưu artifacts vào `artifacts/<RUN_ID>/` và outputs vào `outputs/<date>/<RUN_ID>/<EVAL_ID>/`. Hiện có 3 run: `v0_vgg16`, `v1_resnet18` (feature extraction) và `v1_resnet18_finetune` (**run fine-tune duy nhất**). Fine-tune: train mới nạp `v1_resnet18/best_model.pth`; resume nạp checkpoint của chính run fine-tune. Test set chỉ dùng đánh giá cuối.

Không commit dataset, TensorBoard logs hoặc checkpoint `.pth` lên GitHub. Config, history, metrics và hình kết quả được quản lý theo quy ước artifacts của dự án. **Lưu ý:** do chuyển sang state_dict native (hướng B), checkpoint/artifacts cũ không dùng được; cần retrain 3 run hiện hành khi chạy format mới.

### 4. Giám sát bằng TensorBoard

Trong quá trình huấn luyện, loss/accuracy (train/validation) được ghi log TensorBoard theo từng run/phiên (`logs/<RUN_ID>/<SESSION_ID>/`). Để mở giao diện trực quan:

```bash
# Trong terminal (đã kích hoạt .venv), trỏ đến thư mục log phù hợp:
tensorboard --logdir=logs
```

Mở trình duyệt truy cập: [http://localhost:6006](http://localhost:6006)

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
| Tải Pretrained Weights rất chậm | Trọng số mạng (đặc biệt là VGG16, checkpoint native G1 ~512MB) lớn | Quá trình tải chỉ diễn ra một lần duy nhất (được cache tại thư mục `~/.cache/torch/hub/checkpoints`). Hãy kiên nhẫn đợi tải xong. |
| `CUDA out of memory` | Bộ nhớ GPU không đủ khi xử lý batch 224×224 | Giảm `BATCH_SIZE` xuống `16` trong cell cấu hình ở Section 0. |
| Cảnh báo BatchNorm khi backbone bị freeze | PyTorch giữ eval mode cho BatchNorm | `set_train_mode(model, cfg)` trong `engine.py` giữ trạng thái `eval()` cho BatchNorm của phần đóng băng khi train. |
## Ghi chú tìm hiểu dự án
- Đã tìm hiểu cấu trúc thư mục của dự án.
- Thư mục going_modular chứa các module Python.
- Thư mục tests chứa các bài kiểm thử.
- Thư mục notebooks chứa các notebook phục vụ thử nghiệm.