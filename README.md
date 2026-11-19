# CIFAR-10 Image Classification — Lab 2 (PyTorch)

> **Môn học:** Deep Learning  
> **Chủ đề:** Hands-on practice with pre-trained neural network architectures (Transfer Learning & Fine-tuning)  
> **Tập dữ liệu:** CIFAR-10 (60.000 ảnh màu 32×32, 10 lớp)

Dự án này tập trung vào thực hành ứng dụng các kiến trúc mạng nơ-ron học sâu đã được huấn luyện trước (**Pre-trained Models** trên ImageNet) bằng PyTorch. Toàn bộ luồng thực nghiệm, huấn luyện, phân tích và đánh giá được điều phối tại **Jupyter Notebook**, kết hợp cùng gói thư viện tái sử dụng tinh gọn **`going_modular/`**.

---

## 🎯 Kiến trúc và Mô hình Thực nghiệm

Dự án có **3 run** hiện hành (artifacts, outputs và logs tách riêng theo `RUN_ID`):

| Run (`RUN_ID`) | Mô hình | Cơ chế huấn luyện | Epochs | LR (head / backbone) | Batch | Khởi tạo từ | Mục đích |
|---|---|---|---|---|---|---|---|
| `v0_vgg16` | VGG16 (`vgg16(weights='IMAGENET1K_V1')`) | Feature Extraction (G1: đóng băng backbone, chỉ train `classifier[6]` 4096→10, trainable 40,970 tham số) | 20 (early stopping) | 1e-3 / — | 32 | ImageNet | So sánh kiến trúc |
| `v1_resnet18` | ResNet-18 (`resnet18(weights='IMAGENET1K_V1')`) | Feature Extraction (chỉ train `fc` 512→10) | 20 (early stopping) | 1e-3 / — | 32 | ImageNet | **Baseline** |
| `v1_resnet18_finetune` | ResNet-18 | Fine-Tuning: mở khóa `layer4` + `fc` | 20 (early stopping) | 1e-3 / 1e-4 | 32 | `v1_resnet18/best_model.pth` | Tác động của Fine-Tuning (**run fine-tune duy nhất**) |

> **Ghi chú epochs:** cấu hình hiện tại dùng 20 epoch tối đa với Early Stopping (patience=5, min_delta=1e-4) và LR scheduler ReduceLROnPlateau (factor=0.5, patience=3). Huấn luyện có thể dừng sớm nếu Validation không cải thiện.

> **Hướng B (native, hiện hành):** `build_model` trả model torchvision đã adapt trực tiếp (không wrapper `MyModel`), `state_dict` dùng key gốc (`conv1.*`, `layer*`, `fc.*` cho ResNet-18; `features.*`, `classifier.*` cho VGG16). Checkpoint/artifacts format cũ (`backbone.*`/`head.*`) **không tương thích** (không migrate). Khi chuyển sang format mới cần `FORCE_RETRAIN=True` hoặc dọn `artifacts/v0_vgg16`, `artifacts/v1_resnet18`, `artifacts/v1_resnet18_finetune` rồi train lại theo thứ tự `v0_vgg16` → `v1_resnet18` → `v1_resnet18_finetune`. `utils.load_model` / `utils.load_checkpoint` sẽ báo lỗi rõ nếu phát hiện format cũ.

---

## 📁 Cấu trúc Thư mục Dự án

```
lab2/
│
├── notebooks/
│   └── pretrained_models.ipynb   # 🎯 NOTEBOOK CHÍNH (entry point duy nhất, điều phối các run)
│
├── going_modular/                     # Thư viện hỗ trợ mô-đun hoá
│   ├── __init__.py                    # Export API (12 symbol flat theo hợp đồng spec 00 mục 2.8.6)
│   ├── data_setup.py                  # Chia train/val/test (stratified) & DataLoader với transform của weights
│   ├── model_builder.py               # ModelConfig, build_model, count_params, set_train_mode, get_param_groups
│   ├── engine.py                      # train / train_step / evaluate_step, chọn best model, checkpoint & resume
│   ├── evaluate.py                    # Suy luận, confusion matrix, per-class accuracy, biểu đồ, bảng so sánh
│   └── utils.py                       # seed, device, lưu/nạp artifacts & checkpoint, session/eval ID, guard format native
│
├── data/                              # CIFAR-10 (không commit)
├── artifacts/                         # Trạng thái train để resume/reuse
│   └── <RUN_ID>/                      # config.json, history.json, best_model.pth, last_checkpoint.pth
├── outputs/                           # Kết quả đánh giá
│   └── YYYY-MM-DD/<RUN_ID>/<EVAL_ID>/ # test_metrics.json, confusion matrix, per-class, dự đoán sai (PNG)
├── logs/                              # TensorBoard (không commit)
│   └── <RUN_ID>/<SESSION_ID>/
├── tests/                             # Bộ kiểm thử tự động (pytest)
│   ├── conftest.py                    # Đăng ký marker `slow`, Matplotlib backend Agg
│   ├── test_data_setup.py
│   ├── test_model_builder.py
│   ├── test_engine.py
│   ├── test_utils.py
│   ├── test_evaluate.py
│   └── test_smoke.py
├── docs/                              # Hệ thống tài liệu đặc tả thiết kế
│   ├── yeucaugoc.md                   # Đề bài gốc
│   ├── define&designt/                # Define & Design Specs (00–05)
│   │   └── tests/                     # Đặc tả kiểm thử (00_test_plan.md – 05_test_evaluate.md)
│   ├── plans/                         # Kế hoạch triển khai (implementation_plan.md, phase0–phase8, run-orchestration-checklist-plan.md)
│   └── analysts/                      # Báo cáo phân tích (Spec_Review_Report.md, spec_vs_code_analyst.md, option_b_native_torchvision_model.md, problems.md, ...)
├── requirements.txt                   # Danh sách thư viện phụ thuộc
└── README.md                          # Tài liệu hướng dẫn dự án
```

> [!NOTE]
> Dự án được tối giản hóa cho môi trường học tập/thực hành: **không dùng scripts CLI rời rạc**, toàn bộ luồng chạy và tham số được cấu hình inline trong notebook giúp dễ quan sát, đối chiếu và báo cáo kết quả.

**Quy ước lưu trữ**

| Nhánh | Đường dẫn | Nội dung | Commit Git |
|---|---|---|---|
| `artifacts/` | `artifacts/<RUN_ID>/` | `config.json`, `history.json`, `best_model.pth`, `last_checkpoint.pth` | Có (trừ `.pth`) |
| `outputs/` | `outputs/YYYY-MM-DD/<RUN_ID>/<EVAL_ID>/` | `test_metrics.json`, ảnh PNG (150 dpi) | Có |
| `logs/` | `logs/<RUN_ID>/<SESSION_ID>/` | TensorBoard events cho từng phiên train/resume | Không |

`YYYY-MM-DD` tính theo múi giờ `Asia/Ho_Chi_Minh`; `EVAL_ID` và `SESSION_ID` là timestamp duy nhất (dạng `YYYYMMDD-HHMMSS+07:00`) nên các lần đánh giá/log trong cùng ngày không ghi đè nhau.

File `.pth` **không commit** lên Git (VGG16 native G1 ≈ 512 MiB); phát hành ngoài Git qua GitHub Release hoặc Drive. Link tải: _(bổ sung khi có)_.

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

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Thư viện chính (`requirements.txt`): `torch`, `torchvision`, `torchinfo`, `tensorboard`, `numpy`, `pandas`, `matplotlib`, `pillow`, `tqdm`, `ipykernel`, `pytest`.

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

Trên Linux / macOS, thay `.venv\Scripts\python.exe` bằng `python` (đã activate `.venv`), ví dụ `python -m pytest tests/ -m "not slow" -q`.

Marker `slow` được đăng ký trong `tests/conftest.py`: các test này tải CIFAR-10 hoặc ImageNet weights nên cần kết nối mạng. Nếu mạng/dữ liệu không sẵn sàng, một số test có thể được skip; hãy đọc phần tổng kết của pytest để phân biệt `passed`, `skipped`, `failed` và `deselected`.

Có thể chạy riêng nhóm test chậm hoặc một module:

```powershell
.venv\Scripts\python.exe -m pytest tests\ -m slow -v
.venv\Scripts\python.exe -m pytest tests\test_model_builder.py -m "not slow" -q
.venv\Scripts\python.exe -m pytest tests\test_engine.py -q
```

## 📓 Mở Notebook và chạy thực nghiệm

Notebook chính là [notebooks/pretrained_models.ipynb](notebooks/pretrained_models.ipynb).

### Chạy trên máy cá nhân

1. Mở notebook trong VS Code (cần extension Jupyter) hoặc JupyterLab.
2. Chọn kernel `.venv` của repository.
3. Chuẩn bị CIFAR-10 theo cấu trúc torchvision nhận tại `data/`. Notebook hiện dùng `download=False` (hàm `create_train_val_test_dataloaders` mặc định `download=True` nếu gọi trực tiếp); pretrained weights có thể được tải khi khởi tạo model ở luồng train.

### Chạy trên Google Colab

1. Đặt dữ liệu CIFAR-10 trong `MyDrive/lab2/data/` (tên thư mục phân biệt chữ hoa/thường). Không cần chép `going_modular/` lên Drive; notebook clone code từ GitHub vào Colab runtime.
2. Mở notebook `notebooks/pretrained_models.ipynb` từ checkout GitHub trong Colab và chọn GPU nếu có.
3. Chạy cell setup để mount Drive. Notebook dùng `MyDrive/lab2` làm data/artifact root, `/content/lab2` cho code checkout và `/content/cifar10_data` làm bản dữ liệu runtime.
4. Sau khi cấu hình `RUN_ID`/`RUN_IDS` và kiểm tra setup, chạy tuần tự các cell data → model → train/reuse → so sánh → đánh giá để thực thi (các) run đã chọn.

### Chọn run và hành động

Notebook hỗ trợ chọn **1 hoặc nhiều run chạy tuần tự** qua `RUN_ID` (đơn) hoặc `RUN_IDS` (list hoặc chuỗi phân tách dấu phẩy), và **người dùng tự chọn hành động** cho từng run qua `RUN_PLAN` / `DEFAULT_RUN_ACTION` (code không tự suy luận trạng thái):

| Action | Ý nghĩa | Điều kiện / hành vi |
|---|---|---|
| `train` | Huấn luyện đầy đủ từ đầu | Nếu artifacts đã tồn tại, không tự xóa; chỉ train lại khi `FORCE_RETRAIN=True` (xóa artifacts cũ của run rồi train mới). |
| `resume` | Tiếp tục khi dang dở (phù hợp Colab ngắt kết nối) | Cần `last_checkpoint.pth` và `done_epochs < target_epochs`; gọi `engine.train(resume=True)`. |
| `reuse` | Nạp artifacts có sẵn, không train | Cần đủ `config.json`, `history.json`, `best_model.pth`. Thiếu riêng `test_metrics.json` thì đánh giá Test một lần rồi lưu vào `outputs/`. Không tự chuyển sang `train` khi lỗi. |

Ví dụ cấu hình:

```python
RUN_IDS = ["v0_vgg16", "v1_resnet18"]
RUN_PLAN = {"v0_vgg16": "reuse", "v1_resnet18": "resume"}
DEFAULT_RUN_ACTION = "train"
FORCE_RETRAIN = False
```

Lưu ý quan trọng:

- `v1_resnet18_finetune` phải chạy **sau** `v1_resnet18` (hoặc khi baseline đã có `best_model.pth`). Fine-tune dựng model với `weights=None`, nạp `v1_resnet18/best_model.pth` rồi mở khóa `layer4`, dùng optimizer 2 nhóm LR (head 1e-3, backbone 1e-4). Khi `resume`, fine-tune nạp checkpoint của **chính run fine-tune**, không nạp lại baseline.
- Chọn best checkpoint bằng **validation**; test set chỉ dùng cho đánh giá cuối.
- Do chuyển sang state_dict native (hướng B), checkpoint/artifacts cũ không dùng được; cần retrain 3 run hiện hành khi chạy format mới. `test_metrics.json`/outputs cũ thuộc kết quả của format cũ.

### Giám sát bằng TensorBoard

Trong quá trình huấn luyện, `Loss/train`, `Loss/val`, `Accuracy/train`, `Accuracy/val` và learning rate từng param group (`LR/group_<i>`) được ghi log theo từng run/phiên (`logs/<RUN_ID>/<SESSION_ID>/`). Để mở giao diện trực quan:

```bash
# Trong terminal (đã kích hoạt .venv), trỏ đến thư mục log:
tensorboard --logdir=logs
```

Mở trình duyệt truy cập: [http://localhost:6006](http://localhost:6006)

---

## 🧩 Package `going_modular`

Import theo module (cách dùng trong notebook):

```python
from going_modular import data_setup, engine, evaluate, model_builder, utils
```

Hoặc flat import 12 symbol theo hợp đồng spec (`docs/define&designt/00_lab2_description.md`):

```python
from going_modular import (
    ModelConfig, build_model, count_params, set_train_mode, get_param_groups,
    create_train_val_test_dataloaders, train, evaluate_step,
    set_seed, get_device, run_inference, evaluate_predictions,
)
```

Các hàm khác (`create_writer`, `save_checkpoint`, `make_session_id`, `plot_confusion_matrix`, ...) vẫn import được trực tiếp từ `going_modular` hoặc qua module tương ứng.

---

## ⚙️ Quy ước Tiền xử lý Dữ liệu (Transforms)

Tiền xử lý được lấy trực tiếp từ weights tương ứng trong torchvision: `VGG16_Weights.IMAGENET1K_V1.transforms()` cho VGG16 và `ResNet18_Weights.IMAGENET1K_V1.transforms()` cho ResNet-18. Với hai weights hiện đang dùng, cấu hình là:

| Thành phần | Thông số | Mô tả |
|---|---|---|
| **Resize** | Cạnh ngắn lên `256` | Phóng to ảnh, giữ nguyên tỷ lệ khung hình |
| **Center crop** | `(224, 224)` | Cắt vùng giữa ảnh theo pipeline của weights |
| **ToTensor** | Chuyển đổi | Chuyển pixel về tensor và dải `[0.0, 1.0]` |
| **Normalize** | Mean/std lấy từ weights | Chuẩn hóa theo cấu hình ImageNet-1K đi kèm weights |

`get_pretrained_transform(model_name="vgg16")` và `get_pretrained_transform(model_name="resnet18")` chọn preset riêng cho từng kiến trúc. `create_train_val_test_dataloaders(transform_model=..., transform_weights=...)` dùng lựa chọn này nếu không truyền `train_transform`/`eval_transform` tùy chỉnh. Hai preset hiện tại có cùng cấu hình, nên có thể dùng chung DataLoader cho cả hai model.

Chia dữ liệu: `val_size=5000` lấy từ tập train 50.000 ảnh bằng **stratified split** (tái lập được theo `seed=42`); test set 10.000 ảnh giữ nguyên. Khi bật augmentation cho tập train, `RandomHorizontalFlip` được áp dụng trước pipeline của weights; validation và test không augmentation.

> Dù model dựng với `weights=None` (fine-tune, `reuse`), ảnh vẫn phải được tiền xử lý nhất quán với cách model đã được huấn luyện, nên transform luôn lấy từ preset ImageNet theo kiến trúc (tham số `transform_model`/`transform_weights`), không phụ thuộc `ModelConfig.weights`.

---

## 🛠️ Xử lý Sự cố Thường gặp

| Hiện tượng | Nguyên nhân | Hướng khắc phục |
|---|---|---|
| `No module named 'going_modular'` | Kernel không nhận đúng thư mục cha | Section 0 trong notebook đã có cell tự động thêm `ROOT` vào `sys.path`. Đảm bảo đã chạy Section 0. |
| Lỗi "Checkpoint/state_dict không tương thích ... `backbone.*`/`head.*`" | Artifacts được lưu bằng format wrapper cũ | Đặt `FORCE_RETRAIN=True` hoặc dọn `artifacts/<RUN_ID>`, rồi train lại theo thứ tự `v0_vgg16` → `v1_resnet18` → `v1_resnet18_finetune`. |
| Tải Pretrained Weights rất chậm | Trọng số (đặc biệt VGG16, checkpoint native G1 ~512 MB) lớn | Chỉ tải một lần (cache tại `~/.cache/torch/hub/checkpoints`). Hãy đợi tải xong. |
| `CUDA out of memory` | Bộ nhớ GPU không đủ khi xử lý batch 224×224 | Giảm `BATCH_SIZE` xuống `16` trong cell cấu hình ở Section 0. |
| Cảnh báo/khác biệt BatchNorm khi backbone bị freeze | `model.train()` mặc định cập nhật running stats của BN | `set_train_mode(model, cfg)` (trong `model_builder.py`) giữ `eval()` cho BatchNorm của phần đóng băng khi `ModelConfig.freeze_bn_stats=True` (mặc định). Khi fine-tune `layer4`, BN của `layer4` ở chế độ train; `bn1`, `layer1`–`layer3` ở `eval`. VGG16 không có BN. |
| Fine-tune báo thiếu baseline | Chưa có `artifacts/v1_resnet18/best_model.pth` | Chạy (hoặc `reuse`) `v1_resnet18` trước `v1_resnet18_finetune`. |