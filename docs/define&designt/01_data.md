# Đặc tả Module Dữ liệu — `data_setup.py`

| | |
| :--- | :--- |
| **Module** | `going_modular/data_setup.py` |
| **Bước đề bài** | Step 5 — Prepare the Data |
| **Phụ thuộc** | `00_lab2_description.md` (cấu hình chung) |
| **Phiên bản / Trạng thái** | 0.2 / Draft |

---

## 1. Mục đích và trách nhiệm

**Làm:** tạo ba `DataLoader` (Train, Validation, Test) từ CIFAR-10, sẵn sàng đưa vào mô hình pre-trained.

**Không làm:** không chứa mô hình, không có vòng lặp huấn luyện, không import `model_builder`.

## 2. Bài toán cần giải

| Vấn đề | Hệ quả nếu bỏ qua | Cách giải |
| :--- | :--- | :--- |
| Ảnh CIFAR-10 là 32×32, mô hình ImageNet cần 3×224×224 đã chuẩn hóa | Feature map suy biến, lệch phân phối trọng số pre-trained | Dùng tiền xử lý đi kèm trọng số (`weights.transforms()`) |
| CIFAR-10 chỉ có Train và Test, không có Validation | Dùng Test để chọn mô hình gây rò rỉ dữ liệu | Tách Validation từ Train gốc |
| `random_split` làm lệch tỷ lệ lớp | Validation không đại diện | Chia **phân tầng** theo nhãn |
| Train cần augmentation, Val/Test thì không | Kết quả đánh giá nhiễu, không so sánh được | Gắn transform riêng cho từng tập |

## 3. Đầu vào / Đầu ra

| | Mô tả |
| :--- | :--- |
| **Đầu vào** | `data_dir`, `batch_size`, `val_size`, `seed`, `num_workers`, `generator`, `download`, `train_transform`, `eval_transform`, `transform_model`, `transform_weights` |
| **Đầu ra** | `train_dl`, `val_dl`, `test_dl` (batch `[B, 3, 224, 224]`, `float32`) và `class_names` (10 tên lớp) |

## 4. Yêu cầu

| Mã | Yêu cầu |
| :--- | :--- |
| FR-1 | Tải CIFAR-10 Train/Test dạng ảnh PIL gốc (`transform=None`). |
| FR-2 | Chia Train gốc thành Train/Validation **phân tầng**, tái lập theo `seed`. |
| FR-3 | Tiền xử lý lấy từ trọng số pre-trained, không gõ cứng mean/std. |
| FR-4 | Augmentation (lật ngang) chỉ áp dụng cho Train; Validation và Test chỉ có biến đổi xác định. |
| FR-5 | Train và Validation dùng chung một dataset gốc, không nhân bản ảnh trên RAM. |
| FR-6 | `shuffle=True` chỉ cho Train, dùng `generator` riêng để thứ tự không bị ảnh hưởng bởi RNG toàn cục. |
| FR-7 | Đầu vào sai báo `ValueError` rõ ràng. |

## 5. Ràng buộc và giả định

- Phạm vi cam kết: CIFAR-10 cân bằng 10 lớp, `val_size=5000`. Nhãn lệch hoặc dataset khác ngoài phạm vi.
- Thuật toán chia viết bằng PyTorch thuần, không dùng scikit-learn.
- Mặc định `num_workers=0` để chạy ổn định trên Colab và Windows.
- `pin_memory = torch.cuda.is_available()`.
- Mặc định `seed=42`, `val_size=5000`, `batch_size=32` (theo `00`).

## 6. Thiết kế

### 6.1. Luồng xử lý

```text
CIFAR-10 gốc (PIL, transform=None)
   │
   ├─ Train gốc (50.000) ──► Chia phân tầng (seed) ──► train_idx (45.000) / val_idx (5.000)
   └─ Test gốc  (10.000)
                        │
                        ▼
        _TransformSubset: (dataset gốc dùng chung + chỉ mục + transform riêng)
          Train = train_idx + train_transform (có augmentation)
          Val   = val_idx   + eval_transform
          Test  = toàn bộ   + eval_transform
                        │
                        ▼
                    3 DataLoader
```

### 6.2. Chia phân tầng

Với mỗi lớp `c` có `N_c` mẫu, số mẫu Validation là `q_c = val_size × N_c / N`. Lấy phần nguyên (tối đa `N_c − 1`), phần dư chia cho các lớp có phần lẻ lớn nhất, hòa thì ưu tiên chỉ số lớp nhỏ. Xáo trộn từng lớp bằng `torch.Generator().manual_seed(seed)`.

Với CIFAR-10, `q_c = 500` là số nguyên nên không có phần dư: mỗi lớp 500 mẫu Validation và 4.500 mẫu Train.

**Lý do dùng generator riêng:** kết quả chia không phụ thuộc trạng thái RNG toàn cục.

### 6.3. Wrapper `_TransformSubset`

Lớp `Dataset` bọc dataset gốc, lấy mẫu theo `indices[i]` rồi áp `transform` ngay lúc truy xuất. Train và Validation cùng trỏ đến **một** đối tượng dataset gốc, chỉ khác `indices` và `transform`, nên không tốn gấp đôi RAM.

### 6.3. Tiền xử lý mặc định

Khi caller không truyền transform, dùng `get_pretrained_transform(augment, model_name)`:

- `augment=True`: thêm `RandomHorizontalFlip` **trước** `weights.transforms()` (lúc ảnh còn là PIL).
- `weights.transforms()` thực hiện: phóng cạnh ngắn lên 256 → `CenterCrop` 224 → tensor [0,1] → chuẩn hóa ImageNet.

Ảnh 32×32 bị phóng 8 lần nên khá mờ; cần lưu ý khi phân tích kết quả.

## 7. Giao diện hàm và ngoại lệ

| Hàm | Đầu vào | Đầu ra | Lỗi |
| :--- | :--- | :--- | :--- |
| `get_pretrained_transform(augment=False, model_name="vgg16", weights_name="IMAGENET1K_V1")` | `bool`, `"vgg16"` hoặc `"resnet18"`, tên weights | `transforms.Compose` | `ValueError` nếu `model_name`/`weights_name` không hỗ trợ |
| `get_pretrained_mean_std(model_name="vgg16", weights_name="IMAGENET1K_V1")` | tên kiến trúc, tên weights | `(mean, std)` | `ValueError` nếu không hỗ trợ |
| `stratified_split_indices(labels, val_size, seed=42)` | nhãn, `int`, `int` | `(train_idx, val_idx)` là hai `list[int]` | `ValueError` (bên dưới) |
| `create_train_val_test_dataloaders(...)` | xem mục 3 | 3 `DataLoader` và `class_names` | `ValueError` từ hàm con |

`stratified_split_indices` báo `ValueError` khi: `val_size <= 0`; `val_size >= N`; có lớp ít hơn 2 mẫu; `val_size` không chừa ít nhất 1 mẫu Train cho mỗi lớp.

Nếu `train_transform` hoặc `eval_transform` là `None`, factory tự dùng `get_pretrained_transform` tương ứng (`augment=True` cho Train, `False` cho Eval). `get_pretrained_mean_std` dùng để lấy thống kê chuẩn hóa của preset khi cần de-normalize ảnh (ví dụ hiển thị batch trong notebook).

## 8. Tích hợp

| Bên dùng | Cách phối hợp |
| :--- | :--- |
| Notebook | Gọi `set_seed`, tạo `torch.Generator` riêng, tạo loader mới cho **mỗi** run; truyền transform từ `data_setup.get_pretrained_transform(...)` và lấy `mean/std` từ `data_setup.get_pretrained_mean_std(...)`. |
| `engine.py` | Train loader để huấn luyện; Validation loader để chọn checkpoint tốt nhất. |
| `evaluate.py` | Test loader và `class_names` để suy luận và tính ma trận nhầm lẫn. |

Test loader **không** được dùng để chọn mô hình hay siêu tham số.

## 9. Tiêu chí chấp nhận và kiểm thử

| Mã | Kiểm tra | Kỳ vọng | Yêu cầu |
| :--- | :--- | :--- | :--- |
| D1 | Chia nhãn giả cân bằng 10 lớp, `val_size=5000` | Train/Val rời nhau, hợp phủ đủ 50.000; mỗi lớp 4.500/500; cùng seed cho cùng kết quả | FR-2 |
| D2 | Đổi seed | Kết quả chia đổi, số mẫu mỗi lớp giữ nguyên | FR-2 |
| D3 | `val_size` sai, lớp thiếu mẫu, đầu vào rỗng | `ValueError` | FR-7 |
| D4 | Hai `_TransformSubset` trên cùng dataset gốc | Transform đúng từng tập, dataset gốc dùng chung | FR-4, FR-5 |
| D5 | Dataset giả, lấy batch từ 3 loader | Shape `(B,3,224,224)`, `float32`; chỉ Train shuffle; Val/Test xác định | FR-4, FR-6 |
| D6 | Kiểm tra cấu hình loader | `pin_memory` khớp `torch.cuda.is_available()`; `num_workers=0` mặc định | FR-6 |
| D7 | Fallback transform | Đúng kích thước và chuẩn hóa; `model_name` lạ báo `ValueError` | FR-3 |

Chạy: `pytest tests/test_data_setup.py -v`. Test nhanh dùng dữ liệu giả, không tải CIFAR-10; test cần dữ liệu thật đánh dấu `slow`.

## 10. Ngoài phạm vi

- Dữ liệu nhãn lệch, dataset khác (ví dụ CIFAR-100).
- Hiệu năng đa tiến trình (`num_workers > 0`) và đánh giá chất lượng ảnh sau khi phóng to.