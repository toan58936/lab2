# Đặc tả Tổng quan Lab 2 — Transfer Learning với Pre-trained Models trên CIFAR-10

| | |
| :--- | :--- |
| **Phiên bản** | 0.2 (Draft) |
| **Trạng thái** | Đặc tả trước khi hiện thực |
| **Vai trò** | Nguồn duy nhất cho mục tiêu, phạm vi, cấu hình, danh mục thí nghiệm và quy ước chung. Các file `01`–`05` trỏ về đây, không định nghĩa lại. |

---

## 1. Bối cảnh và vấn đề

Huấn luyện mạng nơ-ron sâu từ đầu cho phân loại ảnh tốn nhiều dữ liệu, thời gian và tài nguyên GPU. Các mô hình đã huấn luyện trên ImageNet (VGG, ResNet...) đã học sẵn đặc trưng thị giác tổng quát, có thể tái sử dụng cho bài toán khác.

**Vấn đề:** phân loại CIFAR-10 (60.000 ảnh màu 32×32, 10 lớp) bằng cách tận dụng mô hình pre-trained, cụ thể là cần xác định:

- cách nạp và chỉnh mô hình cho đúng số lớp mới;
- nên đóng băng hay tinh chỉnh các tầng trích xuất đặc trưng;
- siêu tham số (learning rate, batch size) ảnh hưởng thế nào đến kết quả.

## 2. Mục tiêu

| Mã | Mục tiêu (đo được) |
| :--- | :--- |
| O1 | Nạp được VGG16 và ResNet-18 pre-trained từ `torchvision.models`. |
| O2 | Thay được tầng phân loại cuối và áp dụng được hai chiến lược: **Feature Extraction** (đóng băng backbone) và **Fine-Tuning** (mở khóa khối cuối). |
| O3 | So sánh được kết quả giữa hai kiến trúc, giữa hai chiến lược, và giữa các giá trị learning rate / batch size. |
| O4 | Giám sát được quá trình huấn luyện bằng TensorBoard. |
| O5 | Đánh giá được mô hình trên tập Test độc lập. |

## 3. Phạm vi

**Trong phạm vi**
- Kiến trúc: VGG16, ResNet-18.
- Dữ liệu: CIFAR-10.
- Sáu thí nghiệm ở mục 7.3.
- Quy trình 7 bước của đề bài (mục 8).

**Ngoài phạm vi**
- Grid search toàn diện, nhiều seed, kiểm định thống kê giữa các mô hình.
- Early stopping, learning-rate scheduler.
- Kiến trúc khác (DenseNet...) ngoài VGG16 và ResNet-18.

## 4. Yêu cầu

### 4.1. Yêu cầu chức năng

| Mã | Yêu cầu | Bước đề bài |
| :--- | :--- | :--- |
| FR-1 | Môi trường cài đủ `torch`, `torchvision`. | 1 |
| FR-2 | Nạp mô hình pre-trained bằng `weights=` tường minh. | 2 |
| FR-3 | In được cấu trúc, tên tầng và số tham số (tổng / trainable / frozen). | 3 |
| FR-4 | Thay head theo số lớp; hỗ trợ freeze và unfreeze N khối cuối. | 4 |
| FR-5 | Chia Train / Validation / Test; tiền xử lý ảnh khớp trọng số; tạo `DataLoader`. | 5 |
| FR-6 | Huấn luyện với loss, optimizer, chọn checkpoint tốt nhất theo Validation. | 6 |
| FR-7 | Ghi loss, accuracy, learning rate lên TensorBoard. | 6 |
| FR-8 | Đánh giá Test: accuracy, confusion matrix, accuracy theo lớp. | 7 |
| FR-9 | Hai chế độ chạy: `train` và `reuse` (nạp kết quả đã lưu). | — |

### 4.2. Yêu cầu phi chức năng

| Mã | Yêu cầu |
| :--- | :--- |
| NFR-1 | **Tái lập:** cùng seed cho cùng kết quả trên cùng môi trường. |
| NFR-2 | **Không rò rỉ dữ liệu:** Test chỉ dùng đánh giá cuối, không dùng để chọn mô hình hay siêu tham số. |
| NFR-3 | **Chạy được trên Google Colab** và máy cá nhân. |
| NFR-4 | **Kiểm thử được:** mỗi module có unit test chạy nhanh, không cần mạng hay GPU. |

## 5. Ràng buộc và giả định

- CIFAR-10 chỉ có hai tập gốc (Train 50.000, Test 10.000); Validation phải tách từ Train.
- Dùng `weights=IMAGENET1K_V1`; **không** dùng `pretrained=True` (đề bài ghi nhưng API này đã deprecated).
- Ảnh 32×32 được phóng lên 224×224 theo yêu cầu đầu vào của mô hình ImageNet.
- Giả định có GPU (ví dụ Colab T4) cho các lần chạy đầy đủ.
- File `.pth` lớn (VGG16 ≈ 512 MiB) không đưa vào Git.

## 6. Tiêu chí chấp nhận

1. Chạy trọn vẹn 7 bước của đề bài cho ít nhất một thí nghiệm.
2. Có đủ sáu thí nghiệm ở mục 7.3, mỗi thí nghiệm có kết quả Test.
3. TensorBoard hiển thị đường cong loss/accuracy của các thí nghiệm.
4. Có bảng so sánh các thí nghiệm kèm nhận xét ngắn.
5. Toàn bộ unit test nhanh đạt.
6. Chế độ `reuse` tái hiện được kết quả mà không huấn luyện lại.

---

## 7. Thiết kế tổng thể

### 7.1. Kiến trúc

Notebook điều phối; logic nằm trong package `going_modular`.

| Module | Trách nhiệm | Đặc tả chi tiết |
| :--- | :--- | :--- |
| `data_setup.py` | Chia dữ liệu, tiền xử lý, tạo DataLoader | `01_data.md` |
| `model_builder.py` | Dựng mô hình, freeze/unfreeze, thống kê tham số | `02_model.md` |
| `engine.py` | Vòng lặp train/validation, checkpoint, TensorBoard | `03_engine.md` |
| `utils.py` | Seed, device, lưu/nạp artifacts | `04_utils.md` |
| `evaluate.py` | Suy luận Test, độ đo, biểu đồ | `05_evaluation_results.md` |

Luồng: `Notebook → data_setup + model_builder → engine.train → artifacts → evaluate`.

### 7.2. Cấu hình mặc định

| Tham số | Giá trị | Ghi chú |
| :--- | :--- | :--- |
| `seed` | 42 | |
| `val_size` | 5000 | 10% của Train gốc |
| `batch_size` | 32 | |
| `epochs` (Feature Extraction) | 3 | Cấu hình thử nghiệm hiện tại (có thể tăng lên 5 theo thiết kế ban đầu) |
| `lr_head` | 1e-3 | Optimizer Adam |
| `epochs` (Fine-Tuning) | 3 | Chạy tiếp từ baseline |
| `lr_backbone` | 1e-4 | Chỉ áp dụng cho khối được mở khóa |
| `unfreeze_last_n` | 1 | ResNet-18: `layer4` |

### 7.3. Danh mục thí nghiệm

| Thí nghiệm | Kiến trúc | Chiến lược | Epochs | `lr_head` / `lr_backbone` | Batch | Khởi tạo từ | Mục đích |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `v0_vgg16` | VGG16 | Feature Extraction | 3 | 1e-3 / — | 32 | ImageNet | So sánh kiến trúc |
| `v1_resnet18` | ResNet-18 | Feature Extraction | 3 | 1e-3 / — | 32 | ImageNet | **Baseline** |
| `v1_resnet18_finetune` | ResNet-18 | Fine-Tuning | 3 | 1e-3 / 1e-4 | 32 | `v1_resnet18` | Tác động của Fine-Tuning (run fine-tune duy nhất) |

> **Ghi chú epochs:** Cấu hình hiện tại dùng 3 epoch để thử nghiệm nhanh; có thể tăng Feature Extraction lên 5 epoch (theo thiết kế ban đầu) khi chạy chính thức.

Mỗi run `v2` chỉ thay **một** yếu tố so với baseline. Đây là khảo sát có kiểm soát, không phải grid search.

### 7.4. Artifacts, outputs và logs

Mỗi thí nghiệm chia thành ba nhánh:

| Nhánh | Đường dẫn | Nội dung | Commit Git |
|-------|-----------|----------|------------|
| `artifacts/` | `artifacts/<RUN_ID>/` | Trạng thái train để resume/reuse: `config.json`, `history.json`, `best_model.pth`, `last_checkpoint.pth` | Có (trừ `.pth`) |
| `outputs/` | `outputs/YYYY-MM-DD/<RUN_ID>/<EVAL_ID>/` | Kết quả đánh giá: `test_metrics.json`, ảnh confusion matrix, per-class, dự đoán sai | Có (JSON/PNG) |
| `logs/` | `logs/<RUN_ID>/<SESSION_ID>/` | TensorBoard events cho từng phiên train/resume | Không |

`YYYY-MM-DD` là ngày theo múi giờ `Asia/Ho_Chi_Minh`; `EVAL_ID` và `SESSION_ID` là timestamp duy nhất (`YYYYMMDD-HHMMSS+07:00`) để các lần đánh giá/log trong cùng ngày không ghi đè nhau.

File `.pth` phát hành ngoài Git (GitHub Release hoặc Drive); `README.md` ghi link tải.

### 7.5. Chế độ chạy (do người dùng chọn)

Người dùng tự thiết lập hành động cho từng run (ví dụ `RUN_PLAN` / `DEFAULT_RUN_ACTION` trong notebook); code **không tự suy luận** nên dùng trạng thái nào.

| Action | Ý nghĩa | Điều kiện / hành vi |
| :--- | :--- | :--- |
| `train` | Huấn luyện đầy đủ từ đầu | Nếu artifacts đã tồn tại: không tự xóa; chỉ cho phép chạy lại khi `FORCE_RETRAIN=True` (xóa artifacts cũ của run rồi train mới). Fine-tune nạp `v1_resnet18/best_model.pth` trước khi train. |
| `resume` | Tiếp tục từ trạng thái dang dở | Yêu cầu `last_checkpoint.pth`; kiểm tra `done_epochs < target_epochs` rồi gọi `engine.train(resume=True)` để nạp model/optimizer/history/epoch và chạy tiếp. Với fine-tune, nạp checkpoint của **chính run fine-tune** (không nạp baseline). |
| `reuse` | Bỏ qua huấn luyện, dùng artifacts có sẵn | Yêu cầu `config.json`, `history.json`, `best_model.pth`. Không train. Thiếu riêng `test_metrics.json` thì đánh giá Test một lần rồi lưu vào `outputs/`. |

- **Chọn run:** có thể chạy 1 run (`RUN_ID`) hoặc nhiều run tuần tự (`RUN_IDS`, list hoặc chuỗi phân tách dấu phẩy). Mỗi run dùng spec/dataloader/artifacts/output/log riêng. `v1_resnet18_finetune` là **run fine-tune duy nhất**, phải chạy sau `v1_resnet18` (hoặc baseline đã có `best_model.pth`).

### 7.6. Quy ước chung

1. Gọi `set_seed(42)` trước mỗi thí nghiệm; tạo DataLoader riêng cho từng thí nghiệm.
2. Transform lấy từ `data_setup.get_pretrained_transform(...)` (hoặc qua `create_train_val_test_dataloaders`) theo kiến trúc/weights của run; `data_setup` không import `model_builder`.
3. Fine-Tuning dựng mô hình với `weights=None`, nạp checkpoint baseline `v1_resnet18/best_model.pth` trước khi train, rồi mở khóa `layer4`, dùng optimizer mới (param groups head/backbone).
4. Khi chạy nhiều run, tạo dataloader theo spec của từng run (batch size/transform có thể khác).
5. Chỉ có một run fine-tune (`v1_resnet18_finetune`, mở `layer4`); các run khác là feature extraction.
6. Chọn checkpoint bằng Validation; không dùng Test để điều chỉnh bất cứ thứ gì.

---

## 8. Truy vết yêu cầu

| Bước đề bài | Yêu cầu | Nơi đáp ứng |
| :--- | :--- | :--- |
| 1 | FR-1 | `requirements.txt` |
| 2 | FR-2 | `build_model` (`02`) |
| 3 | FR-3 | `count_params` (`02`) |
| 4 | FR-4 | `build_model`, `unfreeze_last_n` (`02`) |
| 5 | FR-5 | `data_setup.py` (`01`) |
| 6 | FR-6, FR-7 | `engine.py` (`03`) |
| 7 | FR-8 | `evaluate.py` (`05`) |

## 9. Tài liệu liên quan

`01_data.md`, `02_model.md`, `03_engine.md`, `04_utils.md`, `05_evaluation_results.md`, `plans/implementation_plan.md`.