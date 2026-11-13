# Tài Liệu Phân Tích Kiến Trúc Hệ Thống Modular PyTorch (Lab 2)
**Người thực hiện:** Võ Phạm Phương Anh  
**Học phần:** Deep Learning

---

## 1. Lý do chuyển đổi từ Notebook sang Modular Scripting
Trong thực tế phát triển các dự án Deep Learning quy mô nhóm, việc viết toàn bộ code trên một file Jupyter Notebook duy nhất thường gây xung đột khi làm việc qua GitHub. Việc tách dự án thành thư mục `going_modular/` mang lại các lợi ích lớn:
- **Tái sử dụng mã nguồn (Reusability):** Các hàm tạo dữ liệu, xây dựng mô hình và vòng lặp huấn luyện có thể gọi lại ở nhiều thử nghiệm khác nhau.
- **Dễ dàng bảo trì và gỡ lỗi (Maintainability):** Khi phát sinh lỗi ở khâu nào (Data, Model hay Loss), thành viên phụ trách chỉ cần chỉnh sửa đúng file tương ứng mà không làm hỏng code của người khác.
- **Tích hợp kiểm thử tự động (Unit Testing):** Cho phép thư mục `tests/` kiểm tra độc lập đầu vào và đầu ra của từng thành phần.

---

## 2. Sơ đồ phân rã chức năng các Module trong `going_modular/`

| Tên File | Thành phần chính | Chức năng chi tiết | Đầu ra (Return) |
| :--- | :--- | :--- | :--- |
| `data_setup.py` | `create_dataloaders()` | Tải dữ liệu CIFAR-10, áp dụng Transform và đóng gói Batch | `train_dataloader`, `test_dataloader`, `class_names` |
| `model_builder.py` | Lớp kế thừa `nn.Module` | Định nghĩa các khối tích chập (`Conv2d`, `ReLU`, `MaxPool2d`) và lớp phân loại (`Linear`) | Đối tượng mô hình PyTorch (`model`) |
| `engine.py` | `train_step()`, `test_step()`, `train()` | Thực hiện lan truyền tiến, tính hàm mất mát (Loss), cập nhật trọng số qua Optimizer | Dictionary chứa lịch sử `train_loss`, `train_acc`, `test_loss`, `test_acc` |
| `utils.py` | `save_model()`, hàm vẽ biểu đồ | Lưu trữ trọng số mô hình đã huấn luyện xuống ổ cứng dưới định dạng `.pth` hoặc `.pt` | File trọng số lưu trong thư mục `models/` |

---

## 3. Phân tích luồng hoạt động của Vòng lặp Huấn luyện (`engine.py`)

### 3.1. Giai đoạn `train_step` (Huấn luyện trên 1 Epoch)
1. Chuyển mô hình sang chế độ huấn luyện thông qua lệnh `model.train()` (kích hoạt Dropout và BatchNorm nếu có).
2. Duyệt qua từng lô dữ liệu `(X, y)` trong `train_dataloader`:
   - Chuyển Tensor `X` và `y` sang thiết bị tính toán đích (`device = "cuda"` hoặc `"cpu"`).
   - **Bước 1 - Forward pass:** Đưa `X` qua mô hình để lấy dự đoán `y_pred = model(X)`.
   - **Bước 2 - Calculate loss:** Tính sai số giữa `y_pred` và nhãn thực tế `y` bằng hàm `loss_fn` (thường dùng `nn.CrossEntropyLoss()`).
   - **Bước 3 - Optimizer zero grad:** Xóa đạo hàm tích lũy từ bước trước bằng `optimizer.zero_grad()`.
   - **Bước 4 - Loss backward:** Lan truyền ngược tính đạo hàm riêng của Loss theo từng trọng số bằng `loss.backward()`.
   - **Bước 5 - Optimizer step:** Cập nhật trọng số mô hình thông qua `optimizer.step()`.
3. Cộng dồn giá trị Loss và độ chính xác (Accuracy), sau đó chia trung bình cho tổng số Batch.

### 3.2. Giai đoạn `test_step` (Đánh giá trên tập Kiểm thử)
1. Chuyển mô hình sang chế độ đánh giá thông qua lệnh `model.eval()`.
2. Sử dụng ngữ cảnh `with torch.inference_mode():` để tắt bộ theo dõi đạo hàm (Autograd), giúp tiết kiệm bộ nhớ RAM/VRAM và tăng tốc độ tính toán.
3. Thực hiện Forward pass trên toàn bộ `test_dataloader` để ghi nhận `test_loss` và `test_acc` sau mỗi Epoch.

---

## 4. Sự phối hợp giữa `docs/`, `notebooks/` và `tests/`
- **Thư mục `docs/`:** Chứa tài liệu định hướng (`assignments.md`, `implementation.md`), giúp các thành viên nắm rõ yêu cầu từng Phase trước khi viết code.
- **Thư mục `notebooks/`:** Nơi kết nối các module trong `going_modular/` lại với nhau để chạy thực nghiệm, trực quan hóa biểu đồ Loss/Accuracy và so sánh hiệu năng giữa các kiến trúc mạng.
- **Thư mục `tests/`:** Đảm bảo chất lượng mã nguồn trước khi gộp nhánh (Merge Pull Request) vào nhánh chính `main`.