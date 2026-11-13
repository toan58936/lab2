# Checklist Kiểm Thử Chất Lượng & Quy Trình Làm Việc Nhóm (GitHub Workflow)
**Người thực hiện:** Võ Phạm Phương Anh  
**Dự án:** CIFAR-10 Image Classification — Lab 2 (PyTorch)

---

## 1. Checklist Kiểm thử Đơn vị (Unit Testing Checklist)

### 1.1. Kiểm thử Module Dữ liệu (`data_setup`)
- [x] Kiểm tra khả năng tự động tạo thư mục lưu trữ dữ liệu khi đường dẫn chưa tồn tại.
- [x] Xác nhận số lượng lớp trả về trong `class_names` đúng bằng `10`.
- [x] Kiểm tra kiểu dữ liệu của ảnh sau khi biến đổi là `torch.FloatTensor`.
- [x] Kiểm tra kích thước (Shape) của 1 Batch dữ liệu ảnh đạt chuẩn `[batch_size, 3, 32, 32]`.
- [x] Kiểm tra kích thước của 1 Batch nhãn (Labels) đạt chuẩn `[batch_size]`.
- [x] Đảm bảo `train_dataloader` được bật cờ xáo trộn (`shuffle=True`) và `test_dataloader` tắt cờ xáo trộn (`shuffle=False`).

### 1.2. Kiểm thử Module Mô hình (`model_builder`)
- [x] Khởi tạo mô hình thành công mà không phát sinh lỗi cú pháp hoặc lệch chiều ma trận.
- [x] Kiểm tra Forward Pass với một Tensor ngẫu nhiên `torch.randn(1, 3, 32, 32)` cho ra đúng kích thước đầu ra là `[1, 10]` (tương ứng với điểm Logits của 10 lớp CIFAR-10).
- [x] Đảm bảo toàn bộ các tham số của mô hình (`model.parameters()`) đều có thuộc tính `requires_grad=True` (hoặc khóa đúng các tầng Feature Extractor khi áp dụng Transfer Learning).

### 1.3. Kiểm thử Quá trình Huấn luyện (`engine` & `utils`)
- [x] Giá trị hàm mất mát (`loss`) là một số thực hữu hạn (không bị `NaN` hoặc `Inf`).
- [x] Độ chính xác (`accuracy`) nằm trong khoảng hợp lệ từ `0.0` đến `1.0` (hoặc `0%` đến `100%`).
- [x] File trọng số sau khi gọi hàm `save_model()` được lưu thành công và có thể tải lại bằng `load_state_dict()`.

---

## 2. Quy trình Quản lý Phiên bản trên GitHub của Nhóm

Để đảm bảo an toàn cho mã nguồn chung trên nhánh `main` và ghi nhận đầy đủ lịch sử đóng góp của từng thành viên, quy trình chuẩn được áp dụng như sau:

| Bước | Lệnh Git / Thao tác | Mục đích thực hiện |
| :-: | :--- | :--- |
| 1 | `git clone https://github.com/toan58936/lab2.git` | Tải toàn bộ kho mã nguồn từ GitHub về máy tính cá nhân |
| 2 | `git checkout -b phuong-anh` | Tạo và chuyển sang nhánh làm việc riêng biệt, tách biệt với nhánh `main` |
| 3 | Soạn thảo code / tài liệu trong `/analyst` | Thực hiện nhiệm vụ được phân công và ghi chú tài liệu phân tích |
| 4 | `git add .` hoặc `git add <tên_file>` | Đưa các file vừa thêm/sửa vào vùng chờ (Staging Area) |
| 5 | `git commit -m "Nội dung cập nhật"` | Đóng gói các thay đổi kèm thông điệp mô tả rõ ràng vào lịch sử Git |
| 6 | `git push -u origin phuong-anh` | Đẩy toàn bộ các bản Commit từ máy cá nhân lên nhánh riêng trên GitHub |
| 7 | Tạo **Pull Request** & **Merge** | Đề xuất gộp kết quả làm việc từ nhánh cá nhân vào nhánh chính `main` |

---

## 3. Tổng kết kết quả đóng góp cá nhân (Phương Anh)
- Hoàn thành việc thiết lập môi trường Git & VS Code kết nối với kho lưu trữ chung của nhóm.
- Nghiên cứu cấu trúc dự án (`docs/`, `going_modular/`, `notebooks/`, `tests/`) và tài liệu phân công nhiệm vụ.
- Xây dựng bộ 3 tài liệu phân tích chuyên sâu về Data Setup, Kiến trúc Hệ thống Modular và Checklist Kiểm thử trong thư mục `analyst/`.
- Tuân thủ nghiêm ngặt quy trình tạo nhánh, Commit theo từng hạng mục và tạo Pull Request trên GitHub.