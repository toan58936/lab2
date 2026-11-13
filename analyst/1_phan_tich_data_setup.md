# Báo Cáo Phân Tích Chi Tiết Module Data Setup (CIFAR-10)
**Người thực hiện:** Võ Phạm Phương Anh  
**Môn học:** Deep Learning - Lab 2 (PyTorch Modular)

---

## 1. Tổng quan nhiệm vụ của Module `data_setup.py`
Trong kiến trúc Modular PyTorch của dự án Lab 2, file `going_modular/data_setup.py` đóng vai trò là tầng tiếp nhận và chuẩn bị dữ liệu đầu vào (Data Pipeline). Mục tiêu cốt lõi của module này bao gồm:
- Tự động kết nối và tải bộ dữ liệu chuẩn **CIFAR-10** thông qua thư viện `torchvision.datasets`.
- Thiết lập quy trình tiền xử lý ảnh (Image Preprocessing) và tăng cường dữ liệu (Data Augmentation).
- Đóng gói các tập dữ liệu thành đối tượng `DataLoader` giúp chia nhỏ dữ liệu theo từng lô (`batch_size`) để đưa vào huấn luyện trên GPU/CPU.
- Trả về danh sách tên các lớp (`class_names`) phục vụ cho việc hiển thị và đánh giá kết quả dự đoán.

---

## 2. Đặc tả kỹ thuật bộ dữ liệu CIFAR-10
Bộ dữ liệu CIFAR-10 bao gồm 60.000 ảnh màu tự nhiên có độ phân giải thấp, được chia đều vào 10 lớp đối tượng khác nhau:

| STT | Mã lớp (Label Index) | Tên lớp tiếng Anh | Tên lớp tiếng Việt | Số lượng mẫu Train | Số lượng mẫu Test |
| :-: | :-: | :--- | :--- | :-: | :-: |
| 1 | 0 | `airplane` | Máy bay | 5.000 | 1.000 |
| 2 | 1 | `automobile` | Ô tô | 5.000 | 1.000 |
| 3 | 2 | `bird` | Chim | 5.000 | 1.000 |
| 4 | 3 | `cat` | Mèo | 5.000 | 1.000 |
| 5 | 4 | `deer` | Hươu | 5.000 | 1.000 |
| 6 | 5 | `dog` | Chó | 5.000 | 1.000 |
| 7 | 6 | `frog` | Ếch | 5.000 | 1.000 |
| 8 | 7 | `horse` | Ngựa | 5.000 | 1.000 |
| 9 | 8 | `ship` | Tàu thủy | 5.000 | 1.000 |
| 10 | 9 | `truck` | Xe tải | 5.000 | 1.000 |

### Thông số kích thước Tensor:
- **Kích thước ảnh gốc:** `32 x 32` pixels.
- **Số kênh màu (Channels):** `3` kênh (Red, Green, Blue - RGB).
- **Định dạng Tensor đầu ra sau khi qua DataLoader:** `[batch_size, 3, 32, 32]`.
- **Định dạng nhãn (Labels):** Tensor 1 chiều kích thước `[batch_size]` chứa các số nguyên từ `0` đến `9`.

---

## 3. Phân tích các tham số trong hàm `create_dataloaders`
Hàm khởi tạo DataLoader được thiết kế linh hoạt với các tham số đầu vào chuẩn hóa:

1. **`root_dir` / `data_dir` (str):**
   - Đường dẫn thư mục lưu trữ dữ liệu tải về (thường đặt tại `data/`).
   - Nếu dữ liệu đã tồn tại trên ổ cứng, PyTorch sẽ bỏ qua bước tải lại để tiết kiệm thời gian.

2. **`transform` (transforms.Compose):**
   - Quy định chuỗi các phép biến đổi áp dụng lên ảnh trước khi đưa vào mạng nơ-ron.
   - Bước bắt buộc: `transforms.ToTensor()` chuyển giá trị pixel từ miền `[0, 255]` (kiểu `uint8`) sang miền `[0.0, 1.0]` (kiểu `float32`).
   - Bước chuẩn hóa: `transforms.Normalize(mean, std)` giúp phân phối dữ liệu có trung bình bằng 0 và độ lệch chuẩn bằng 1, hỗ trợ thuật toán Gradient Descent hội tụ ổn định.

3. **`batch_size` (int):**
   - Số lượng mẫu ảnh được xử lý đồng thời trong một bước lan truyền tiến (Forward Pass) và lan truyền ngược (Backward Pass).
   - Các giá trị phổ biến: `32`, `64`, hoặc `128` tùy thuộc vào dung lượng bộ nhớ VRAM.

4. **`num_workers` (int):**
   - Số tiến trình con (subprocesses) dùng để đọc dữ liệu song song từ ổ cứng lên RAM.
   - Mặc định sử dụng `os.cpu_count()` hoặc đặt về `0` khi chạy trên môi trường Windows để tránh lỗi đa luồng.

---

## 4. Các kỹ thuật Data Augmentation đề xuất cho tập Train
Để hạn chế hiện tượng quá khớp (Overfitting) trên ảnh kích thước nhỏ 32x32 của CIFAR-10, tập huấn luyện có thể kết hợp thêm:
- **`RandomHorizontalFlip(p=0.5)`:** Lật ngang ảnh ngẫu nhiên với xác suất 50%.
- **`RandomCrop(32, padding=4)`:** Đệm viền 4 pixels và cắt ngẫu nhiên lại vùng 32x32 để mô hình học được tính bất biến vị trí.
- **`ColorJitter`:** Thay đổi nhẹ độ sáng và độ tương phản của ảnh.

*(Lưu ý: Tập Test tuyệt đối không áp dụng các phép biến đổi ngẫu nhiên mà chỉ giữ lại bước `ToTensor()` và `Normalize()` để đảm bảo tính khách quan khi đánh giá).*