# Thực hành 2 - Thực hành với các kiến trúc mạng nơ-ron đã được huấn luyện trước (pre-trained)

## Bài tập

1. Thử nghiệm với các mô hình pre-trained khác nhau (VGG, DenseNet, v.v.).
2. Điều chỉnh các siêu tham số (learning rate, batch size, v.v.).
3. Sử dụng TensorBoard để theo dõi quá trình huấn luyện.

## Mục tiêu

- Làm quen với việc tải và sử dụng các mô hình pre-trained từ `torchvision.models`.
- Hiểu cách điều chỉnh (adapt) và tinh chỉnh (fine-tune) các mô hình pre-trained cho những tác vụ cụ thể.

## Các bước thực hiện

### 1. Thiết lập môi trường

- Đảm bảo bạn đã cài đặt PyTorch và `torchvision`.
- Nếu chưa, hãy cài đặt bằng lệnh: `pip install torch torchvision`

### 2. Tải một mô hình pre-trained

- `torchvision.models` cung cấp nhiều mô hình pre-trained như ResNet, VGG, DenseNet, v.v.
- `pretrained=True` sẽ tải các trọng số đã được huấn luyện trước trên tập dữ liệu ImageNet.

### 3. Khám phá kiến trúc mô hình

- In mô hình ra để xem xét cấu trúc của nó.
- Xác định các lớp (layers) và tham số (parameters) của mô hình.

### 4. Điều chỉnh mô hình cho một tác vụ cụ thể

- Thông thường, lớp cuối cùng của mô hình pre-trained (lớp phân loại) cần được thay thế để khớp với số lượng lớp (class) trong tập dữ liệu của bạn.
- Bạn có thể đóng băng (freeze) các lớp phía trước để chúng không cập nhật trọng số trong quá trình huấn luyện lại (transfer learning).
- Hoặc bạn có thể mở băng (unfreeze) một vài lớp cuối để tinh chỉnh (fine-tune) mô hình.

### 5. Chuẩn bị dữ liệu

- Tải và chuẩn bị tập dữ liệu của bạn.
- Dùng `torchvision.transforms` để tiền xử lý dữ liệu.
- Dùng `torch.utils.data.DataLoader` để tạo các batch dữ liệu.

### 6. Huấn luyện mô hình

- Định nghĩa hàm mất mát (loss function) và bộ tối ưu (optimizer).
- Viết vòng lặp huấn luyện (training loop) để huấn luyện mô hình trên tập dữ liệu của bạn.

### 7. Đánh giá mô hình

Đánh giá hiệu năng của mô hình trên tập dữ liệu kiểm thử (test dataset).