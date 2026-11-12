# Mô tả chi tiết phần Model — Lab 2

> **Mục đích:** Tài liệu này mô tả **phần mô hình** của bài lab ở mức khái niệm/thiết kế, đặt trước phần code. Đọc file này trước khi đi vào `going_modular/model_builder.py` hoặc notebook.
> **Liên quan:** spec chi tiết ở `docs/define&designt/02_model.md`; mã nguồn ở `going_modular/model_builder.py`.

---

## 1. Bài toán của phần model

CIFAR-10 gồm 60.000 ảnh màu 32×32 chia 10 lớp (máy bay, ô tô, chim, mèo, nai, chó, ếch, ngựa, tàu thủy, xe tải). Ảnh rất nhỏ và số lượng không lớn so với ImageNet — huấn luyện CNN từ đầu sẽ tốn kém và dễ overfit.

**Ý tưởng:** mượn "thị giác" đã học sẵn của các mạng huấn luyện trên ImageNet (1,2 triệu ảnh, 1000 lớp), chỉ thay phần "đầu ra" cho khớp 10 lớp của CIFAR-10, rồi chọn chiến lược cập nhật trọng số:

- **Feature Extraction:** backbone (phần trích xuất đặc trưng) đóng băng hoàn toàn, chỉ train head mới.
- **Fine-Tuning:** mở khóa một số khối cuối của backbone để tinh chỉnh cho dữ liệu mới.

## 2. Hai kiến trúc được dùng

| | **VGG16** | **ResNet-18** |
|---|---|---|
| Nguồn gốc | Simonyan & Zisserman, 2014 — CNN tuần tự cổ điển | He et al., 2015 — mạng Residual (liên kết tắt) |
| Backbone | `features` gồm **5 khối conv**, mỗi khối kết thúc bằng `MaxPool2d`; sau đó `avgpool` + flatten | `conv1` + `bn1` + 4 nhóm residual `layer1`–`layer4` |
| Phần phân loại gốc | `classifier` (3 lớp Linear khổng lồ, 1000 lớp ImageNet) | `fc` (Linear 512→1000) |
| Cách thay head | `classifier = Identity`; tính `in_features` động bằng một lần forward giả (output 25088) | `fc = Identity`; `in_features` đọc trực tiếp = 512 |
| Head mới | `Linear(25088 → 10)` | `Linear(512 → 10)` |
| BatchNorm | Không có | Có (`bn1`, các `bn` trong từng layer) |
| Số khối có thể unfreeze | 5 | 4 |
| Tổng tham số (~10 lớp) | ~14,97 triệu | ~11,18 triệu |
| Tham số trainable khi Feature Extraction | ~250.890 | 5.130 |

**Tại sao chọn hai mạng này:** VGG16 đại diện cho kiến trúc tuần tự cổ điển, nhiều tham số, head lớn; ResNet-18 đại diện cho kiến trúc hiện đại có residual connection, nhẹ và hội tụ nhanh — cho phép so sánh hai thế hệ thiết kế trên cùng một bài toán.

**Trọng số:** nạp `IMAGENET1K_V1` từ `torchvision.models` — tức bộ trọng số torchvision định danh rõ, tái lập được (không dùng `DEFAULT` hay `pretrained=True` đã deprecated). Khi cần nạp lại từ checkpoint riêng (fine-tune/reuse) thì dựng với `weights=None` để không tải lại ImageNet.

## 3. Cấu trúc mô hình sau khi adapt

```
Ảnh (B, 3, 224, 224) ──► Backbone (frozen hoặc một phần trainable) ──► Head mới ──► Logits (B, 10)
```

- **Backbone:** toàn bộ mạng torchvision gốc, giữ nguyên kiến trúc và trọng số ImageNet; chỉ phần phân loại cuối bị gỡ (`fc`/`classifier` → `Identity`).
- **Head:** một `nn.Linear(in_features, 10)` duy nhất, **luôn ở trạng thái trainable** (có thể kèm `Dropout` phía trước nếu `dropout > 0`).
- **Đầu ra:** logits `(B, 10)`, chưa qua softmax — loss `CrossEntropyLoss` sẽ tự xử lý.

Cả mô hình được bọc trong class `MyModel(nn.Module)` với hai thuộc tính `self.backbone` và `self.head`, `forward` chỉ là `head(backbone(x))`. Cách bọc này giúp cấu trúc rõ ràng và `state_dict` có prefix `backbone.` / `head.` nhất quán khi lưu checkpoint.

## 4. Hai chiến lược huấn luyện

| | **Feature Extraction** | **Fine-Tuning** |
|---|---|---|
| Cấu hình | `freeze_backbone=True`, `unfreeze_last_n=0` | `freeze_backbone=True`, `unfreeze_last_n=N≥1` |
| Head mới | Trainable | Trainable |
| N khối cuối backbone | Frozen | **Trainable** |
| Phần còn lại của backbone | Frozen | Frozen |
| Learning rate | head `1e-3` | head `1e-3`, backbone `1e-4` |
| Mục đích | Học cách dùng đặc trưng có sẵn, nhanh, ít rủi ro | Điều chỉnh đặc trưng tầng cao cho CIFAR-10, kỳ vọng cao hơn nhưng dễ overfit |

> Huấn luyện **toàn bộ** mô hình cũng được hỗ trợ (`freeze_backbone=False`, `unfreeze_last_n=0`), nhưng trong danh mục thí nghiệm của bài không dùng.

**Cơ chế đóng băng/mở khóa:** đóng băng = đặt `requires_grad=False` cho mọi tham số backbone. Fine-tuning được hiện thực bằng "đóng băng toàn bộ trước, rồi mở khóa N khối cuối" — với ResNet-18, `unfreeze_last_n=1` tương ứng mở `layer4`. Việc tách riêng `freeze_backbone` và `unfreeze_last_n` giúp validation bắt lỗi cấu hình mâu thuẫn (vd: `freeze_backbone=False` mà vẫn `unfreeze_last_n>0` là vô nghĩa → báo `ValueError`).

**Tại sao LR backbone nhỏ hơn:** các tầng đã có trọng số tốt, chỉ cần tinh chỉnh nhẹ; head mới khởi tạo ngẫu nhiên nên cần LR lớn hơn để học nhanh.

## 5. BatchNorm — điểm tinh tế của Fine-Tuning

`requires_grad=False` **không** ngăn BatchNorm cập nhật `running_mean`/`running_var` khi `model.train()`. Nếu cứ train toàn bộ ở chế độ train, BN của phần backbone bị đóng băng sẽ "nhớ" thống kê của CIFAR-10 thay vì ImageNet → phá hỏng đặc trưng pre-trained.

Giải pháp: hàm `set_train_mode(model, cfg)` gọi `model.train()` rồi đưa về `eval()` mọi `BatchNorm2d` có **toàn bộ tham số bị đóng băng**. Hệ quả:

- Feature Extraction: mọi BN của ResNet-18 ở `eval` → giữ nguyên thống kê ImageNet.
- Fine-Tuning `unfreeze_last_n=1`: BN trong `layer4` (được mở khóa) vẫn train bình thường; `bn1`, `layer1`–`layer3` giữ `eval`.
- VGG16 không có BN nên hàm này không có tác dụng.

## 6. Nhóm tham số cho optimizer

`get_param_groups(model, cfg, lr_head, lr_backbone)` chia tham số **trainable** thành hai nhóm cho Adam:

- Nhóm 1: tham số của head → `lr_head = 1e-3`.
- Nhóm 2 (chỉ tạo khi có backbone trainable): tham số còn lại → `lr_backbone = 1e-4`.

Ràng buộc: không tạo nhóm rỗng, không trùng lặp, không bỏ sót tham số nào.

## 7. `ModelConfig` — cấu hình hóa mô hình

`ModelConfig` là `dataclass(frozen=True)` (bất biến), chỉ chứa thông tin **mô hình** — siêu tham số huấn luyện (lr, epochs, batch size, seed) nằm ở chỗ khác:

| Trường | Mặc định | Ý nghĩa |
|---|---|---|
| `arch` | `"resnet18"` | Khóa trong registry: `resnet18`, `vgg16` |
| `num_classes` | `10` | Số lớp đầu ra |
| `weights` | `"IMAGENET1K_V1"` | Tên bộ trọng số; `None` = khởi tạo ngẫu nhiên |
| `freeze_backbone` | `True` | Đóng băng backbone |
| `unfreeze_last_n` | `0` | Số khối cuối mở khóa (tối đa 4 với ResNet-18, 5 với VGG16) |
| `dropout` | `0.0` | Dropout trước Linear mới; hợp lệ `[0, 1)` |
| `freeze_bn_stats` | `True` | Giữ BN đã đóng băng ở `eval` |

Hỗ trợ `to_dict()`/`from_dict()` để lưu vào `artifacts/<RUN_ID>/config.json`; `from_dict` từ chối khóa lạ bằng `TypeError`.

## 8. Kiến trúc mở rộng qua registry

`ARCH_SPECS` là registry chỉ lưu **phần khác biệt** giữa các kiến trúc: cách lấy/thay head và cách chia backbone thành các "khối" để unfreeze:

- `resnet18`: khối = `[layer1, layer2, layer3, layer4]` (4 khối).
- `vgg16`: backbone `features` được tự động chia thành 5 khối theo ranh giới `MaxPool2d`.

Muốn thêm DenseNet... chỉ cần thêm một mục registry — logic chung (`build_model`, `get_param_groups`, `count_params`, `set_train_mode`) không đổi. Đây là hiện thực của yêu cầu FR-9 trong spec.

## 9. Ánh xạ ba run thực nghiệm

| Run | `arch` | `weights` | `freeze_backbone` | `unfreeze_last_n` | Vai trò |
|---|---|---|---|---|---|
| `v0_vgg16` | `vgg16` | `IMAGENET1K_V1` | `True` | `0` | Feature extraction — so sánh kiến trúc |
| `v1_resnet18` | `resnet18` | `IMAGENET1K_V1` | `True` | `0` | **Baseline** feature extraction |
| `v1_resnet18_finetune` | `resnet18` | `None` (nạp `best_model.pth` của baseline) | `True` | `1` (`layer4`) | Fine-tuning, LR backbone 1e-4 |

Lưu ý luồng fine-tune: **dựng model với `weights=None`** (không tải lại ImageNet) → mở khóa `layer4` → nạp checkpoint baseline vào `state_dict` → tạo optimizer param-groups mới → train tiếp 3 epoch. Nạp `state_dict` không ảnh hưởng `requires_grad` nên thứ tự "mở khóa trước, nạp checkpoint sau" là an toàn.

## 10. Kiểm chứng

Phần model có bộ test riêng `tests/test_model_builder.py` chạy CPU với `weights=None` (không cần mạng), gồm các nhóm:

- Shape đầu ra `(B, 10)`, head đúng `in_features`, `Dropout` đặt trước `Linear`.
- Feature extraction: chỉ head trainable, `trainable = 5.130` (ResNet-18) / `~250.890` (VGG16).
- Unfreeze N khối, `count_params` đúng `total/trainable/frozen`.
- BatchNorm: BN frozen ở `eval`, `running_mean` không đổi sau forward.
- `get_param_groups`: không nhóm rỗng, LR đúng, không trùng/sót tham số.
- Validation: mọi cấu hình sai đều `ValueError`; config round-trip dict; cùng seed → cùng trọng số.
- Nhóm `slow` (cần mạng): xác nhận trọng số ImageNet thật tải được và backbone khác khởi tạo ngẫu nhiên.

---

*Tài liệu này mô tả phần model ở mức thiết kế. Khi cần đặc tả hàm từng hàm một, xem `docs/define&designt/02_model.md`; khi cần code, xem `going_modular/model_builder.py`.*
