# Đặc tả Module Mô hình — `model_builder.py`

| Mục | Nội dung |
| :--- | :--- |
| **Name** | `model_builder` — VGG16 và ResNet-18 thích nghi cho CIFAR-10 |
| **Created date** | `[điền ngày tạo]` |
| **Updated date** | 2026-11-11 (hướng B, native torchvision) |
| **Mục tiêu** | Dựng mô hình pre-trained sẵn sàng huấn luyện, hỗ trợ Feature Extraction (đóng băng backbone) và Fine-Tuning (mở khóa N khối cuối) |
| **Bước đề bài** | Step 2, 3, 4 |
| **Phụ thuộc** | `00_lab2_description.md` |
| **Source** | Code: `going_modular/model_builder.py`. Test: `tests/test_model_builder.py`. Tài liệu: mục 10 |

---

## 1. Model architecture

Cả hai mạng được nạp từ `torchvision.models` với trọng số `IMAGENET1K_V1` (hoặc `None`), giữ nguyên backbone và **thay lớp phân loại cuối** cho 10 lớp CIFAR-10 theo kiểu native (sát `docs/yeucaugoc.md`, API `weights=...` tương đương pretrained).

| | VGG16 (G1) | ResNet-18 |
| :--- | :--- | :--- |
| Họ mạng | Tuần tự cổ điển | Residual (liên kết tắt) |
| Backbone | 5 khối conv trong `features`, `avgpool`, `classifier[0..5]` giữ nguyên | `conv1`, `bn1`, `layer1`–`layer4` |
| Head gốc → head mới | **chỉ thay** `classifier[6]` → `Linear(4096→10)` (Dropout+Linear nếu `dropout>0`) | `fc` → `Linear(512→10)` (Dropout+Linear nếu `dropout>0`) |
| Có BatchNorm | Không | Có |
| Số khối có thể unfreeze | 5 | 4 |
| Tổng tham số (10 lớp) | 134,301,514 | 11,181,642 |
| Trainable khi Feature Extraction | 40,970 | 5,130 |

```text
Ảnh (B,3,224,224) ──► torchvision model đã adapt ──► Logits (B,10)
```

**Format:** `build_model` trả về model torchvision native (`torchvision.models.vgg16`/`resnet18` đã adapt), **không dùng wrapper** `backbone/head`. `state_dict` dùng key gốc (`features.*`, `classifier.*`, `conv1/layer*/fc.*`). Checkpoint/artifacts format cũ (`backbone.*`/`head.*`) **không tương thích** và phải train lại.

## 2. Input / Output

| | Mô tả |
| :--- | :--- |
| **Input của module** | `ModelConfig` (xem mục 5.1) |
| **Input của mô hình** | Tensor `(B, 3, 224, 224)`, `float32`, đã chuẩn hóa ImageNet |
| **Output của mô hình** | Logits `(B, 10)` (chưa softmax) |
| **Output phụ** | Transform tiền xử lý, thống kê tham số, nhóm tham số cho optimizer |

## 3. Yêu cầu

| Mã | Yêu cầu |
| :--- | :--- |
| FR-1 | Nạp `vgg16`, `resnet18` với `IMAGENET1K_V1` hoặc `weights=None`. |
| FR-2 | Thay head đúng vị trí theo từng họ mạng, đầu ra `num_classes`. |
| FR-3 | Đóng băng toàn bộ backbone; head mới luôn trainable. |
| FR-4 | Mở khóa `N` khối cuối của backbone (Fine-Tuning). |
| FR-5 | Thống kê tham số `total / trainable / frozen` và in tóm tắt kiến trúc (Step 3). |
| FR-6 | Giữ BatchNorm đã đóng băng ở chế độ `eval` khi huấn luyện. |
| FR-7 | Tạo nhóm tham số với learning rate riêng cho head và backbone. |
| FR-8 | Tiền xử lý/weights nhất quán với run (thực hiện qua `data_setup`/weights transforms; `model_builder` không còn cung cấp `get_transforms`). |
| FR-9 | Thêm kiến trúc mới chỉ cần thêm một mục vào registry. |
| FR-10 | Cấu hình chuyển qua lại với `dict` để lưu vào `config.json`. |

## 4. Ràng buộc

- `num_classes = 10`.
- Ghi cụ thể `IMAGENET1K_V1`; **không** dùng `DEFAULT` (alias có thể đổi giữa các phiên bản) và **không** dùng `pretrained=True` (đã deprecated).
- Trả về model torchvision native đã adapt (`nn.Module` ResNet/VGG), **không** dùng wrapper `MyModel`; `state_dict` dùng key gốc (`fc.*`/`layer*`/`features.*`/`classifier.*`). Checkpoint format cũ `backbone.*`/`head.*` không tương thích.
- Module không train, không đọc dữ liệu, không lưu/nạp file.

## 5. Thiết kế

### 5.1. `ModelConfig`

`dataclass(frozen=True)`: bất biến, chỉ chứa thông tin mô hình (không chứa lr, epochs, batch size, seed).

| Tham số | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `arch` | `"resnet18"` | Khóa trong registry (`resnet18`, `vgg16`) |
| `num_classes` | `10` | Số lớp đầu ra |
| `weights` | `"IMAGENET1K_V1"` | Tên bộ trọng số; `None` = khởi tạo ngẫu nhiên |
| `freeze_backbone` | `True` | Đóng băng backbone trước khi thay head |
| `unfreeze_last_n` | `0` | Số khối cuối được mở khóa |
| `dropout` | `0.0` | Dropout trước `Linear` mới |
| `freeze_bn_stats` | `True` | Giữ BN đã đóng băng ở `eval` |

### 5.2. Hai chiến lược

| | Feature Extraction | Fine-Tuning |
| :--- | :--- | :--- |
| Cấu hình | `freeze_backbone=True`, `unfreeze_last_n=0` | `freeze_backbone=True`, `unfreeze_last_n=N≥1` |
| Head mới | Trainable | Trainable |
| N khối cuối | Frozen | Trainable |
| Phần còn lại | Frozen | Frozen |
| Learning rate | head `1e-3` | head `1e-3`, backbone `1e-4` |

Huấn luyện toàn bộ mô hình: `freeze_backbone=False`, `unfreeze_last_n=0`.

### 5.3. Luồng `build_model`

```text
1. validate(cfg)
2. nạp model torchvision native (weights=...)
3. nếu freeze_backbone: requires_grad=False cho toàn bộ tham số
4. thay head native: resnet18.fc / vgg16.classifier[6] (head mới luôn trainable)
5. nếu unfreeze_last_n > 0: mở khóa N khối cuối backbone
```

### 5.4. Quyết định thiết kế

| Quyết định | Lý do |
| :--- | :--- |
| Model torchvision native (không wrapper `MyModel`) | Sát `yeucaugoc.md`, `state_dict`/cấu trúc quen thuộc, dễ thuyết trình |
| VGG16 G1: chỉ thay `classifier[6]` | Giữ pretrained FC `classifier[0..5]` (4096), head 10 lớp gọn (40,970 trainable) |
| Freeze **trước**, thay head **sau** | Head mới luôn trainable; optimizer không nhận danh sách rỗng |
| `set_train_mode` thay `model.train()` | `requires_grad=False` không ngăn BN cập nhật running mean/var |
| Guard checkpoint format cũ | Từ chối `backbone.*`/`head.*` (không migrate), buộc retrain 3 run |

### 5.5. BatchNorm

`set_train_mode(model, cfg)`: gọi `model.train()`, rồi nếu `freeze_bn_stats=True` thì đưa về `eval()` mọi BN có **toàn bộ tham số bị đóng băng**.

Với Fine-Tuning `unfreeze_last_n=1`: BN của `layer4` ở train; `bn1`, `layer1`–`layer3` ở eval. VGG16 không có BN nên hàm không có tác dụng.

### 5.6. Nhóm tham số

`get_param_groups(model, cfg, lr_head, lr_backbone)`:

- Nhóm 1: tham số trainable của head, LR `lr_head`.
- Nhóm 2 (chỉ khi có backbone trainable): các tham số trainable còn lại, LR `lr_backbone`.

Không tạo nhóm rỗng, không trùng, không sót.

## 6. Giao diện hàm và ngoại lệ

| Hàm | Đầu vào | Đầu ra | Lỗi |
| :--- | :--- | :--- | :--- |
| `build_model(cfg)` | `ModelConfig` | `nn.Module` (ResNet/VGG native đã adapt) | `ValueError` nếu cấu hình sai |
| `get_param_groups(model, cfg, lr_head, lr_backbone)` | model, cfg, LR | `list[dict]` | — |
| `count_params(model)` | `nn.Module` | `{total, trainable, frozen}` | — |
| `set_train_mode(model, cfg)` | `nn.Module`, `ModelConfig` | `None` | — |
| `ModelConfig.to_dict()` / `from_dict(d)` | | `dict` / `ModelConfig` | `TypeError` nếu `d` có khóa lạ |

`build_model` báo `ValueError` khi: `arch` không có trong registry; `unfreeze_last_n < 0`; `unfreeze_last_n` lớn hơn số khối (4 với ResNet-18, 5 với VGG16); `dropout` ngoài `[0, 1)`; `freeze_backbone=False` mà `unfreeze_last_n > 0`.

## 7. Tích hợp

| Bên dùng | Cách dùng |
| :--- | :--- |
| Notebook | Tạo `ModelConfig`; lấy `get_param_groups` cho optimizer; tiền xử lý dùng `data_setup.get_pretrained_transform`/weights transforms theo run |
| `engine.py` | Nhận `set_train_mode` qua tham số `train_mode_fn` (bọc `cfg` bằng `functools.partial`) |
| `utils.py` | Lưu/nạp `cfg.to_dict()` trong `config.json` |

**Fine-Tuning và `reuse`:** dựng kiến trúc với `weights=None`, nạp checkpoint, rồi (với fine-tune) mở khóa `layer4` và tạo optimizer mới. Không tải lại ImageNet weights.

## 8. Tiêu chí chấp nhận và kiểm thử

| Mã | Kiểm tra | Kỳ vọng | Yêu cầu |
| :--- | :--- | :--- | :--- |
| T1, T2, T2b | Shape đầu ra, vị trí head, dropout | Logits `(B,10)`; head đúng `out_features`; `Dropout` trước `Linear` | FR-1, FR-2 |
| T3, T4 | Feature Extraction | Chỉ head trainable; trainable = 5,130 (ResNet-18), 40,970 (VGG16 G1 native) | FR-3, FR-5 |
| T5, T5b, T5c | Unfreeze N khối, huấn luyện toàn bộ | N khối cuối trainable; trainable tăng; `freeze_backbone=False` thì tất cả trainable | FR-4 |
| T6–T8b | Cấu hình sai | `ValueError` | FR-4 |
| T9–T10c | BatchNorm | BN frozen ở eval; BN của khối mở khóa ở train; `running_mean` không đổi sau forward | FR-6 |
| T11 | `get_param_groups` | Không nhóm rỗng, LR đúng, tham số không trùng/sót | FR-7 |
| T12–T13c | Tái lập, round-trip config | Cùng seed cùng trọng số; khóa lạ bị từ chối; config bất biến | FR-10 |
| (removed) | `get_transforms` đã loại bỏ khỏi `model_builder`; transform do `data_setup`/weights cung cấp | FR-8 (ngoài module) |
| S1, S2 | Trọng số ImageNet thật (`slow`) | Chỉ head trainable; backbone khác khởi tạo ngẫu nhiên | FR-1 |

Chạy: `pytest tests/test_model_builder.py -v -m "not slow"`. Test nhanh dùng `weights=None`, không cần mạng.

## 9. Ngoài phạm vi

- Kiến trúc ngoài VGG16 và ResNet-18 (thêm mới theo FR-9).
- Huấn luyện, đánh giá độ chính xác, lưu/nạp checkpoint.

## 10. Source

1. PyTorch / torchvision, *Models and pre-trained weights*: https://docs.pytorch.org/vision/stable/models.html
2. PyTorch Tutorials, *Transfer Learning for Computer Vision*: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
3. Daniel Bourke, *Learn PyTorch for Deep Learning*, chương 05 và 06: https://www.learnpytorch.io
4. Simonyan & Zisserman, *Very Deep Convolutional Networks* (VGG), 2014.
5. He et al., *Deep Residual Learning for Image Recognition* (ResNet), 2015.