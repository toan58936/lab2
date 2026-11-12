# Đặc tả Module Huấn luyện — `engine.py`

| Mục | Nội dung |
| :--- | :--- |
| **Name** | `engine` — vòng lặp huấn luyện và kiểm định |
| **Created date** | `[điền ngày tạo]` |
| **Updated date** | 2026-11-07 (v0.2) |
| **Mục tiêu** | Huấn luyện mô hình qua nhiều epoch, chọn checkpoint tốt nhất theo Validation, ghi history và log TensorBoard, hỗ trợ tiếp tục huấn luyện (resume) tại ranh giới epoch |
| **Bước đề bài** | Step 6 — Train the Model |
| **Phụ thuộc** | `00_lab2_description.md`, `04_utils.md` (lưu/nạp checkpoint và history) |
| **Source** | Code: `going_modular/engine.py`. Test: `tests/test_engine.py`. Tài liệu: mục 10 |

---

## 1. Luồng xử lý

Một epoch:

```text
train_step ──► evaluate_step ──► ghi history ──► ghi TensorBoard
                                      │
                                      ▼
                  val tốt hơn best? ──có──► lưu best_model.pth
                                      │
                                      ▼
                       lưu last_checkpoint.pth + history.json
```

Sau epoch cuối: nếu `restore_best`, nạp lại trọng số tốt nhất vào mô hình.

## 2. Input / Output

| | Mô tả |
| :--- | :--- |
| **Input** | `model`, `train_dataloader`, `val_dataloader`, `optimizer`, `loss_fn`, `epochs`, `device` |
| **Input tùy chọn** | `train_mode_fn`, `writer`, `start_epoch`, `save_path`, `monitor`, `restore_best`, `checkpoint_dir`, `resume`, `expected_phase_config` |
| **Output** | `history` (xem mục 5.1) |
| **Tác dụng phụ** | Ghi `best_model.pth`, `last_checkpoint.pth`, `history.json`, log TensorBoard (mục 5.3) |

## 3. Yêu cầu

| Mã | Yêu cầu |
| :--- | :--- |
| FR-1 | Loss và accuracy tính trung bình **theo số mẫu**, không theo số batch. |
| FR-2 | `evaluate_step` không tính gradient và không đổi trọng số. |
| FR-3 | Chọn checkpoint tốt nhất bằng Validation (`val_acc` hoặc `val_loss`), không dùng Test. |
| FR-4 | Chế độ train có thể tùy biến qua hook `train_mode_fn` (để giữ BatchNorm đã đóng băng ở `eval`). |
| FR-5 | `history` đủ 6 khóa, mỗi khóa một phần tử cho mỗi epoch hoàn tất. |
| FR-6 | Ghi TensorBoard theo tag chuẩn; `writer` là tùy chọn. |
| FR-7 | Lưu `last_checkpoint.pth` và `history.json` sau mỗi epoch; lỗi ghi phải dừng run, không bỏ qua. |
| FR-8 | Resume tiếp tục từ epoch kế tiếp khi checkpoint khớp cấu hình; không khớp thì báo lỗi, không tự chạy lại từ đầu. |
| FR-9 | Tham số sai báo `ValueError` rõ ràng. |

## 4. Ràng buộc

- Engine **không biết** `ModelConfig`; mọi hành vi riêng của mô hình đi qua tham số.
- Engine **không tạo** optimizer và không dùng tập Test.
- `epochs` là số epoch chạy trong phase hiện tại. `start_epoch` chỉ là offset để đánh số log và biểu đồ, **không** dùng để resume.
- Resume chỉ tiếp tục ở ranh giới epoch; không khôi phục trạng thái RNG, nên không đảm bảo cùng chuỗi batch như lần chạy liên tục.

## 5. Thiết kế

### 5.1. Schema `history`

| Khóa | Kiểu mỗi epoch | Ghi chú |
| :--- | :--- | :--- |
| `train_loss`, `val_loss` | `float` | Trung bình theo số mẫu |
| `train_acc`, `val_acc` | `float` | Trong `[0, 1]` |
| `lr` | `list[float]` | Một giá trị cho mỗi nhóm tham số của optimizer |
| `epoch_time` | `float` (giây) | Gồm cả Train và Validation |

### 5.2. Quy tắc chọn best

| `monitor` | Hướng | Cải thiện khi |
| :--- | :--- | :--- |
| `val_acc` (mặc định) | Tối đa | Lớn hơn **nghiêm ngặt** best hiện tại |
| `val_loss` | Tối thiểu | Nhỏ hơn **nghiêm ngặt** best hiện tại |

Bằng nhau thì giữ epoch sớm hơn.

### 5.3. Tác dụng phụ

| Thời điểm | Ghi gì |
| :--- | :--- |
| Mỗi epoch | History, log TensorBoard |
| Khi best cải thiện | `best_model.pth` (qua `utils.save_model`) |
| Mỗi epoch (khi có `checkpoint_dir`) | `last_checkpoint.pth`, `history.json` (qua `utils`) |
| Kết thúc | Nạp lại best vào model nếu `restore_best` |

Tag TensorBoard: `Loss/train`, `Loss/val`, `Accuracy/train`, `Accuracy/val`, `LR/group_<i>`. Bước ghi của epoch thứ `k` trong phase là `start_epoch + k` (với `k` từ 1). Ví dụ `epochs=2, start_epoch=5` ghi ở bước 6 và 7.

### 5.4. Resume

1. Cần `checkpoint_dir` và `expected_phase_config`.
2. Nạp `last_checkpoint.pth`: model, optimizer, số epoch đã hoàn tất, best score/epoch, history.
3. Từ chối nếu phase config không khớp, hoặc số epoch đã hoàn tất không nhỏ hơn `epochs`.
4. Tiếp tục từ epoch kế tiếp; history được nối tiếp.

### 5.5. Quyết định thiết kế

| Quyết định | Lý do |
| :--- | :--- |
| Loss nhân với kích thước batch rồi chia tổng số mẫu | Batch cuối nhỏ không bị tính nặng như batch đầy |
| Nhận `train_mode_fn` thay vì `ModelConfig` | Engine độc lập với module mô hình; `model.train()` sẽ làm BN đã đóng băng cập nhật thống kê |
| Lưu `best_model.pth` và `last_checkpoint.pth` riêng | Hai thứ khác ý nghĩa: best để đánh giá, last để tiếp tục train |
| So sánh nghiêm ngặt khi chọn best | Kết quả xác định, tránh ghi đè không cần thiết |
| Ghi LR theo từng nhóm tham số | Fine-Tuning dùng hai LR khác nhau, cần quan sát cả hai |

## 6. Giao diện hàm và ngoại lệ

| Hàm | Đầu vào chính | Đầu ra | Lỗi |
| :--- | :--- | :--- | :--- |
| `train_step(model, dataloader, loss_fn, optimizer, device, train_mode_fn=None)` | như tên | `(loss, acc)` | `ValueError` nếu loader không có mẫu |
| `evaluate_step(model, dataloader, loss_fn, device)` | như tên | `(loss, acc)`, model ở `eval` | `ValueError` nếu loader không có mẫu |
| `train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs, device, ...)` | xem mục 2 | `history` | `ValueError` (bên dưới) |

`train` báo `ValueError` khi: `epochs <= 0`; `start_epoch < 0`; `monitor` không phải `val_acc` hoặc `val_loss`; `resume=True` mà thiếu `checkpoint_dir`; checkpoint đã hoàn tất đủ `epochs`. Checkpoint có phase config không khớp hoặc state_dict format cũ (`backbone.*`/`head.*`, hướng B native không tương thích) bị từ chối bởi `utils.load_checkpoint`/guard.

Lưu ý thứ tự tham số: `optimizer` đứng trước `loss_fn`.

## 7. Tích hợp

| Bên dùng | Cách phối hợp |
| :--- | :--- |
| Notebook | Dựng model, optimizer (từ `get_param_groups`), loader, writer; truyền `train_mode_fn=partial(set_train_mode, cfg=cfg)`; đóng writer sau khi xong |
| `model_builder.py` | Cung cấp `set_train_mode` và nhóm tham số; engine không import module này |
| `utils.py` | Engine gọi `save_model`, `save_checkpoint`, `save_history`; resume gọi `load_checkpoint(..., arch=expected_phase_config.get("arch"))` để khôi phục và từ chối checkpoint format cũ |
| `evaluate.py` | Không dùng; engine chỉ làm việc với Train và Validation |

## 8. Tiêu chí chấp nhận

| Tiêu chí | Yêu cầu | Test |
| :--- | :--- | :--- |
| Loss không lệch do batch cuối nhỏ | FR-1 | E3 |
| Chỉ tham số trainable thay đổi sau `train_step` | FR-1 | E2 |
| `evaluate_step` không đổi trọng số, để model ở `eval` | FR-2 | E4 |
| Checkpoint là epoch tốt nhất theo Validation | FR-3 | E8, E9, E15 |
| `train_mode_fn` được gọi; BN đã đóng băng không đổi `running_mean` | FR-4 | E5, E6 |
| `history` đủ 6 khóa, đúng độ dài; `epoch_time` mỗi epoch một giá trị | FR-5 | E7, E16 |
| Tag và bước TensorBoard đúng; chạy được khi `writer=None` | FR-6 | E11, E12 |
| Checkpoint và history ghi nguyên tử; lỗi ghi không làm hỏng file cũ | FR-7 | E14 |
| Resume đúng epoch kế tiếp; config không khớp bị từ chối | FR-8 | E13 |
| Tham số sai báo `ValueError` | FR-9 | E10 |

Chi tiết từng test nằm trong `tests/03_test_engine.md`. Chạy: `pytest tests/test_engine.py -v`.

## 9. Ngoài phạm vi

- Early stopping, learning-rate scheduler, huấn luyện nhiều GPU.
- Khôi phục trạng thái RNG khi resume.
- Đo hiệu năng GPU và hành vi riêng của GPU/MPS.

## 10. Source

1. PyTorch Tutorials, *Optimizing Model Parameters* (vòng lặp train/validation): https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
2. PyTorch Docs, *TensorBoard `SummaryWriter`*: https://docs.pytorch.org/docs/stable/tensorboard.html
3. Daniel Bourke, *Learn PyTorch for Deep Learning*, chương 05 (`engine.py`): https://www.learnpytorch.io