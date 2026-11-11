# Kế hoạch Triển khai Lab 2 — Pre-trained Models trên CIFAR-10

> **Căn cứ:** Đặc tả tổng thể `00` và các đặc tả module `01`–`05`.
> **Bộ đặc tả kiểm thử:** `docs/define&designt/tests/00_test_plan.md` đến `05_test_evaluate.md`.
> **Notebook điều phối:** `notebooks/pretrained_models.ipynb`.
> **Trạng thái:** Kế hoạch thực thi chính thức sau rà soát và hiệu chỉnh kiến trúc.

---

## 1. Trạng thái và nguyên tắc triển khai

Các tài liệu `00`–`05` mô tả thiết kế mục tiêu, không mặc nhiên xác nhận code hiện tại đã đáp ứng thiết kế. Các module và notebook cũ cần được đối chiếu, refactor và kiểm thử theo từng cổng bên dưới. Không thay đổi ngoài phạm vi Lab 2 trong quá trình refactor.

### Phạm vi thực nghiệm đã chốt (6 runs)

1. `v0_vgg16` — VGG16 Feature Extraction, 3 epochs (thử nghiệm; có thể tăng lên 5).
2. `v1_resnet18` — ResNet-18 Feature Extraction baseline, 3 epochs.
3. `v1_resnet18_finetune` — **run fine-tune duy nhất**: tiếp tục từ checkpoint tốt nhất của `v1_resnet18`, mở `layer4` và head, 3 epochs.

Ba runs `v2` mỗi run chỉ thay đúng một yếu tố so với `v1_resnet18`, giữ nguyên seed, dữ liệu, kiến trúc và số epochs; không bổ sung run đối chứng 8 epoch. Đây là khảo sát có kiểm soát LR và batch size theo yêu cầu lab, không phải grid search toàn diện.

### Chính sách phạm vi đối với Checkpoint và Legacy Code

* **Checkpoint cũ:** Không hỗ trợ tương thích ngược với format checkpoint cũ (do mô hình mới trả trực tiếp torchvision model, không còn thuộc tính bọc `.model`). Bộ artifacts mới sẽ được tạo sạch (clean start) theo đúng quy trình thực nghiệm của Lab. Tuy nhiên, **tuyệt đối không tự ý xóa** các tệp artifacts hiện có trong thư mục nếu chưa có sao lưu hoặc xác nhận tường minh.
* **Hàm và API cũ ngoài đặc tả:** Các hàm legacy (`accuracy_fn`, `print_train_time` trong `utils.py`, `eval_model` trong `engine.py`, các class wrapper `CIFAR10ModelV0/V1` trong `model_builder.py`) không bị xóa đột ngột. Trong quá trình triển khai, cần rà soát call sites trong toàn workspace (đặc biệt trong `pretrained_models.ipynb`), refactor thay thế ở call sites bằng API chuẩn trước khi loại bỏ hoặc đánh dấu deprecated.

---

## 2. Các quyết định thiết kế được dùng làm chuẩn

- `docs/define&designt/00_lab2_description.md` là nguồn duy nhất cho cấu hình chung, schema `config.json`, sáu runs, artifacts và các quy ước toàn hệ thống.
- Notebook chuẩn là `notebooks/pretrained_models.ipynb`.
- Notebook lấy transforms từ `model_builder.get_transforms(cfg, weights=data.transform_weights)` và truyền vào `data_setup` (Dependency Injection); `data_setup.get_pretrained_transform(...)` là fallback độc lập, không import `model_builder`.
- Tạo DataLoader riêng cho mỗi run sau khi gọi `set_seed(42)`; Train DataLoader nhận `torch.Generator` riêng để thứ tự shuffle không bị ảnh hưởng bởi thao tác ngẫu nhiên khác.
- `history` có sáu keys: `train_loss`, `val_loss`, `train_acc`, `val_acc`, `lr`, `epoch_time`.
- Mỗi run lưu `best_model.pth`, `last_checkpoint.pth`, `history.json`, `config.json`; sau đánh giá bổ sung `test_metrics.json`. Không commit checkpoint `.pth`.
- Fine-tuning có optimizer mới; LR head là `1e-3`, LR backbone mở khóa là `1e-4`; trục epoch TensorBoard tiếp tục từ epoch 5 (`start_epoch=5`).
- `deterministic=True` tăng tính tái lập trong cùng môi trường phần cứng/phần mềm, không bảo đảm kết quả giống hệt giữa các môi trường khác nhau.
- `train.epochs` là số epoch mục tiêu của phase; `start_epoch` chỉ là offset log/báo cáo. Resume mặc định tắt, chỉ tiếp tục ở ranh giới epoch với phase config khớp; RNG không được khôi phục.
- Chọn run tường minh bằng `RUN_ID` trong danh sách sáu runs; mỗi lần chạy notebook chỉ train/resume hoặc reuse run đã chọn, không mặc định tuần tự train cả sáu trong một phiên.
- Phân biệt `RUN_MODE="train"` với `RUN_MODE="reuse"`; trong chế độ train, `RESUME=True` chỉ dùng để tiếp tục run chưa hoàn thành từ `last_checkpoint.pth`. `RESUME=False` bắt đầu run mới và không được âm thầm ghi đè artifacts đã tồn tại.
- Khi chạy trên Colab, đặt thư mục artifacts của run dưới Google Drive đã mount. Ghi `last_checkpoint.pth` và history sau mỗi epoch hoàn tất; nếu lưu thất bại, phát lỗi và dừng thay vì tiếp tục một run không được bảo toàn. Resume chỉ khôi phục trạng thái tại epoch cuối đã lưu; phần epoch đang chạy khi runtime ngắt phải chạy lại.
- Fine-tune và `reuse` dựng kiến trúc với `model.weights=null`, rồi nạp checkpoint; `data.transform_weights` là cấu hình preprocessing riêng.
- `reuse` yêu cầu đủ `best_model.pth`, `history.json`, `config.json`; nếu chỉ thiếu `test_metrics.json`, chạy một lần inference Test để sinh tệp. Không tự chuyển sang train.
- `test_metrics.json` giữ `y_true`, `y_pred` và confidence tối đa theo mẫu, không lưu logits. `train_time_phase_s` và `train_time_total_s` phân biệt thời gian phase với thời gian cộng phase nguồn.
- `pin_memory=torch.cuda.is_available()` theo khả năng CUDA của runtime, không phụ thuộc thiết bị được yêu cầu cho model. Stratified split được cam kết cho cấu hình CIFAR-10 cân bằng của lab; mean/std lấy từ preprocessing preset, chỉ fallback ImageNet khi config xác nhận.
- Tập Test chỉ dùng để đánh giá cuối; chọn checkpoint bằng Validation.
- **Hợp đồng xuất API của `going_modular/__init__.py`:** Test của từng module có thể sử dụng import trực tiếp từ submodule tương ứng (`from going_modular.model_builder import ...`) trong quá trình phát triển Phase 2–6. Mỗi Phase cập nhật export tương ứng khi API sẵn sàng. Phase 7 nghiệm thu toàn diện hợp đồng export phẳng 14 symbols theo `00_lab2_description.md` mục 2.8.6.

---

## 3. Các giai đoạn triển khai chi tiết

### Phase 0 — Hoàn tất Define & Design
**Trạng thái: Hoàn tất.**
- Đồng bộ `00`–`05` theo các quyết định ở mục 2 và nhật ký `Spec_Review_Report.md`.
- Ghi rõ ba runs cốt lõi và ba runs khảo sát LR/batch size; không đưa run đối chứng epoch vào phạm vi.
- Ghi rõ mọi quyết định về epoch/resume, khởi tạo fine-tune, reuse, metrics, thiết bị dữ liệu và preprocessing trong `Spec_Review_Report.md`.
- **Cổng:** Tên notebook, schema, artifacts, transform flow, phạm vi sáu runs và các chính sách đã chốt thống nhất xuyên suốt tài liệu.

### Phase 1 — Hạ tầng kiểm thử và cấu hình dự án
- Bổ sung các thư viện bắt buộc vào `requirements.txt`: `pytest`, `numpy`, `pillow`. Giữ `torchinfo` tạm thời cho đến khi Phase 7 refactor xong call sites trong notebook.
- Đối chiếu và hoàn thiện `.gitignore`, README và cấu trúc package với `00`.
- Thiết lập thư mục `tests/`, tạo `tests/conftest.py` (marker `slow`, backend Matplotlib `Agg`, cấu hình đường dẫn import).
- Tạo smoke test thực thi tối thiểu (`tests/test_smoke.py`) để kiểm tra import môi trường và tính sẵn sàng của package, giúp lệnh thu thập và chạy test có ý nghĩa thực tế ngay từ đầu.
- **Cổng:** Môi trường được cài đặt đầy đủ; `pytest tests/test_smoke.py` và `pytest --collect-only` thành công; dữ liệu, logs và checkpoints không bị thêm nhầm vào Git.

### Phase 2 — `model_builder.py` và `test_model_builder.py`
- Hiện thực `ModelConfig` dataclass (frozen), registry `ARCH_SPECS` cho VGG16/ResNet-18, logic thay head, freeze/unfreeze (`unfreeze_last_n`), `set_train_mode` giữ BatchNorm eval, tiền xử lý `get_transforms` và `get_param_groups`.
- Bổ sung `count_params` (trả dict `{total, trainable, frozen}`) cho yêu cầu khảo sát kiến trúc Step 3 (khảo sát kiến trúc qua `torchinfo.summary` trực tiếp trong notebook).
- Kiểm tra cấu hình `weights=None` dùng được để dựng lại kiến trúc khi nạp checkpoint, đồng thời transform vẫn lấy riêng từ `data.transform_weights`.
- Triển khai bộ test `tests/test_model_builder.py` bám sát đặc tả `docs/define&designt/tests/02_test_model_builder.md`.
- Cập nhật các export liên quan đến model trong `going_modular/__init__.py`.
- **Cổng:** Toàn bộ test nhanh `T1–T14c` đạt; test cần tải ImageNet weights `S1–S2` được đánh dấu `@pytest.mark.slow`; xác nhận số tham số kỳ vọng khớp lý thuyết.

### Phase 3 — `data_setup.py` và `test_data_setup.py`
- Hiện thực stratified split (`stratified_split_indices`), `_TransformSubset`, transform fallback độc lập và factory tạo ba DataLoader.
- Bổ sung tham số `generator` cho Train DataLoader để cách ly thứ tự shuffle; đảm bảo fallback transform kích hoạt `augment=True` cho tập Train.
- Nhận transform chuẩn qua tham số Dependency Injection do Notebook cung cấp; giữ module dữ liệu độc lập với `model_builder`.
- Triển khai bộ test `tests/test_data_setup.py` bám sát đặc tả `docs/define&designt/tests/01_test_data_setup.md`.
- Cập nhật các export liên quan đến data trong `going_modular/__init__.py`.
- **Cổng:** Toàn bộ test nhanh `D1–D15` đạt; test cần tải CIFAR-10 thật `S1–S3` được đánh dấu `slow`.

### Phase 4 — `utils.py` và `test_utils.py`
- Hiện thực `set_seed` (hỗ trợ Python, NumPy, PyTorch CPU/CUDA, cờ deterministic), `get_device` (trả `torch.device`, hỗ trợ `prefer`), `get_env_info`, và `create_writer`.
- Hiện thực các hàm lưu/nạp artifacts: `save_config`, `save_history` (riêng biệt, ghi nguyên tử qua file tạm), `save_model`, `load_model` (với `weights_only=True`), `artifacts_exist`.
- Hiện thực `save_checkpoint` và `load_checkpoint` phục vụ resume tại ranh giới epoch kèm kiểm tra khớp phase config.
- Hiện thực `save_test_metrics` và `load_test_metrics`.
- Triển khai bộ test `tests/test_utils.py` bám sát đặc tả `docs/define&designt/tests/04_test_utils.md`.
- Cập nhật các export liên quan đến utils trong `going_modular/__init__.py`.
- **Cổng:** Toàn bộ test `U1–U11` đạt; cơ chế atomic write và phát hiện lỗi thiếu artifact hoạt động chính xác.

### Phase 5 — `engine.py` và `test_engine.py`
- Hiện thực `train_step`, `evaluate_step` (tính loss trung bình có trọng số theo số mẫu).
- Hiện thực vòng lặp `train` hoàn chỉnh: chọn checkpoint tốt nhất theo Validation (`val_acc` hoặc `val_loss`), `restore_best`, lưu `last_checkpoint.pth` và history nguyên tử mỗi epoch.
- Ghi nhận đầy đủ 6 keys trong `history`, đo `epoch_time` (gồm cả Train và Validation), ghi log TensorBoard theo tags chuẩn và hỗ trợ offset epoch toàn cục qua `start_epoch`.
- Nhận `train_mode_fn` thay vì phụ thuộc trực tiếp vào `ModelConfig`.
- Triển khai bộ test `tests/test_engine.py` bám sát đặc tả `docs/define&designt/tests/03_test_engine.md`.
- Cập nhật các export liên quan đến engine trong `going_modular/__init__.py`.
- **Cổng:** Toàn bộ test `E1–E12` đạt với mô hình và dữ liệu đồ chơi; cơ chế checkpoint best, resume và TensorBoard step offset được xác minh.

### Phase 6 — `evaluate.py` và `test_evaluate.py`
- **Các helper đánh giá độc lập:** Hiện thực `InferenceResult`, `run_inference` (suy luận 1 lượt), `compute_confusion_matrix` (keyword-only), các độ đo phân loại, khoảng tin cậy Wilson 95%, phân tích lỗi (`top_confusions`, `misclassified_indices`), `denormalize`, `get_mean_std` (có cờ `transform_weights`), `measure_inference_time`, `file_size_mb`, và các hàm `plot_*`. (Các helper này có thể phát triển song song).
- **Cổng tích hợp kết quả và chi phí:** Kết nối `build_cost_record` (nhận `num_params` từ `model_builder`, `history` từ `engine`, `source_history` để tính `train_time_total_s`), `evaluate_predictions` (trả đủ block metrics và block predictions: `y_true`, `y_pred`, `confidence`), và `format_comparison_table`.
- Triển khai bộ test `tests/test_evaluate.py` bám sát đặc tả `docs/define&designt/tests/05_test_evaluate.md`.
- Cập nhật các export liên quan đến evaluate trong `going_modular/__init__.py`.
- **Cổng:** Toàn bộ test `V1–V24` đạt trong môi trường không màn hình (`MPLBACKEND=Agg`); bản ghi serialize được JSON; tích hợp thành công các hợp đồng từ Phase 2, 4, 5.

### Phase 7 — Tích hợp Notebook & Hoàn thiện Package
- Rà soát và sửa các lỗi gọi hàm trong `notebooks/pretrained_models.ipynb`:
  - Sửa cell phân tích ma trận nhầm lẫn: chuyển sang gọi `compute_confusion_matrix(y_true=y_true, y_pred=y_pred, num_classes=10)` theo đúng tham số keyword-only và thứ tự nhãn.
  - Thay thế việc gọi `torchinfo.summary()` bằng `torchinfo.summary()` trực tiếp trong notebook cell.
  - Thay thế các lời gọi hàm legacy (`eval_model`, `print_train_time`) bằng pipeline đánh giá và tính chi phí chuẩn từ `engine` và `evaluate`.
- Tích hợp `RUN_MODE="train"` và `RUN_MODE="reuse"`, kết nối Google Drive / Colab nếu cần, quản lý thư mục artifacts riêng biệt cho từng run.
- Thêm lựa chọn `RUN_ID` để chạy đúng một trong sáu runs mỗi lần; trong chế độ train, thêm `RESUME` để phân biệt bắt đầu mới với tiếp tục run dở. Chế độ reuse chỉ nạp run đã hoàn thành và không được tự chuyển sang train.
- Trên Colab, mount Drive trước khi chọn artifact root; lưu checkpoint resume và history sau mỗi epoch vào thư mục run trên Drive. Không dựa vào thao tác sao lưu sau khi cả run hoàn tất mới bảo vệ checkpoint.
- Khi bắt đầu run mới, nếu artifacts của `RUN_ID` đã tồn tại thì dừng với hướng dẫn rõ ràng, không tự xóa/ghi đè. Khi resume, xác nhận checkpoint tồn tại và phase config khớp; nếu không khớp hoặc không đọc/ghi được thì báo lỗi, không tự khởi động lại từ đầu.
- Trong Fine-Tuning: dựng mô hình tương ứng từ `best_model.pth` của baseline với `weights=None`, mở khóa `layer4`, tạo optimizer mới với 2 nhóm LR và nối tiếp epoch qua `start_epoch=5`.
- Kiểm tra toàn diện hợp đồng export phẳng của package `going_modular/__init__.py` đủ 14 symbols theo `00` mục 2.8.6.
- Sau khi notebook không còn call site nào dùng `torchinfo`, tiến hành gỡ bỏ `torchinfo` khỏi `requirements.txt`.
- Thử nghiệm pipeline ngắn (smoke test notebook) trên tập dữ liệu nhỏ trước khi chạy chính thức.
- Mô phỏng ngắt sau một epoch đã lưu, khởi động lại notebook và xác minh cùng `RUN_ID` tiếp tục từ epoch kế tiếp; các run đã hoàn tất không bị train lại.
- **Cổng:** Notebook chạy trơn tru từ đầu đến cuối cho một run được chọn; chế độ `reuse` nạp đúng kiến trúc, history và metrics/dự đoán mà không huấn luyện lại; resume phục hồi trạng thái đã lưu và không ghi đè artifacts run khác; không còn lỗi runtime hoặc mismatch API.

### Phase 8 — Huấn luyện và đánh giá đầy đủ (6 runs)

| Thứ tự | Run | Phụ thuộc | Ghi chú cấu hình |
| :--- | :--- | :--- | :--- |
| 1 | `v0_vgg16` | Không | VGG16, Feature Extraction, 3 epochs (thử nghiệm), batch 32, LR head 1e-3 |
| 2 | `v1_resnet18` | Không | ResNet-18, Feature Extraction, 3 epochs (thử nghiệm), batch 32, LR head 1e-3 (baseline) |
| 3 | `v1_resnet18_finetune` | `best_model.pth` của `v1_resnet18` | ResNet-18, mở `layer4` (**run fine-tune duy nhất**), 3 epochs, batch 32, LR head 1e-3, LR backbone 1e-4 |

- Thực thi trên môi trường tăng tốc GPU (Colab / GPU local); ghi nhận thiết bị và phiên bản thư viện vào `config.json`.
- Mỗi run có thể thực thi ở phiên Colab riêng; không yêu cầu hoàn tất cả run trong một phiên. Trên Colab, giữ thư mục artifacts trong Drive và cập nhật checkpoint/history sau mỗi epoch hoàn tất.
- Chỉ đánh giá Test sau khi phase huấn luyện của run đã hoàn tất; resume run dở không đánh giá Test và không tạo `test_metrics.json` cho kết quả chưa hoàn chỉnh.
- Đánh giá Test đúng một lượt sau khi hoàn tất huấn luyện từng mô hình; tuyệt đối không dùng kết quả Test để điều chỉnh cấu hình.
- **Cổng:** Mỗi run đã hoàn tất có đủ `best_model.pth`, `last_checkpoint.pth`, `history.json`, `config.json` và `test_metrics.json`; run bị ngắt có checkpoint/history tại epoch cuối đã lưu và resume được; các run xuất hiện trong bảng so sánh tổng hợp.

### Phase 9 — Báo cáo kết quả và nghiệm thu
- Tạo bảng so sánh Markdown tổng hợp và các biểu đồ theo đặc tả `05_evaluation_results.md` (lưu tại `artifacts/figures/`).
- Diễn giải kết quả phân loại: accuracy đi kèm Wilson 95% CI, đánh giá trade-off giữa độ chính xác và chi phí (thời gian huấn luyện, latency, số tham số, VRAM, kích thước file).
- Báo cáo rõ ràng phạm vi khảo sát siêu tham số có kiểm soát (LR và batch size trên ResNet-18); không tuyên bố grid search toàn diện.
- Cập nhật `README.md` hướng dẫn chi tiết cách tải weights, điều hướng tài liệu và tái lập thực nghiệm.

---

## 4. Quan hệ phụ thuộc và Luồng thực thi

```text
Phase 1 (Hạ tầng kiểm thử & Config)
   │
   ├──► Phase 2 (model_builder) ──┐
   ├──► Phase 3 (data_setup)      │
   ├──► Phase 4 (utils) ──────────┼──► Phase 5 (engine) ──┐
   │                              │                       │
   └──► Phase 6A (eval helpers) ──┴───────────────────────┼──► Phase 6B (eval integration)
                                                          │
                                                          ▼
                                                   Phase 7 (Tích hợp Notebook)
                                                          │
                                                          ▼
                                                   Phase 8 (Huấn luyện 6 runs)
                                                          │
                                                          ▼
                                                   Phase 9 (Báo cáo & Nghiệm thu)
```

**Nguyên tắc phụ thuộc:**
1. **Song song hóa:** Sau Phase 1, các module chức năng Phase 2 (`model_builder`), Phase 3 (`data_setup`), Phase 4 (`utils`) và các helper độc lập của Phase 6A (`evaluate`) có thể được hiện thực song song vì chúng giao tiếp qua tham số hàm (Dependency Injection) hoặc độc lập logic.
2. **Cổng hội tụ:** 
   - Phase 5 (`engine`) yêu cầu Phase 2 (cung cấp hàm train mode / kiến trúc) và Phase 4 (cung cấp cơ chế lưu checkpoint / atomic write).
   - Phase 6B (Cổng tích hợp đánh giá chi phí, format bảng và lưu test metrics) yêu cầu hợp đồng từ Phase 2 (`count_params`), Phase 4 (`save/load_test_metrics`) và Phase 5 (`history["epoch_time"]`).
3. **Cổng tích hợp hệ thống:** Phase 7 chỉ bắt đầu khi toàn bộ các unit tests của Phase 2–6 đều đạt 100%.
