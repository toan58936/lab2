# Đặc tả Thiết kế Module Tiện ích — `utils.py`

> **Tài liệu:** `docs/define&designt/04_utils.md` (Define & Design cho tiện ích và quản lý artifacts)
> **Module liên quan:** `going_modular/utils.py`
> **Phụ thuộc:** `docs/define&designt/00_lab2_description.md` (cấu hình và schema), `docs/define&designt/03_engine.md` (history/checkpoint), `docs/define&designt/05_evaluation_results.md` (test metrics)
> **Trạng thái:** Pre-implementation Design Specification.

---

## PHẦN I: DEFINE — Xác định Bài toán & Yêu cầu

### 1.1. Bối cảnh và mục tiêu

`utils.py` cung cấp các tiện ích dùng chung cho toàn pipeline. Module không xây dựng mô hình, không tạo DataLoader và không chứa vòng lặp huấn luyện.

Trách nhiệm:

* Thiết lập seed và chế độ deterministic cho Python, NumPy và PyTorch.
* Chọn thiết bị, ghi nhận môi trường thực thi.
* Tạo TensorBoard writer theo tên experiment.
* Lưu/nạp model state, history, config, kết quả đánh giá và resume checkpoint.
* Kiểm tra artifacts trước khi Notebook chạy `RUN_MODE="reuse"`.

### 1.2. Yêu cầu chức năng

| Mã | Yêu cầu |
| :--- | :--- |
| UFR-1 | `set_seed` thiết lập seed cho Python, NumPy, PyTorch CPU/CUDA và bật deterministic behavior theo tùy chọn. |
| UFR-2 | `get_device` tự chọn CUDA, MPS hoặc CPU theo thứ tự ưu tiên, đồng thời cho phép yêu cầu thiết bị cụ thể và báo lỗi nếu không khả dụng. |
| UFR-3 | `get_env_info` trả về thông tin thiết bị và phiên bản thư viện dùng để ghi vào config. |
| UFR-4 | `create_writer` tạo `SummaryWriter` tại `runs/<experiment>`. |
| UFR-5 | Lưu `state_dict`, history, config, test metrics và resume checkpoint; không serialize toàn bộ đối tượng model. |
| UFR-6 | Ghi file nguyên tử: lỗi giữa chừng không làm hỏng file đích đã tồn tại. |
| UFR-7 | Nạp model/checkpoint có `map_location`, dùng chế độ `weights_only=True` khi payload tương thích. |
| UFR-8 | `artifacts_exist` và các hàm load kiểm tra đủ artifacts cần thiết, báo lỗi cụ thể khi thiếu hoặc sai schema. |
| UFR-9 | Hỗ trợ resume gồm model state, optimizer state, epoch đã hoàn tất, best score và history. |
| UFR-10 | Các tệp JSON lưu bằng UTF-8, không mất Unicode và chỉ chứa kiểu dữ liệu JSON hợp lệ. |

### 1.3. Tiêu chí chấp nhận

1. Cùng seed tạo cùng dãy số trên cùng môi trường; tài liệu không cam kết tái lập tuyệt đối giữa phần cứng/phiên bản thư viện khác nhau.
2. Thiết bị được yêu cầu nhưng không hỗ trợ phải báo lỗi rõ ràng, không tự chuyển thiết bị.
3. Ghi lỗi không làm thay đổi file cũ và không để lại file tạm.
4. Lưu rồi nạp lại model tạo cùng đầu ra với cùng input.
5. Thiếu file trong `reuse` gây `FileNotFoundError` nêu đường dẫn/tên artifact thiếu.
6. Resume khôi phục đúng model, optimizer, epoch, best score và history.
7. Test `U1`–`U11` ở mục 2.8 đạt.

---

## PHẦN II: DESIGN — Thiết kế Chi tiết

### 2.1. Vai trò trong hệ thống

```text
Notebook ── set_seed / get_device / get_env_info / create_writer
    │
    ├── engine.train ── save_model / save_checkpoint / save_history
    ├── Notebook trước train ── save_config
    ├── RUN_MODE="reuse" ── artifacts_exist / load_experiment / load_model
    └── evaluate.py ── save_test_metrics
```

### 2.2. Seed và thiết bị

**`set_seed(seed=42, deterministic=True)`**

* Thiết lập `random.seed`, `numpy.random.seed`, `torch.manual_seed` và `torch.cuda.manual_seed_all` khi CUDA khả dụng.
* Khi deterministic bật, đặt backend flags phù hợp của PyTorch/CUDA; thuật toán không khả dụng theo deterministic mode phải được xử lý rõ ràng theo PyTorch, không tuyên bố cross-hardware determinism.
* Notebook gọi hàm trước khi dựng model, DataLoader và trước mỗi run.
* Mỗi run tạo generator riêng cho Train DataLoader; resume không khôi phục RNG và chỉ cam kết tiếp tục từ ranh giới epoch.

**`get_device(prefer=None)`**

* Không chỉ định: chọn `cuda`, rồi `mps`, cuối cùng `cpu`.
* Có `prefer`: chỉ nhận thiết bị hợp lệ và sẵn sàng; nếu không khả dụng thì báo `ValueError`.
* Trả về `torch.device`, không trả chuỗi thiết bị.
* Notebook truyền cùng thiết bị vào model, engine và utils. Theo quyết định của lab, `pin_memory` được bật khi `torch.cuda.is_available()`, kể cả khi run chọn thiết bị khác.

**`get_env_info(device)`** ghi tên thiết bị và phiên bản PyTorch/torchvision, tối thiểu đủ để diễn giải số liệu thời gian và tái tạo config.

### 2.3. TensorBoard writer

`create_writer(experiment_name, log_root="runs")` trả về `torch.utils.tensorboard.SummaryWriter` tại `runs/<experiment_name>`. `engine.py` ghi scalar; Notebook chịu trách nhiệm `close()` writer khi hoàn tất, kể cả khi có lỗi (dùng `try/finally` hoặc context quản lý tương đương).

`make_session_log_dir(log_root, run_id, session_id)` tạo thư mục `log_root/<run_id>/<session_id>/` cho mỗi phiên train/resume, giúp các phiên trong cùng ngày không ghi đè nhau. `SESSION_ID` là timestamp theo múi giờ `Asia/Ho_Chi_Minh` (`YYYYMMDD-HHMMSS+07:00`).

### 2.4. Chính sách artifacts

| File | Nội dung | Sinh bởi |
| :--- | :--- | :--- |
| `best_model.pth` | `state_dict` checkpoint tốt nhất theo Validation | `engine.py` gọi `save_model` |
| `last_checkpoint.pth` | Model/optimizer state, epoch, best score và history để resume | `engine.py` gọi `save_checkpoint` |
| `history.json` | Sáu series chuẩn theo `03_engine.md` | `engine.py` gọi `save_history` nguyên tử sau mỗi epoch và khi kết thúc |
| `config.json` | Cấu hình run theo schema duy nhất ở `00` | Notebook/orchestrator gọi `save_config` nguyên tử trước khi train |
| `test_metrics.json` | Metrics và cost record của đánh giá cuối | `evaluate.py` gọi `save_test_metrics` |

`best_model.pth`, `last_checkpoint.pth` không commit vào Git. `history.json`, `config.json`, `test_metrics.json` có thể commit theo chính sách ở `00`. Việc đánh giá chưa chạy không được tạo file metrics rỗng hay giả lập thành công. Nếu `test_metrics.json` không có trong chế độ `reuse`, Notebook chạy đánh giá Test một lần, ghi metrics rồi dùng lại ở các lần sau.

### 2.5. Ghi và nạp an toàn

1. Ghi nội dung vào file tạm cùng thư mục với file đích, flush/close, rồi thay thế nguyên tử bằng `os.replace`.
2. Nếu quá trình ghi thất bại, giữ nguyên file đích cũ, dọn file tạm cụ thể đã tạo rồi phát lại lỗi.
3. `save_model` chỉ lưu `model.state_dict()` (native torchvision, key gốc). `load_model(model, path, device, arch=None)` ánh xạ thiết bị bằng `map_location`, gọi `ensure_compatible_state_dict(...)` để từ chối format cũ (`backbone.*`/`head.*`, không migrate), rồi nạp vào model.
4. Nạp weights bằng `torch.load(..., weights_only=True)`; không hạ xuống `weights_only=False` khi gặp payload không hợp lệ.
5. Resume checkpoint chỉ chứa state dict và các kiểu dữ liệu primitive/tensor tương thích với chế độ nạp an toàn. Optimizer state lưu dạng state dict, không lưu nguyên đối tượng optimizer, model hoặc callable. Khi nạp, `ensure_compatible_state_dict` kiểm tra format; checkpoint cũ wrapper không tương thích và yêu cầu train lại.
6. JSON ghi UTF-8, `ensure_ascii=False`; validate schema history/config/test metrics ở biên load.
7. `artifacts_exist` kiểm tra đủ `best_model.pth`, `history.json`, `config.json`; `test_metrics.json` là tùy chọn cho reuse inference.
8. Resume checkpoint lưu phase config để từ chối tiếp tục nếu cấu hình run không khớp; checkpoint đánh dấu số epoch đã hoàn tất trong phase. Không lưu trạng thái RNG.

### 2.6. Giao diện hàm dự kiến và ngoại lệ

| Hàm | Đầu vào | Đầu ra | Lỗi |
| :--- | :--- | :--- | :--- |
| `set_seed(seed=42, deterministic=True)` | seed và cờ deterministic | `None` | Lỗi hợp lệ đầu vào nếu seed không hợp lệ |
| `get_device(prefer=None)` | tên thiết bị tùy chọn | `torch.device` | `ValueError` nếu không hợp lệ/không khả dụng |
| `get_env_info(device)` | `torch.device` | `dict[str, str]` | Lỗi runtime liên quan thiết bị được truyền |
| `create_writer(experiment_name, log_root="runs")` | tên run và root | `SummaryWriter` | Lỗi I/O được truyền lên |
| `save_model(model, path)` | model, đường dẫn | `Path` | Lỗi I/O được truyền lên |
| `ensure_compatible_state_dict(state_dict, arch=None, source_path=None)` | state_dict, arch tùy chọn | `None` | `ValueError` nếu phát hiện key legacy `backbone.*`/`head.*` (format cũ, không migrate) |
| `load_model(model, path, device, arch=None)` | model, đường dẫn, device, arch tùy chọn | model trên device | `FileNotFoundError`; `ValueError` nếu format checkpoint cũ/không tương thích; lỗi state dict khác được truyền lên |
| `save_experiment(artifact_dir, history, config)` | thư mục, history, config | đường dẫn file đã lưu | `ValueError` nếu schema không hợp lệ |
| `save_config(artifact_dir, config)` | thư mục và config theo schema `00` | đường dẫn `config.json` | `ValueError` nếu schema không hợp lệ |
| `save_history(artifact_dir, history)` | thư mục và sáu series history | đường dẫn `history.json` | `ValueError` nếu schema không hợp lệ |
| `load_experiment(artifact_dir)` | thư mục | `(history, config)` | `FileNotFoundError`/`ValueError` |
| `load_test_metrics(artifact_dir)` | thư mục | metrics và predictions | `FileNotFoundError`/`ValueError` |
| `save_test_metrics(artifact_dir, metrics)` | thư mục, JSON record | đường dẫn file | `ValueError` nếu không serialize được |
| `artifacts_exist(artifact_dir)` | thư mục | `bool` | Không nuốt lỗi truy cập file |
| `save_checkpoint(path, model, optimizer, phase_epochs_completed, best_score, best_epoch, history, phase_config)` | trạng thái train của phase | `Path` | Lỗi I/O/schema được truyền lên |
| `load_checkpoint(path, model, optimizer, device, expected_phase_config, arch=None)` | file, trạng thái đích, device, config mong đợi, arch tùy chọn | `phase_epochs_completed`, best score/epoch, history | `FileNotFoundError`/`ValueError` nếu payload, config sai hoặc format state_dict cũ (`backbone.*`/`head.*`) |

`save_experiment` chỉ là tiện ích ghi đồng thời config và history khi cần; luồng chuẩn không dùng hàm gộp này: Notebook gọi `save_config` trước train, engine gọi `save_history` sau từng epoch.

### 2.7. Tương tác với các module khác

| Module | Tương tác |
| :--- | :--- |
| `engine.py` | Dùng checkpoint save/load và gọi `save_history` sau mỗi epoch; không ghi config. |
| `model_builder.py` | `utils` không dựng kiến trúc; Notebook dựng đúng cấu hình (native, weights/state đúng) trước khi gọi `load_model`/`load_checkpoint` (truyền `arch` khi có). |
| `evaluate.py` | Tạo JSON record; gọi `save_test_metrics` để lưu. |
| Notebook | Gọi seed/device/environment; ghi config trước train; điều phối `train`/`reuse`; nếu thiếu metrics thì đánh giá Test một lần; đóng writer; phát lỗi khi artifacts huấn luyện thiếu. |

### 2.8. Kế hoạch kiểm thử (`tests/test_utils.py`)

| Mã | Kiểm tra | Kỳ vọng |
| :--- | :--- | :--- |
| U1 | Thiết lập cùng seed hai lần, sinh số Python/NumPy/PyTorch | Chuỗi số lặp lại trong cùng môi trường. |
| U2 | Chọn CPU, thiết bị không hợp lệ/không khả dụng | Trả `torch.device` đúng hoặc báo `ValueError`. |
| U3 | Lưu/nạp JSON có Unicode, config/history theo schema | Round-trip giữ dữ liệu; schema sai bị từ chối. |
| U4 | Lưu/nạp model state dict | Cùng kiến trúc/input cho cùng đầu ra. |
| U5 | Giả lập lỗi giữa atomic write trên tệp đã có | File đích còn nguyên; tệp tạm được dọn. |
| U6 | Kiểm tra artifacts bắt buộc và thiếu từng file | `artifacts_exist` đúng; hàm load nêu tên file thiếu. |
| U7 | Nạp JSON/checkpoint sai schema hoặc payload không an toàn | Báo lỗi cụ thể; không hạ xuống `weights_only=False`. |
| U8 | Tạo writer, ghi scalar, flush/close | Event file xuất hiện đúng thư mục. |
| U9 | Lưu/nạp checkpoint resume | Khôi phục model, optimizer, phase epochs completed, best score/epoch và history. |
| U10 | Resume với config phase không khớp | Bị từ chối rõ ràng. |
| U11 | Lưu/nạp test metrics và predictions | JSON round-trip; không tạo metrics giả khi chưa đánh giá. |

Chạy kiểm thử bằng `pytest tests/test_utils.py -v`.

### 2.9. Ngoài phạm vi

* Lưu và quản lý checkpoint bên ngoài filesystem của run (ví dụ tự động đồng bộ cloud).
* Mã hóa/định danh artifact và xác thực chữ ký.
* Đồng bộ RNG toàn trạng thái để tiếp tục đúng từng sample giữa các phiên bản phần cứng/phần mềm khác nhau.
