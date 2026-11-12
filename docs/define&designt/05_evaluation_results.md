> **Cập nhật 2026-11-11 (hướng B):** model/state_dict native, run hiện hành `v0_vgg16`, `v1_resnet18`, `v1_resnet18_finetune` (3 run, artifacts/outputs/logs tách theo run). Các tham chiếu 6 thí nghiệm/v2 runs và ví dụ tham số trong tài liệu này là thiết kế/lịch sử so sánh; khi đánh giá sau retrain cần dùng kết quả của 3 run hiện hành.

# Đặc tả Thiết kế Module Đánh giá & Phân tích Kết quả — `evaluate.py`

> **Tài liệu:** `docs/define&designt/05_evaluation_results.md` (Define & Design cho bộ công cụ đánh giá và phân tích thực nghiệm)
> **Module liên quan:** `going_modular/evaluate.py`
> **Bước thực hành tương ứng:** Step 7 — Evaluate the Model (kèm đối sánh đa kiến trúc theo yêu cầu thực nghiệm)
> **Phụ thuộc:** `docs/define&designt/00_lab2_description.md` (tổng quan), `docs/define&designt/01_data.md` (`test_dataloader`, giải chuẩn hóa), `docs/define&designt/02_model.md` (`count_params`), `docs/define&designt/03_engine.md` (`history`), `docs/define&designt/04_utils.md` (artifacts)
> **Trạng thái:** Pre-implementation Design Specification (Đã chốt thiết kế chính thức).

---

## PHẦN I: DEFINE — Xác định Bài toán & Yêu cầu

### 1.1. Bối cảnh & Mục tiêu

Module `evaluate.py` đánh giá mô hình đã huấn luyện trên tập Test độc lập và biến kết quả thành các con số, biểu đồ và bảng so sánh có thể đưa vào báo cáo. Đây là nơi trả lời hai câu hỏi của đề bài: mô hình hoạt động tốt đến đâu (Step 7) và khi so sánh các họ kiến trúc (VGG16 với ResNet-18) thì mô hình nào hơn, với **chi phí** bao nhiêu.

Mục tiêu:

* Mỗi run suy luận trên toàn bộ tập Test **một lượt** để tính loss, accuracy, ma trận nhầm lẫn và dự đoán; không dùng Test để chọn checkpoint hay chỉnh hyperparameter.
* Tính các độ đo phân loại bằng PyTorch thuần (nhất quán với `01_data.md`, không dùng `scikit-learn`).
* Cho biết mức **chắc chắn thống kê** của accuracy (khoảng tin cậy) để không kết luận quá mức khi so sánh.
* Đo **chi phí**: thời gian huấn luyện, thời gian suy luận, dung lượng tệp mô hình, số tham số, bộ nhớ GPU đỉnh.
* Hỗ trợ phân tích lỗi (cặp lớp nhầm nhiều, ảnh bị dự đoán sai) và trực quan hóa kết quả.
* Tổng hợp kết quả của nhiều thí nghiệm thành một bảng so sánh.

**Sáu thí nghiệm cần so sánh:**

| Thí nghiệm | Thư mục artifacts | Mô tả |
| :--- | :--- | :--- |
| `v0_vgg16` | `artifacts/v0_vgg16/` | VGG16, Feature Extraction |
| `v1_resnet18` | `artifacts/v1_resnet18/` | ResNet-18, Feature Extraction |
| `v1_resnet18_finetune` | `artifacts/v1_resnet18_finetune/` | ResNet-18, Fine-Tuning |
| `v2_resnet18_lr1e-2` | `artifacts/v2_resnet18_lr1e-2/` | ResNet-18, Feature Extraction, head LR $10^{-2}$ |
| `v2_resnet18_lr1e-4` | `artifacts/v2_resnet18_lr1e-4/` | ResNet-18, Feature Extraction, head LR $10^{-4}$ |
| `v2_resnet18_bs64` | `artifacts/v2_resnet18_bs64/` | ResNet-18, Feature Extraction, batch size 64 |

### 1.2. Đầu vào / Đầu ra

| | Mô tả |
| :--- | :--- |
| **Đầu vào** | Mô hình đã huấn luyện (hoặc nạp lại ở chế độ `reuse`), `test_dataloader` (`shuffle=False`), `class_names`, `history` của từng thí nghiệm, đường dẫn `best_model.pth` |
| **Đầu ra** | Bản ghi kết quả Test (`dict` sẵn sàng ghi JSON), bản ghi chi phí, biểu đồ (`Figure`), bảng so sánh dạng Markdown |
| **Đầu ra phụ** | Danh sách cặp lớp nhầm nhiều nhất; chỉ mục các ảnh bị dự đoán sai; tệp ảnh trong `outputs/YYYY-MM-DD/<RUN_ID>/<EVAL_ID>/` |

### 1.3. Yêu cầu chức năng

| Mã | Yêu cầu |
| :--- | :--- |
| FR-1 | Suy luận một lượt qua `DataLoader`, giữ logits và nhãn thật; loss trung bình tính từ logits. |
| FR-2 | Dựng ma trận nhầm lẫn (hàng là nhãn thật, cột là dự đoán) và **chặn đầu vào sai** bằng `ValueError`. |
| FR-3 | Tính accuracy, precision, recall, F1 theo lớp và macro-average; chia cho $0$ cho kết quả $0$, không bao giờ `NaN`. |
| FR-4 | Tính khoảng tin cậy $95\%$ cho accuracy trên tập Test. |
| FR-5 | Đo thời gian suy luận (trung vị, có warm-up, đồng bộ GPU), bộ nhớ GPU đỉnh, dung lượng tệp mô hình, số tham số, thời gian huấn luyện. |
| FR-6 | Liệt kê các cặp lớp nhầm nhiều nhất (không liệt kê ô có số đếm $0$) và chỉ mục các ảnh bị dự đoán sai. |
| FR-7 | Giải chuẩn hóa tensor để hiển thị ảnh; mean/std lấy từ transform, không gõ cứng. |
| FR-8 | Mọi hàm vẽ **trả về `Figure`**, hỗ trợ lưu tệp và đóng figure để tránh rò bộ nhớ; chạy được khi không có màn hình. |
| FR-9 | Đường cong học của Giai đoạn 2 nối tiếp Giai đoạn 1 trên trục epoch (không chồng lên nhau). |
| FR-10 | Tổng hợp kết quả Test gồm metrics, cost và predictions (`y_true`, `y_pred`, confidence tối đa) thành JSON; `reuse` nạp nếu có, nếu thiếu thì đánh giá Test một lần rồi lưu. |
| FR-11 | Sinh bảng so sánh nhiều thí nghiệm; giá trị thiếu hiển thị bằng dấu gạch dài, không báo lỗi. |

### 1.4. Ràng buộc

* PyTorch thuần cho độ đo, không phụ thuộc `scikit-learn`; cần `matplotlib` trong `requirements.txt`.
* Tập Test chỉ dùng để **đánh giá cuối cùng**, không dùng để chọn mô hình (đã chọn bằng Validation ở `engine.py`).
* `test_dataloader` có `shuffle=False`, để vị trí dự đoán khớp chỉ mục ảnh.
* Quy ước hướng ma trận: **hàng là nhãn thật, cột là nhãn dự đoán**.
* Hàm vẽ không đặt backend matplotlib trong code thư viện; test chạy với backend `Agg`.
* Số liệu thời gian và bộ nhớ chỉ có ý nghĩa khi ghi kèm thiết bị đã đo.

### 1.5. Tiêu chí chấp nhận

1. Ma trận nhầm lẫn dựng đúng trên ví dụ tính tay; tổng mỗi hàng bằng số mẫu thật của lớp đó.
2. Không có `NaN` trong precision, recall, F1 kể cả khi một lớp không bao giờ được dự đoán.
3. Loss từ `run_inference` bằng loss tính một lần trên toàn tập.
4. Hoán đổi `y_true` và `y_pred` không thể xảy ra âm thầm (tham số keyword-only).
5. Khoảng tin cậy khớp giá trị đã biết (ví dụ $p = 0.5$, $n = 100$ cho khoảng xấp xỉ $[0.404, 0.596]$).
6. Mọi hàm vẽ trả về `Figure` và tạo được tệp ảnh.
7. Bản ghi kết quả serialize được bằng `json.dumps`.
8. Bộ test ở mục 2.11 đều đạt.

---

## PHẦN II: DESIGN — Thiết kế Chi tiết

### 2.1. Vị trí trong kiến trúc hệ thống

```text
test_dataloader (01) ─┐
mô hình đã nạp (02/03) ┼──► run_inference ──► InferenceResult (logits, y_true, loss)
                       │                              │
                       │              ┌───────────────┼──────────────────┐
                       │              ▼               ▼                  ▼
                       │   compute_confusion   evaluate_predictions   misclassified_indices
                       │   _matrix + metrics   (record JSON)          + plot_sample_predictions
                       │              │               │
                       └─► measure_inference_time ─► build_cost_record ◄── history (03), best_model.pth
                                                      │
                                   ┌──────────────────┴─────────────┐
                                   ▼                                ▼
                         test_metrics.json (utils)      format_comparison_table
                                                                    │
                                                         plot_* ──► artifacts/figures/
```

### 2.2. Thiết kế cấu hình và dữ liệu lưu trữ

**Tham số đo suy luận** (truyền khi gọi, không đặt cứng trong hàm):

| Tham số | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `warmup` | `3` | Số lượt chạy bỏ đi trước khi đo (khởi tạo CUDA/cuDNN thường làm lượt đầu chậm). |
| `repeats` | `10` | Số lượt đo; kết quả lấy **trung vị** để bớt ảnh hưởng của lượt nhiễu. |
| batch đo | một batch cố định lấy từ Test | Dùng cùng một batch cho mọi mô hình để so sánh công bằng. |

**Tệp mới:** `test_metrics.json` (bổ sung cho `best_model.pth`, `history.json`, `config.json` ở `03`). Lưu vào `outputs/YYYY-MM-DD/<RUN_ID>/<EVAL_ID>/` theo quy ước ở `00` mục 7.4.

```json
{
  "metrics": {
    "num_samples": 10000, "loss": 0.31, "accuracy": 0.8842,
    "accuracy_ci95": [0.8777, 0.8904],
    "macro_precision": 0.884, "macro_recall": 0.884, "macro_f1": 0.884,
    "per_class": { "airplane": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 1000} },
    "confusion_matrix": [[0, 0]]
  },
  "cost": {
    "num_params_total": 11181642, "num_params_trainable": 5130,
    "model_file_mb": 42.7, "ms_per_image": 3.1, "ms_per_batch": 99.2,
    "timing_batch_size": 32, "peak_gpu_mb": null, "device": "cpu",
    "train_time_phase_s": 215.0, "train_time_total_s": 215.0
  },
  "predictions": {
    "y_true": [0, 1], "y_pred": [0, 2], "confidence": [0.98, 0.61]
  }
}
```

Các con số trong ví dụ chỉ minh họa cấu trúc. `predictions` lưu class index và xác suất lớn nhất theo từng mẫu, không lưu logits. Lưu kết quả để chế độ `reuse` có thể dựng bảng và phân tích lỗi mà không inference lại.

### 2.3. Suy luận một lượt

`run_inference(model, dataloader, device, loss_fn=None) -> InferenceResult`

```text
1. model.eval(); trong torch.inference_mode():
2.   với mỗi batch: logits = model(X); lưu logits (CPU) và y
3. nối thành logits [N, C] và y_true [N]; nếu rỗng thì ValueError
4. nếu có loss_fn: loss = loss_fn(logits, y_true) trên toàn tập
5. return InferenceResult(logits, y_true, loss)
```

* `y_pred = logits.argmax(1)` và `probs() = softmax(logits)` được suy ra theo yêu cầu, không chạy lại mô hình.
* Mỗi run chỉ duyệt toàn bộ Test một lượt để tạo metrics/dự đoán; benchmark latency ở mục 2.6 là phép đo riêng trên cùng một batch cố định, không duyệt lại toàn bộ tập và không dùng nhãn để quyết định mô hình.
* Loss tính trên toàn tập bằng đúng loss trung bình có trọng số theo số mẫu mà `evaluate_step` (`03`) tính, nên hai nơi cho cùng kết quả.
* `predict_dataloader` và `predict_probs` được giữ lại để tương thích với mã đã có.
* Bộ nhớ logits: $10,000 \times 10$ số thực, không đáng kể.

### 2.4. Ma trận nhầm lẫn và độ đo

**Dựng ma trận:** `compute_confusion_matrix(*, y_true, y_pred, num_classes)`, mã hóa mỗi cặp thành chỉ số $\text{y\_true} \times C + \text{y\_pred}$ rồi `bincount` và `reshape`.

Tham số là **keyword-only** vì hoán đổi `y_true` và `y_pred` không gây lỗi mà chỉ chuyển vị ma trận, một lỗi rất khó phát hiện. Hàm báo `ValueError` khi: `num_classes < 1`; hai tensor không phải 1-D hoặc khác độ dài; có nhãn nằm ngoài $[0, C-1]$.

**Ví dụ tính tay** ($C = 3$): `y_true = [0, 0, 1, 2, 2]`, `y_pred = [0, 1, 1, 2, 0]`.

$$\text{cm} = \begin{pmatrix} 1 & 1 & 0 \\ 0 & 1 & 0 \\ 1 & 0 & 1 \end{pmatrix}$$

| Lớp | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- |
| 0 | $1/2 = 0.5$ | $1/2 = 0.5$ | $0.5$ |
| 1 | $1/2 = 0.5$ | $1/1 = 1.0$ | $0.6667$ |
| 2 | $1/1 = 1.0$ | $1/2 = 0.5$ | $0.6667$ |

Accuracy $= 3/5 = 0.6$; macro precision $= 0.6667$; macro recall $= 0.6667$; macro F1 $\approx 0.6111$.

**Các độ đo** (`classification_metrics`):

$$\text{precision}_c = \frac{TP_c}{\text{số lần dự đoán lớp } c}, \quad \text{recall}_c = \frac{TP_c}{\text{số mẫu thật của lớp } c}, \quad F1_c = \frac{2 P_c R_c}{P_c + R_c}$$

* **Quy ước chia cho $0$:** mẫu số bằng $0$ cho kết quả $0$. Ví dụ lớp không bao giờ được dự đoán có precision $0$. Quy ước này tránh `NaN` lan sang macro-average và biểu đồ.
* **Macro-average** là trung bình đơn giản trên **mọi lớp**, kể cả lớp có precision $0$ theo quy ước trên.
* `per_class_accuracy` giữ lại, về bản chất bằng recall theo lớp.
* CIFAR-10 cân bằng ($1,000$ mẫu mỗi lớp trong Test) nên accuracy và macro recall gần như trùng nhau; hai chỉ số khác nhau khi dữ liệu mất cân bằng.

### 2.5. Khoảng tin cậy và cách diễn giải so sánh

**Khoảng tin cậy Wilson** cho accuracy $p = \text{correct}/n$:

$$\text{center} = \frac{p + z^2/2n}{1 + z^2/n}, \qquad \text{half} = \frac{z\sqrt{p(1-p)/n + z^2/4n^2}}{1 + z^2/n}, \qquad z = 1.96$$

Ví dụ kiểm chứng: $p = 0.5$, $n = 100$ cho khoảng xấp xỉ $[0.4038, 0.5962]$. Hàm báo `ValueError` nếu `total <= 0` hoặc `correct` ngoài $[0, \text{total}]$.

**Độ rộng thực tế:** với $n = 10{,}000$ và accuracy khoảng $88\%$, nửa độ rộng khoảng $0.64$ điểm phần trăm.

**Nguyên tắc diễn giải khi so sánh mô hình** (ghi vào báo cáo):

1. Wilson CI mô tả bất định của accuracy riêng từng run; không dùng độ chồng lấp của CI hoặc độ rộng CI làm kiểm định hơn-kém giữa các mô hình.
2. Báo cáo accuracy Validation và Test; mỗi run có một lượt inference toàn bộ Test để đánh giá cuối. Không dùng Test để chọn checkpoint hay chỉnh hyperparameter.
3. Đặt accuracy cạnh chi phí (mục 2.6): VGG16 khoảng $134$ triệu tham số, ResNet-18 khoảng $11$ triệu.
4. Khoảng cách giữa train accuracy và validation accuracy trong `history` là dấu hiệu overfitting quan sát được.
5. Ảnh CIFAR-10 $32 \times 32$ bị phóng to $8$ lần nên khá mờ (`01_data.md`); nên bình luận ảnh hưởng này khi phân tích lớp dễ nhầm.

### 2.6. Đo chi phí

| Chỉ số | Hàm / nguồn | Cách đo | Lưu ý |
| :--- | :--- | :--- | :--- |
| Thời gian huấn luyện | `build_cost_record(history=..., source_history=None)` | `train_time_phase_s` là tổng epoch của run hiện tại; `train_time_total_s` cộng thời gian nguồn `init_from` nếu có | Fine-tune báo riêng thời gian 3 epoch của phase và tổng thời gian baseline + fine-tune; nguồn lấy từ history của baseline đã lưu. |
| Thời gian suy luận | `measure_inference_time` | Trung vị của `repeats` lượt trên một batch cố định, sau `warmup` lượt bỏ đi | Trên GPU gọi `torch.cuda.synchronize()` trước khi đọc đồng hồ, vì phép tính GPU bất đồng bộ. |
| Bộ nhớ GPU đỉnh | `measure_inference_time` | `torch.cuda.max_memory_allocated` sau `reset_peak_memory_stats` | `None` trên CPU. |
| Dung lượng tệp | `file_size_mb` | Kích thước `best_model.pth` (MiB) | `.pth` bị loại khỏi Git nên phải ghi vào JSON. |
| Số tham số | `build_cost_record(num_params=...)` | Đầu ra `count_params` của `02` | Gồm tổng và trainable. |
| Thiết bị | `measure_inference_time` | Tên GPU hoặc `cpu` | Bắt buộc ghi kèm, vì Colab có thể cấp GPU khác nhau giữa các phiên. |

**Ước tính tham khảo** (tính tay, cần đối chiếu với tệp thật): `state_dict` lưu số thực 32 bit, nên VGG16 ($134{,}301{,}514$ tham số) cỡ $512$ MiB và ResNet-18 ($11{,}181{,}642$ tham số) cỡ $43$ MiB.

### 2.7. Hình vẽ và giải chuẩn hóa

**Nguyên tắc chung:** mọi hàm `plot_*` tạo `Figure`, gọi `finalize_figure(fig, save_path, show, close=True)` rồi **trả về `Figure`**.

* Trả về `Figure` để notebook hiển thị, lưu vào `outputs/YYYY-MM-DD/<RUN_ID>/<EVAL_ID>/`, hoặc ghi lên TensorBoard bằng `add_figure`; đồng thời test kiểm tra được kết quả.
* `finalize_figure` dùng `fig.savefig` (không dùng trạng thái toàn cục của pyplot) và `plt.close(fig)` để tránh rò bộ nhớ.
* Không đặt backend trong thư viện; khi chạy trên máy không có màn hình dùng backend `Agg` (biến môi trường `MPLBACKEND=Agg`).

| Hàm | Nội dung |
| :--- | :--- |
| `plot_confusion_matrix(cm, class_names, normalize=False, ...)` | Heatmap có số trong từng ô; `normalize=True` hiển thị tỷ lệ theo hàng |
| `plot_per_class_accuracy(values, class_names, ...)` | Biểu đồ cột accuracy theo lớp, chú thích giá trị |
| `plot_sample_predictions(images, true, pred, class_names, ...)` | Lưới ảnh; tiêu đề xanh kèm `✓` nếu đúng, đỏ kèm `✗` nếu sai (không chỉ dựa vào màu) |
| `plot_history(histories, start_epochs=None, ...)` | Đường cong loss và accuracy; validation nét liền, train nét đứt |

**`start_epochs`:** truyền ví dụ `{"resnet18_finetune": 5}` để đường cong Giai đoạn 2 bắt đầu ở epoch $6$ và nối tiếp Giai đoạn 1, khớp với `start_epoch` ở `03`.

**Giải chuẩn hóa:**

$$X_{\text{vis}} = \text{clamp}\left( X_{\text{norm}} \odot \text{std} + \text{mean},\; 0.0,\; 1.0 \right)$$

* `denormalize(images, mean, std)` nhận tensor `[C, H, W]` hoặc `[N, C, H, W]`, báo `ValueError` nếu số kênh không khớp.
* `get_mean_std(transform, transform_weights=None)` ưu tiên đọc mean và std từ transform/preprocessing preset của torchvision weights.
* Chỉ fallback sang ImageNet mean/std nếu `config.data.transform_weights` xác nhận `IMAGENET1K_V1`; nếu không đọc được và config không xác nhận nguồn đó, phải yêu cầu mean/std tường minh hoặc báo `ValueError`, không tự đoán.

### 2.8. Phân tích lỗi và tổng hợp so sánh

**Phân tích lỗi:**

* `top_confusions(cm, class_names, k)` trả về các bộ `(lớp thật, lớp dự đoán, số lần)` giảm dần; **không bao giờ liệt kê ô có số đếm $0$**, nên mô hình hoàn hảo sẽ trả về danh sách rỗng thay vì dòng như `("cat", "cat", 0)`.
* `misclassified_indices(y_true, y_pred, logits=None, max_items=None)` trả về vị trí các ảnh bị đoán sai theo thứ tự tập dữ liệu; nếu truyền `logits` thì xếp theo độ tự tin giảm dần, để xem trước những lần nhầm "chắc chắn" nhất. Vị trí này ứng với ảnh thứ $i$ của tập Test **chỉ khi** `shuffle=False`.

**Tổng hợp một thí nghiệm:** `evaluate_predictions(result, class_names)` trả về bản ghi metrics và `predictions` (`y_true`, `y_pred`, confidence tối đa; không lưu logits) ở mục 2.2 (thuần số và danh sách, serialize được bằng JSON); `build_cost_record(...)` trả về bản ghi cost.

**So sánh nhiều thí nghiệm:** `format_comparison_table(runs)` nhận `runs[tên] = {"metrics", "cost", tùy chọn "best_val_acc"}` và trả về bảng Markdown:

| Cột | Ý nghĩa |
| :--- | :--- |
| Run | Tên thí nghiệm |
| Val acc | Accuracy Validation tốt nhất (từ `history`) |
| Test acc, 95% CI | Accuracy Test và khoảng tin cậy |
| Macro F1 | F1 trung bình các lớp |
| Params, Trainable | Tổng tham số và tham số huấn luyện được |
| Model (MB) | Dung lượng `best_model.pth` |
| ms/image | Thời gian suy luận trung vị mỗi ảnh |
| Train time phase (s) | Tổng thời gian huấn luyện phase hiện tại |
| Train time total (s) | Tổng thời gian phase hiện tại và phase nguồn `init_from` nếu có |

Giá trị thiếu hiển thị bằng dấu gạch dài (—).

### 2.9. Giao diện hàm và ngoại lệ

| Hàm | Đầu vào | Đầu ra | Ngoại lệ |
| :--- | :--- | :--- | :--- |
| `run_inference(model, dataloader, device, loss_fn)` | như tên | `InferenceResult` | `ValueError` nếu dataloader rỗng |
| `predict_dataloader(model, dataloader, device)` | như tên | `(y_pred, y_true)` | như trên |
| `predict_probs(model, images, device)` | `[N, C, H, W]` | `[N, num_classes]` | không |
| `compute_confusion_matrix(*, y_true, y_pred, num_classes)` | tham số keyword-only | `[C, C]` kiểu số nguyên | `ValueError` (xem mục 2.4) |
| `classification_metrics(cm)` | ma trận vuông | `dict` độ đo | `ValueError` nếu không vuông |
| `wilson_interval(correct, total, z=1.96)` | `int`, `int`, `float` | `(thấp, cao)` | `ValueError` nếu tham số không hợp lệ |
| `top_confusions(cm, class_names, k=5)` | như tên | `List[Tuple[str, str, int]]` | `ValueError` nếu `class_names` lệch kích thước |
| `misclassified_indices(y_true, y_pred, logits, max_items)` | như tên | tensor chỉ mục | không |
| `get_mean_std(transform, transform_weights=None)` | transform và nguồn weights đã cấu hình | `(mean, std)` | `ValueError` khi không đọc được và ImageNet fallback không được xác nhận |
| `denormalize(images, mean, std)` | tensor, danh sách | tensor trong $[0, 1]$ | `ValueError` nếu số kênh không khớp |
| `measure_inference_time(model, images, device, warmup, repeats)` | như tên | `dict` | `ValueError` nếu `warmup < 0` hoặc `repeats < 1` |
| `file_size_mb(path)` | `str` | `float` (MiB) | `FileNotFoundError` nếu thiếu tệp |
| `build_cost_record(num_params, model_path, timing, history, source_history=None)` | như tên; history phase nguồn tùy chọn | `dict` gồm phase/total train time | `FileNotFoundError` nếu `model_path` thiếu |
| `evaluate_predictions(result, class_names)` | như tên | `dict` JSON-ready | `ValueError` truyền từ hàm con |
| `format_comparison_table(runs)` | `Dict[str, Dict]` | `str` Markdown | `KeyError` nếu thiếu `metrics` |
| `plot_*` | xem mục 2.7 | `Figure` | `ValueError` khi đầu vào rỗng hoặc lệch kích thước |
| `finalize_figure(fig, save_path, show, close)` | như tên | `Figure` | không |

### 2.10. Tích hợp với các module khác

| Module | Cách phối hợp |
| :--- | :--- |
| `data_setup.py` | Dùng `test_dataloader` (`shuffle=False`) và `class_names`; mean/std lấy từ preprocessing transform/weights qua `get_mean_std`. |
| `model_builder.py` | `count_params(model)` cấp số tham số cho `build_cost_record`. |
| `engine.py` | `history` cung cấp đường cong học và `epoch_time`; `evaluate_step` và `run_inference` cho cùng loss trên cùng dữ liệu. |
| `utils.py` | Lưu và nạp `test_metrics.json` cùng các artifacts khác. |
| Notebook | Điều phối: nạp mô hình, `run_inference`, `evaluate_predictions`, đo chi phí, vẽ, ghi `test_metrics.json`, in bảng so sánh. |

### 2.11. Kế hoạch kiểm thử

| Nhóm | Kiểm tra tối thiểu | Kỳ vọng |
| :--- | :--- | :--- |
| Inference | Mô hình đồ chơi với nhiều batch, gồm batch cuối nhỏ; đầu vào rỗng | Logits/nhãn được ghép đúng, loss khớp loss trên toàn tập; loader rỗng báo `ValueError`. |
| Classification | Ma trận tính tay, nhãn ngoài miền, tensor sai shape/độ dài, lớp không được dự đoán | Ma trận đúng hướng; dữ liệu sai bị từ chối; precision/recall/F1 không có NaN. |
| Wilson CI | `correct=50`, `total=100`; tổng bằng 0 hoặc correct ngoài miền | CI xấp xỉ `[0.4038, 0.5962]`; dữ liệu sai báo `ValueError`. |
| Predictions JSON | Tạo record rồi serialize bằng `json.dumps` | Có `y_true`, `y_pred`, confidence tối đa; không chứa tensor/logits. |
| Reuse | Có metrics thì nạp và không chạy inference; thiếu metrics thì chạy inference Test một lần, lưu record | Lần reuse sau nạp metrics đã lưu; không train lại. |
| Cost | Baseline có history; fine-tune có history phase và source history; thiếu timing | `train_time_phase_s` và `train_time_total_s` đúng; trường không đo được là `null`. |
| Denormalize | Đọc mean/std từ transform ImageNet; transform custom không có stats; config không xác nhận ImageNet | Đọc đúng stats; ImageNet fallback chỉ dùng khi config xác nhận; trường hợp không xác định báo lỗi. |
| Plots/table | Các hàm plot và bảng nhận dữ liệu đầy đủ/thiếu tùy chọn | Figure trả về/lưu được; dữ liệu thiếu hiển thị bằng dấu gạch dài. |

Chạy kiểm thử bằng `MPLBACKEND=Agg pytest tests/test_evaluate.py -v`.

### 2.12. Mở rộng

* **Nhiều seed:** chạy mỗi thí nghiệm với vài seed và báo cáo trung bình kèm độ lệch chuẩn, thay vì chỉ khoảng tin cậy trên một lần chạy.
* **Kiểm định ghép cặp:** có thể dùng kiểm định McNemar trên các mẫu hai mô hình dự đoán khác nhau trong một phiên bản mở rộng; không thuộc phạm vi đánh giá hiện tại.
* **Hiệu chuẩn độ tự tin (calibration):** dựng biểu đồ độ tin cậy và đo sai số hiệu chuẩn từ `probs()`.
* **FLOPs:** thêm vào bản ghi chi phí nếu cần so sánh theo lượng tính toán thay vì thời gian.
* **Grad-CAM:** trực quan hóa vùng ảnh mô hình chú ý khi dự đoán sai.
* **Ghi TensorBoard:** gửi các `Figure` trả về lên TensorBoard bằng `add_figure`.

### 2.13. Quyết định thiết kế của nhóm so với chuẩn từ tài liệu

| Hạng mục | Có trong tài liệu chính thức | Quyết định thiết kế của nhóm |
| :--- | :--- | :--- |
| `model.eval()` và `torch.inference_mode()` khi suy luận | Có (PyTorch) | |
| Quy ước ma trận nhầm lẫn: hàng thật, cột dự đoán | Có (quy ước phổ biến) | |
| Khoảng tin cậy Wilson cho tỷ lệ | Có (thống kê chuẩn) | |
| Cấu trúc package `going_modular` | Có (Learn PyTorch ch.05) | |
| Suy luận một lượt (`InferenceResult`) | | Có |
| Độ đo bằng PyTorch thuần, `bincount` | | Có |
| Tham số keyword-only cho `compute_confusion_matrix` | | Có |
| Quy ước chia cho $0$ bằng $0$; macro tính trên mọi lớp | | Có |
| Hàm vẽ trả về `Figure`; `finalize_figure` | | Có |
| `start_epochs` để nối đường cong hai giai đoạn | | Có |
| Đo thời gian: trung vị, warm-up, đồng bộ CUDA | | Có |
| Tệp `test_metrics.json` và cấu trúc bản ghi | | Có |

### 2.14. Tài liệu tham khảo

1. PyTorch, `torch.inference_mode`, `torch.bincount`, `torch.cuda.synchronize`, `torch.cuda.max_memory_allocated`: https://docs.pytorch.org
2. PyTorch / torchvision, *Models and pre-trained weights* (tiền xử lý đi kèm trọng số): https://docs.pytorch.org/vision/stable/models.html
3. Daniel Bourke, *Learn PyTorch for Deep Learning*, chương 05 (Going Modular) và 07 (Experiment Tracking): https://www.learnpytorch.io
4. Wilson, E. B. (1927). *Probable inference, the law of succession, and statistical inference* (khoảng tin cậy Wilson cho tỷ lệ).

---

## PHẦN III: NHẬT KÝ QUYẾT ĐỊNH THIẾT KẾ ĐÃ CHỐT

Tất cả 8 điểm thiết kế đánh giá và đối sánh đã được thống nhất và chính thức đưa vào đặc tả:

| # | Vấn đề | Quyết định đã chốt | Ghi chú triển khai |
| :--- | :--- | :--- | :--- |
| 1 | **Phạm vi thực nghiệm** | Ba thí nghiệm cốt lõi và ba runs khảo sát LR/batch size | Mỗi run khảo sát chỉ thay một yếu tố so với ResNet-18 baseline; không phải grid search toàn diện |
| 2 | **Phạm vi "chi phí"** | 5 đại lượng kỹ thuật | Đo params, file size, train time, latency, GPU memory |
| 3 | **Batch đo suy luận** | Batch $B=32$ cố định | Lấy từ tập Test, tính cả `ms/batch` và `ms/image` |
| 4 | **Phương pháp khoảng tin cậy** | Wilson Score 95% | Chuẩn thống kê, không tràn biên $[0, 1]$ |
| 5 | **`epoch_time` trong chi phí** | Tính tổng cả Train và Validation | Đồng bộ với `03_engine.md` |
| 6 | **Lấy mean/std từ weights** | Ưu tiên lấy từ preprocessing preset/transform của torchvision weights | Chỉ fallback ImageNet khi config xác nhận `IMAGENET1K_V1`; nếu không thì yêu cầu stats hoặc báo lỗi |
| 7 | **Định dạng file ảnh biểu đồ** | PNG, 150 dpi | Lưu vào `outputs/YYYY-MM-DD/<RUN_ID>/<EVAL_ID>/` phục vụ báo cáo |
| 8 | **Đánh giá nâng cao (McNemar)** | Giữ ở mục Mở rộng | Wilson CI mô tả từng accuracy, không phải kiểm định chênh lệch giữa các mô hình |
