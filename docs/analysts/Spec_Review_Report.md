# Nhật ký Chốt Đặc tả — Lab 2

> **Mục đích:** Ghi lại các quyết định đã chốt sau khi rà soát tài liệu Define & Design.
> **Nguồn sự thật:** [`00_lab2_description.md`](../define&designt/00_lab2_description.md); các đặc tả module nằm trong `docs/define&designt/`.
> **Trạng thái:** Các quyết định dưới đây đã được đưa vào tài liệu. Bộ tài liệu sẵn sàng làm baseline cho giai đoạn triển khai.

## 1. Phạm vi thực nghiệm

Phạm vi đã chốt gồm sáu thí nghiệm:

| Run | Mục tiêu |
| :--- | :--- |
| `v0_vgg16` | Đối sánh VGG16 Feature Extraction |
| `v1_resnet18` | ResNet-18 Feature Extraction, baseline |
| `v1_resnet18_finetune` | Fine-tune `layer4` và head từ checkpoint ResNet-18 |
| `v2_resnet18_lr1e-2` | Khảo sát head learning rate cao hơn baseline |
| `v2_resnet18_lr1e-4` | Khảo sát head learning rate thấp hơn baseline |
| `v2_resnet18_bs64` | Khảo sát batch size 64 so với baseline |

Ba runs `v2` mỗi run chỉ thay một yếu tố so với `v1_resnet18`, giữ nguyên seed/dữ liệu/kiến trúc/epochs. Đây là khảo sát có kiểm soát learning rate và batch size theo yêu cầu lab; không phải grid search toàn diện. Không có run đối chứng 8 epoch.

## 2. Quyết định xuyên module

| Chủ đề | Quyết định |
| :--- | :--- |
| Notebook chuẩn | Giữ tên hiện có `notebooks/pretrained_models.ipynb`. |
| Cấu hình/artifacts | `00` là nguồn duy nhất cho config, runs và chính sách lưu artifacts. |
| History | Sáu keys: `train_loss`, `val_loss`, `train_acc`, `val_acc`, `lr`, `epoch_time`; LR ghi theo optimizer group, epoch time gồm Train và Validation. |
| Artifacts mỗi run | `best_model.pth`, `last_checkpoint.pth`, `history.json`, `config.json`, `test_metrics.json`; file Test metrics chỉ có sau đánh giá. Checkpoint `.pth` không commit. |
| Epoch theo phase | `train.epochs` là số epoch mục tiêu của phase; `start_epoch` chỉ là offset TensorBoard/báo cáo. Fine-tune `epochs=3`, `start_epoch=5` chạy ba epoch và ghi bước 6–8. |
| Resume | Mặc định tắt; chỉ bật theo yêu cầu. Tiếp tục tại ranh giới epoch với checkpoint và phase config khớp; không khôi phục RNG, nên không cam kết cùng chuỗi batch như chạy liên tục. |
| Khởi tạo fine-tune/reuse | Dựng kiến trúc với `model.weights=null`, nạp checkpoint qua `train.init_from` hoặc run artifacts; preprocessing độc lập qua `data.transform_weights`. |
| Quyền ghi artifacts | Notebook/orchestrator ghi `config.json` trước train; engine ghi checkpoint và cập nhật `history.json` nguyên tử sau mỗi epoch. |
| Chính sách reuse | Bắt buộc có checkpoint tốt nhất, history và config; thiếu thì báo lỗi, không tự train. Nếu chỉ thiếu `test_metrics.json`, chạy inference Test một lần, lưu rồi dùng lại. |
| Predictions và chi phí | `test_metrics.json` lưu `y_true`, `y_pred`, confidence tối đa, không lưu logits. Thời gian gồm `train_time_phase_s` và `train_time_total_s` (cộng phase nguồn nếu có). |
| Tiền xử lý | Notebook lấy transform từ `model_builder.get_transforms(cfg, weights=data.transform_weights)` rồi inject vào `data_setup`; data module giữ fallback độc lập, không tạo circular dependency. Mean/std ưu tiên từ transform/preset; chỉ fallback ImageNet khi config xác nhận `IMAGENET1K_V1`. |
| Tài liệu utils | Tách `utils.py` thành `docs/define&designt/04_utils.md`; `docs/define&designt/03_engine.md` chỉ mô tả engine và tham chiếu API utilities. |
| DataLoader và seed | Tạo riêng loader cho mỗi run sau khi gọi seed; Train DataLoader có generator riêng. `pin_memory=torch.cuda.is_available()`, độc lập với device được chọn. |
| Stratified split | Phạm vi cam kết hiện tại chỉ là CIFAR-10 cân bằng, `val_size=5000`; không khẳng định hỗ trợ mọi phân phối nhãn. |
| Huấn luyện | Checkpoint chọn bằng `val_acc` mặc định (có hỗ trợ `val_loss`); không dùng Test để chọn mô hình. |
| Fine-tuning | Head LR `1e-3`, backbone LR `1e-4`; 3 epoch tiếp theo; optimizer mới; `start_epoch=5`; artifacts riêng. |
| TensorBoard | `runs/<experiment>`; epoch offset nối tiếp giữa hai giai đoạn. |
| Tính tái lập | Seed 42 và `deterministic=True`; cam kết trong cùng môi trường, không cam kết giống hệt giữa phần cứng/phiên bản thư viện khác nhau. |
| Evaluation | Wilson 95% CI mô tả từng accuracy, không dùng chồng lấp CI làm kiểm định so sánh; 5 đại lượng chi phí (params, file size, train time, latency, peak GPU memory); latency dùng batch 32; PNG 150 dpi. |
| Ngoài phạm vi | Nhiều seed, McNemar và grid search toàn diện. |

## 3. Chính sách transforms và luồng dữ liệu

1. `model_builder.get_transforms(cfg, weights=data.transform_weights)` cung cấp transform đánh giá khớp với bộ weights.
2. Notebook có thể ghép augmentation train, sau đó truyền cả hai transform vào `create_train_val_test_dataloaders`.
3. `data_setup.get_pretrained_transform(...)` vẫn tồn tại làm fallback khi caller không truyền transform; hàm này không import `model_builder`.
4. Tạo loader độc lập cho mỗi run; tập Test chỉ dùng để đánh giá cuối.

## 4. Kết quả rà soát tài liệu

- [`00_lab2_description.md`](../define&designt/00_lab2_description.md): danh mục sáu runs, config/artifacts, transform injection và nguyên tắc tạo loader là quy định chung.
- [`01_data.md`](../define&designt/01_data.md): mô tả fallback transform và nhận transforms từ Notebook, không yêu cầu data module ủy quyền ngược cho model module.
- [`02_model.md`](../define&designt/02_model.md): xác định API model, transforms, freeze/unfreeze và tương tác với Notebook/data module.
- [`03_engine.md`](../define&designt/03_engine.md): history, resume, TensorBoard và logic train đã đồng bộ.
- [`04_utils.md`](../define&designt/04_utils.md): seed/device, TensorBoard writer và artifact I/O được đặc tả độc lập.
- [`05_evaluation_results.md`](../define&designt/05_evaluation_results.md): đánh giá sáu runs, chi phí, CI và các phương pháp nâng cao ngoài phạm vi đã đồng bộ.
- [`implementation_plan.md`](../plans/implementation_plan.md): kế hoạch theo phase và cổng kiểm thử dựa trên thiết kế đã chốt.

Notebook và thư mục test hiện có trong workspace là các điểm cần đối chiếu/hoàn thiện trong giai đoạn triển khai. Trạng thái Define & Design đã chốt chỉ xác nhận baseline tài liệu; không xác nhận code đã đạt các cổng nghiệm thu.
