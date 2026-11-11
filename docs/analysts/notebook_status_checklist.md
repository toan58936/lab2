# Checklist trạng thái Notebook — `notebooks/pretrained_models.ipynb`

> **Ngày rà soát:** 2026-11-11 (cập nhật sau refactor multi-run)  
> **Phạm vi:** Đối chiếu notebook hiện tại với yêu cầu gốc (`docs/yeucaugoc.md`) và thiết kế (`docs/define&designt/00_lab2_description.md`, `docs/plans/*`).  
> **Mô hình chạy:** Hỗ trợ 1 hoặc nhiều `RUN_ID` chạy tuần tự trong một phiên. Danh sách run: `v0_vgg16`, `v1_resnet18` (feature extraction), `v1_resnet18_finetune` (run fine-tune duy nhất).

## 1. Quy trình đã triển khai

### 0. Setup & cấu hình
- [x] Mount Google Drive trên Colab (Cell 1, Cell 3).
- [x] Xác định môi trường Colab/local, clone/pin code từ GitHub, thiết lập `CODE_ROOT`, `DATA_DIR`, `ARTIFACT_ROOT`, `RUNS_ROOT` (Cell 3).
- [x] Cài `torchinfo`, import torch/torchvision/tensorboard, import `going_modular` và lấy `DEVICE` (Cell 4).
- [x] Cấu hình `RUN_ID`/`RUN_IDS` (đơn/list/chuỗi phân tách dấu phẩy), `RUN_SPECS` 6 run, và **hành động do người dùng chọn** qua `RUN_PLAN`/`DEFAULT_RUN_ACTION` (`train` | `resume` | `reuse`) + `FORCE_RETRAIN` (Cell 5, có ghi chú `#` hướng dẫn). Code không tự suy luận trạng thái.
- [x] Validate run/action và thứ tự phụ thuộc fine-tune; tạo output/log; lưu kết quả trong `run_results`.

### 1. Dữ liệu CIFAR-10
- [x] Kiểm tra/stage CIFAR-10 từ Drive hoặc local (Cell 7).
- [x] Tạo dataloader per-run qua `create_dataloaders_for_spec` (batch size/transform theo spec), `download=False` (Cell 7, 16).
- [x] Kiểm tra batch shape/dtype/pixel range cho run đầu tiên (Cell 8).

### 2. Kiến trúc & mô hình
- [x] Hiển thị `torchinfo.summary` cho **cả ResNet-18 và VGG16** (`depth=3`), sau đó dựng model per-run bằng `ModelConfig` + `build_model` (Cell 11).
- [x] Fine-tune: nạp `best_model.pth` của baseline `v1_resnet18` vào model trước khi train khi run có dependency (Cell 11).
- [x] Kiểm tra output shape `[B, NUM_CLASSES]`, in `model_selected.head`.

### 3. Huấn luyện / Resume / Reuse (tuần tự, theo action người dùng chọn)
- [x] Lặp qua `RUN_ID_LIST`: resolve action/spec/paths, kiểm tra artifacts theo action, tạo dataloader/model/optimizer/writer per-run (Cell 16).
- [x] `train` (mới): nếu artifacts đã tồn tại và `FORCE_RETRAIN=False` → chặn (không tự xóa); `FORCE_RETRAIN=True` → xóa artifacts cũ rồi train từ đầu. `engine.train(..., verbose=True)`, lưu config/history.
- [x] `resume`: yêu cầu `last_checkpoint.pth`, kiểm tra `done_epochs < target_epochs`, gọi `engine.train(resume=True)` để nạp model/optimizer/history/epoch và chạy tiếp (phù hợp Colab ngắt kết nối).
- [x] `reuse`: nạp history/config/best_model từ artifacts, không train.
- [x] Fine-tune: train mới nạp baseline `v1_resnet18/best_model.pth`; resume/reuse nạp checkpoint/best của chính run fine-tune (không nạp baseline).

### 4. So sánh & đánh giá
- [x] Tóm tắt kết quả các run đã chạy (val acc, thời gian, trainable params, epochs, batch, LR) (Cell 19).
- [x] Vẽ learning curves so sánh nhiều run qua `evaluate.plot_history` (Cell 20).
- [x] Đánh giá test per-run: 1 lần inference (`run_inference`), tính metrics (`evaluate_predictions`), lưu/nạp `test_metrics.json`, tái sử dụng `y_true/y_pred/cm` (Cell 23).
- [x] Confusion matrix, per-class accuracy, top-5 nhầm lẫn, lưới ảnh dự đoán sai cho từng run (Cell 24–27).

## 2. Điểm lệch / lưu ý còn lại

### Cấu hình thí nghiệm
- [x] **Epochs (thử nghiệm):** `RUN_SPECS` đặt `epochs=3` cho tất cả run; thiết kế gốc 5 epoch cho Feature Extraction. Có thể tăng khi chạy chính thức (đã ghi chú trong docs).
- [x] **Nhiều run:** hỗ trợ `RUN_IDS` chạy tuần tự (mặc định vẫn `v0_vgg16` nếu không đổi).

### Huấn luyện & fine-tune
- [x] **Fine-tune nạp baseline:** train mới nạp `v1_resnet18/best_model.pth`; resume nạp `last_checkpoint.pth` của run fine-tune (không nạp baseline).
- [x] **Log từng epoch:** `verbose=True`.
- [x] **Artifact guard (user-controlled):** `train` không tự xóa artifacts khi đã tồn tại; chỉ xóa khi `FORCE_RETRAIN=True`. `resume`/`reuse` kiểm tra điều kiện artifacts/epoch; code không tự suy luận action.

### TensorBoard & quan sát
- [ ] **Chưa có cell UI TensorBoard trong notebook:** đã ghi log vào `logs/<RUN_ID>/<SESSION_ID>/` và hướng dẫn trong markdown; có thể bổ sung `%load_ext tensorboard` / `%tensorboard --logdir logs` cho Colab.

### Đánh giá test
- [x] Import/scope `run_inference`, `loss_fn`, `denormalize` đã chuẩn hóa trong luồng đánh giá.
- [x] Inference/predict được tái sử dụng per-run (không tính lại CM/predict lặp trong các cell).
- [ ] **Kiểm tra thực nghiệm:** cần chạy trên dữ liệu/Colab thật để xác nhận end-to-end (không thuộc phạm vi sửa code).

### Nội dung notebook
- [x] Markdown/cấu trúc đã cập nhật theo luồng RUN_ID/RUN_IDS tuần tự.
- [ ] **So sánh tổng hợp nâng cao:** đã có summary/curves/metrics per-run; có thể bổ sung bảng tổng hợp test metrics (accuracy/F1/time/cost) từ artifacts của tất cả 6 run sau khi chạy đủ.

## 3. Quy trình / thành phần còn thiếu

- [ ] **Chạy đủ 3 thí nghiệm và thu thập kết quả:** `v0_vgg16`, `v1_resnet18`, `v1_resnet18_finetune` theo thứ tự phụ thuộc, lưu artifacts/outputs đầy đủ.
- [ ] **Báo cáo so sánh cuối cùng:** bảng/biểu đồ đối chiếu VGG16 vs ResNet-18 (feature extraction) và baseline vs fine-tune.
- [ ] **Cell TensorBoard (tùy chọn):** `%tensorboard` cho Colab/local.
- [ ] **Nghiệm thu Colab/Drive:** clone đúng revision, mount Drive, staging CIFAR-10, ghi artifacts/outputs/logs thực tế.

## 4. Ghi chú nhanh
- Chọn run: `RUN_ID` đơn hoặc `RUN_IDS` (list/chuỗi phân tách dấu phẩy).
- Chọn hành động per-run trong `RUN_PLAN` (hoặc `DEFAULT_RUN_ACTION`): `train` (đầy đủ, từ đầu), `resume` (tiếp tục từ `last_checkpoint.pth`, `done < target`), `reuse` (dùng artifacts có sẵn). Bật `FORCE_RETRAIN=True` chỉ khi muốn xóa artifacts và train lại từ đầu.
- Fine-tune: chỉ `v1_resnet18_finetune` (duy nhất). Chạy `v1_resnet18` trước, sau đó `v1_resnet18_finetune`.
- Test set chỉ dùng đánh giá cuối. Logs: `logs/<RUN_ID>/<SESSION_ID>/`. Output: `outputs/<date>/<RUN_ID>/<EVAL_ID>/`.
