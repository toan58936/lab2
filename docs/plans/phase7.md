> **Cập nhật 2026-11-11 (hướng B, historical context):** checklist này phản ánh tích hợp giai đoạn trước refactor native. Hiện tại model là torchvision native (G1 VGG16 `classifier[6]`, ResNet `fc`), không còn wrapper `MyModel/backbone.head`, `state_dict` key gốc; hợp đồng flat export hiện hành 12 symbols; load/resume/reuse có guard từ chối format cũ và cần retrain 3 run. Rà soát/cập nhật khi dùng làm tham chiếu tiếp.

# Phase 7 — Checklist tích hợp Notebook và package

> **Điểm đến dự kiến:** `docs/plans/phase7.md`  
> **Nguồn:** Phase 7 của `docs/plans/implementation_plan.md`, hợp đồng hệ thống tại `docs/define&designt/00_lab2_description.md`, API Phase 2–6.  
> **Trạng thái:** Bản checklist để người dùng tạo/rà soát; chưa đánh dấu hoàn thành.

## Mục tiêu và ranh giới

Tích hợp `notebooks/pretrained_models.ipynb` với các API module đã hoàn thành: chọn và xử lý một run tường minh, hỗ trợ train/resume/reuse, bảo vệ artifacts, đánh giá và trình bày kết quả, kiểm tra package exports.

- Phase 7 chỉ chạy smoke test ngắn, có giới hạn.
- Không chạy đủ sáu thí nghiệm (Phase 8), không tạo báo cáo kết quả cuối (Phase 9).
- Không xóa API legacy hoặc artifacts chỉ để làm gọn; migration và cleanup legacy là các thay đổi riêng sau khi không còn call sites.
- Tuân thủ chính sách hiện hành: resume ở ranh giới epoch; không khôi phục RNG state.

## 1. Điều kiện đầu vào và rà soát call sites

- [x] Xác nhận các cổng/test của Phase 2–6 đạt trong môi trường dự án; ghi lại kết quả trước khi tích hợp.
- [x] Đối chiếu API hiện tại và chữ ký thực tế của `model_builder`, `data_setup`, `engine`, `utils`, `evaluate`; không dựa vào ví dụ chữ ký cũ trong notebook.
- [x] Kiểm kê các cell hiện dùng wrapper model, `torchinfo`, `checkpoint_path`, `frozen_backbone`, `save_training_metadata`, `load_training_artifacts`, `eval_model`, `predict_dataloader` và lời gọi confusion matrix positional.
- [x] Xác nhận hợp đồng 14 exports công khai trong `00_lab2_description.md`; chỉ yêu cầu flat-export đúng 14 symbols đã chốt, không nhầm với toàn bộ helper nội bộ.

## 2. Cấu hình run và chế độ hoạt động

- [x] Tạo cấu hình cho đúng sáu `RUN_ID`: `v0_vgg16`, `v1_resnet18`, `v1_resnet18_finetune`, `v2_resnet18_lr1e-2`, `v2_resnet18_lr1e-4`, `v2_resnet18_bs64`.
- [x] Mỗi lần chạy chỉ chọn đúng một `RUN_ID`; từ chối giá trị không hợp lệ, không tự chạy tuần tự tất cả runs.
- [x] Bảo đảm thông số mỗi run khớp bảng thực nghiệm trong `00`; fine-tune phụ thuộc baseline `v1_resnet18`.
- [x] Validate `RUN_MODE` chỉ nhận `train` hoặc `reuse`; `RESUME` chỉ phân biệt fresh train với tiếp tục run đang dở trong chế độ `train`.
- [x] Lưu config đã resolve theo schema chuẩn trước khi train; config phase dùng cho resume phải khớp hoàn toàn.

## 3. Bootstrap runtime, dữ liệu và artifact root

- [x] Nếu chạy Colab, mount Drive trước khi chọn artifact root; ngoài Colab dùng root local đã cấu hình.
- [x] Mỗi run có thư mục artifact và TensorBoard log riêng; mọi đường dẫn model/history/config/metrics đều thuộc đúng `RUN_ID`.
- [x] Gọi `set_seed` trước khi dựng model và DataLoader; tạo loader mới cho run.
- [x] Lấy preprocessing từ `model_builder.get_transforms`; truyền `train_transform`/`eval_transform` qua dependency injection vào data factory.
- [x] Truyền `torch.Generator` riêng đã seed cho Train DataLoader.
- [x] Fresh train phải dừng với hướng dẫn rõ ràng nếu artifacts của run đã tồn tại; không tự xóa hoặc ghi đè.

## 4. Luồng train và resume an toàn

- [x] Dựng model bằng `ModelConfig`/`build_model`; hiển thị kiến trúc bằng `torchinfo.summary()` trực tiếp trong notebook.
- [x] Gọi `engine.train` qua API chuẩn, truyền `save_path`, `checkpoint_dir`, `resume`, `expected_phase_config`, `start_epoch` và callback train-mode phù hợp.
- [x] Lưu `last_checkpoint.pth` cùng history sau mỗi epoch hoàn tất bằng các API persistence hiện có; lỗi đọc/ghi phải dừng run và được báo rõ.
- [x] Resume yêu cầu checkpoint tồn tại, đọc được, khớp phase config và chưa hoàn tất mục tiêu epochs; không tự chuyển thành fresh train.
- [x] Xác minh resume tiếp tục từ epoch kế tiếp, giữ history liên tục và không lặp/bỏ epoch đã hoàn tất.
- [x] Nêu rõ giới hạn trong notebook/tài liệu: runtime ngắt giữa epoch thì epoch đó phải chạy lại; RNG state không được khôi phục theo quyết định hiện hành.

## 5. Luồng Fine-Tuning

- [x] Với `v1_resnet18_finetune`, nạp `best_model.pth` của `v1_resnet18` vào ResNet-18 được dựng với `weights=None`.
- [x] Cấu hình mở `layer4` và head; không thao tác qua wrapper `.model`.
- [x] Dùng optimizer mới với parameter groups riêng: head LR `1e-3`, backbone LR `1e-4`.
- [x] Đặt `start_epoch=5`; phase vẫn train đúng 3 epochs, còn TensorBoard/global epoch tiếp tục từ baseline.
- [x] Gắn `source_history` của baseline khi tính tổng train time; không cộng thời gian nguồn vào thời gian phase hiện tại.

## 6. Reuse và đánh giá

- [x] `reuse` yêu cầu `best_model.pth`, `history.json`, `config.json`; thiếu artifact huấn luyện thì báo lỗi, không train thay thế.
- [x] Dựng đúng kiến trúc với `weights=None`, sau đó nạp checkpoint; xác nhận config/run ID khớp.
- [x] Nếu `test_metrics.json` tồn tại thì nạp và dùng lại; nếu chỉ thiếu file metrics thì chạy Test inference một lần, lưu rồi dùng record đó.
- [x] Dùng một lượt `run_inference` cho mỗi đánh giá; tạo metrics/predictions bằng `evaluate_predictions`, không suy luận lặp lại qua legacy helpers.
- [x] Gọi `compute_confusion_matrix` bằng keyword-only với `y_true` và `y_pred` đúng thứ tự.
- [x] Tạo cost record qua `count_params`/`build_cost_record`, gồm history phase và source history khi cần; không dùng Test để chọn checkpoint/hyperparameter.
- [x] Bảo đảm predictions lưu chỉ gồm `y_true`, `y_pred`, confidence; không lưu logits.

## 7. Hoàn thiện Notebook và package contract

- [x] Thay các cell V0/V1/fine-tune rời thành luồng điều phối theo một `RUN_ID` và API module chuẩn.
- [x] Thay `torchinfo.summary()` bằng `torchinfo.summary()` trực tiếp trong notebook; rà soát cả hướng dẫn/cell setup có nhắc đến `torchinfo`.
- [x] Chỉ gỡ `torchinfo` khỏi `requirements.txt` sau khi tìm toàn repository xác nhận không còn import/call/documentation dependency.
- [x] Sửa các lỗi call site đã biết, đặc biệt confusion matrix keyword-only và thứ tự labels.
- [x] Bổ sung smoke test package kiểm tra chính xác `__all__` bằng 14 symbols theo `00`; cập nhật các test/import bị ảnh hưởng.
- [x] Không xóa legacy module API chỉ vì Notebook đã được chuyển; xác nhận riêng rằng không còn caller, rồi dọn trong thay đổi/phase được thống nhất.

## 8. Xác minh tích hợp và cổng nghiệm thu

- [x] Chạy unit tests của Phase 2–6 và package smoke/export test trước khi kiểm tra Notebook.
- [x] Chạy notebook smoke test ngắn với dữ liệu nhỏ/fake hoặc cấu hình giới hạn, không train các run đầy đủ.
- [x] Kiểm tra từng mode: fresh train không overwrite; resume thiếu/sai checkpoint hoặc sai config thì fail; reuse không train.
- [x] Mô phỏng ngắt sau một epoch đã lưu; khởi động lại cùng `RUN_ID`, xác minh resume từ epoch kế tiếp và không làm hỏng artifacts đã lưu.
- [x] Kiểm tra metrics cache: reuse metrics hiện có không chạy inference lại; metrics thiếu duy nhất được tạo một lần.
- [x] Chạy collection/full unit suite sau thay đổi; rà soát diff và `git diff --check`.
- [x] Ghi rõ lệnh, môi trường và kết quả kiểm thử thực tế; không đánh dấu đạt chỉ dựa vào việc checklist/code đã tồn tại.

## Cổng hoàn tất Phase 7

- [x] Một run được chọn chạy được xuyên suốt theo mode đã chọn, không tác động các run khác.
- [x] Fresh train, epoch-boundary resume và reuse tuân thủ đầy đủ chính sách artifact/config.
- [x] Fine-tune khởi tạo đúng từ baseline; đánh giá và cost record dùng contracts Phase 2/4/5/6.
- [x] Notebook không còn lỗi runtime/API đã biết; package contract 14 exports được kiểm tra.
- [x] Smoke/integration checks và unit suite liên quan đạt; chưa chạy full experiments.

**Kết quả Phase 7:** ☐ Chưa đạt  ☒ Đạt  
**Ngày rà soát:** 2026-11-05  
**Ghi chú / quyết định còn mở:**  
