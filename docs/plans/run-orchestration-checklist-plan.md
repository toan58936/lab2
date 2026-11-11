# Kế hoạch: Notebook điều phối 1 hoặc nhiều RUN_ID (chạy tuần tự)

## Mục tiêu và giới hạn

Notebook điều phối chọn `RUN_ID` (đơn) hoặc `RUN_IDS` (danh sách, chạy tuần tự), chế độ `RUN_MODE` (`train`/`reuse`), `RESUME` và quản lý artifact (config/history/checkpoint/metrics theo `RUN_ID`). `going_modular/engine.py` phụ trách vòng lặp train/validation, chọn best model và cơ chế checkpoint/resume.

Ưu tiên dùng API engine hiện có. Không mở rộng engine thành bộ điều phối nhiều run; orchestration tuần tự nằm ở notebook.

## Checklist triển khai

### A. Chốt hợp đồng run và kiểm tra đầu vào

- [x] `RUN_ID`/`RUN_IDS` resolve đúng các mục trong `RUN_SPECS` (hỗ trợ đơn, list, chuỗi phân tách dấu phẩy); kiểm tra đầy đủ 3 spec và dependency của fine-tune (`v1_resnet18_finetune` duy nhất).
- [x] Quy định rõ `RUN_MODE=train|reuse` và `RESUME`: resume chỉ áp dụng cho train; báo lỗi sớm nếu mode, spec hoặc cấu hình không hợp lệ.
- [x] Tạo cấu hình hiệu lực cho từng run được chọn (model/weights/freeze, epochs, batch size, learning rate/optimizer, param groups, seed, dependency); ghi cấu hình vào đúng thư mục run.
- [x] Kiểm tra trạng thái artifact trước khi chạy từng run: train mới (có cơ chế dọn dẹp khi chạy lại), resume cần checkpoint, reuse cần đủ config/history/best checkpoint. Không tự chuyển reuse lỗi thành train.

### B. Thay luồng notebook cố định bằng luồng per-run (tuần tự)

- [x] Chỉ dựng model cho run đang xử lý; khởi tạo `DEVICE` trước các cell train/evaluate.
- [x] Dùng transform/dataloader phù hợp kiến trúc và batch size của run đang xử lý; tạo lại dataloader khi batch_size/transform thay đổi.
- [x] Hợp nhất các cell VGG16, ResNet-18 baseline và fine-tune thành luồng per-run; chạy theo `RUN_IDS` không vô tình bỏ sót run.
- [x] Truyền đúng optimizer, epochs, log directory, artifact directory, config kỳ vọng, `RESUME` và epoch offset vào `engine.train`; `verbose=True` để hiển thị tiến trình epoch.
- [x] Với fine-tune, nạp `best_model.pth` của baseline `v1_resnet18` vào model trước khi train; thiết lập optimizer parameter groups theo spec.
- [x] Notebook không tự triển khai vòng lặp train/validation hay chọn best model.

### C. Reuse, đánh giá và so sánh

- [x] Reuse nạp model/history/config từ artifact của từng run mà không train lại.
- [x] Đánh giá test sau khi mỗi run hoàn thành; lưu/nạp `test_metrics.json` theo run/EVAL_ID, tránh tính lại inference nhiều lần trong cùng cell.
- [x] Tóm tắt và vẽ history cho các run đã chạy; đánh giá test/CM/per-class/top-confusions/sample predictions cho từng run.

### D. Đồng bộ tài liệu

- [x] `docs/define&designt/00_lab2_description.md` — cập nhật chế độ chạy/chọn run tuần tự và quy ước fine-tune.
- [x] `docs/analysts/notebook_status_checklist.md` — cập nhật trạng thái notebook.
- [x] Không tạo tài liệu orchestration mới.

## Tiêu chí nghiệm thu

- Mỗi giá trị RUN_ID chọn đúng spec, model, đường dẫn và nhánh xử lý; hỗ trợ chạy tuần tự danh sách run với thứ tự phụ thuộc hợp lệ.
- Fine-tune nạp baseline checkpoint trước khi train.
- Notebook JSON hợp lệ và code cells biên dịch được.
- Engine sử dụng đúng API hiện có (`verbose`, `train_mode_fn`, checkpoint/resume).
