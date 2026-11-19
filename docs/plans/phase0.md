# Phase 0 — Checklist hoàn tất Define & Design

> **Nguồn chính:** [`implementation_plan.md`](./implementation_plan.md)
> **Mục tiêu:** Xác nhận các quyết định thiết kế Lab 2 đã được chốt và nhất quán trước khi bắt đầu Phase 1 (hạ tầng kiểm thử).
> **Phạm vi:** Rà soát/cập nhật tài liệu; chưa triển khai code hoặc chạy thí nghiệm.

## 1. Điều kiện đầu vào

- [x] Đã đọc kế hoạch tổng thể và xác định Phase 0 là giai đoạn hoàn tất Define & Design.
- [x] Đã xác định `docs/define&designt/00_lab2_description.md` là nguồn chuẩn cho cấu hình chung, schema, runs, artifacts và quy ước toàn hệ thống.
- [x] Đã xác định các đặc tả module `01_data.md`–`05_evaluation_results.md` và `docs/define&designt/tests/00_test_plan.md`–`05_test_evaluate.md` là các tài liệu cần đối chiếu.
- [x] Đã tách các nhận định về code hiện tại khỏi quyết định thiết kế mục tiêu; tài liệu thiết kế không được xem là bằng chứng code đã đáp ứng.

## 2. Chốt phạm vi thực nghiệm

- [x] Có đúng sáu runs trong `00` và kế hoạch:
  - [x] `v0_vgg16`: VGG16 Feature Extraction, 20 epochs (early stopping).
  - [x] `v1_resnet18`: ResNet-18 Feature Extraction, 20 epochs (early stopping); baseline.
  - [x] `v1_resnet18_finetune`: khởi tạo từ checkpoint tốt nhất của baseline, mở `layer4` và head, 20 epochs (early stopping).
  - [x] `v2_resnet18_lr1e-2`: ResNet-18 Feature Extraction, chỉ đổi head LR thành `1e-2`.
  - [x] `v2_resnet18_lr1e-4`: ResNet-18 Feature Extraction, chỉ đổi head LR thành `1e-4`.
  - [x] `v2_resnet18_bs64`: ResNet-18 Feature Extraction, chỉ đổi batch size thành `64`.
- [x] Các runs `v2` giữ nguyên seed, dữ liệu, kiến trúc và số epochs; không thêm run đối chứng 8 epochs hoặc mở rộng thành grid search.
- [x] Mục đích và giới hạn kết luận của từng run được ghi thống nhất trong `00`, `05` và kế hoạch triển khai.

## 3. Đồng bộ hợp đồng thiết kế giữa các tài liệu

- [x] Schema `config.json` trong `00` thống nhất với các module sử dụng config.
- [x] Quy ước `history` thống nhất sáu keys: `train_loss`, `val_loss`, `train_acc`, `val_acc`, `lr`, `epoch_time`.
- [x] Quy trình Feature Extraction và Fine-Tuning thống nhất:
  - [x] Fine-Tuning khởi tạo từ checkpoint baseline với kiến trúc dựng bằng `weights=None`.
  - [x] Optimizer Fine-Tuning là optimizer mới; LR head `1e-3`, LR backbone mở khóa `1e-4`.
  - [x] BatchNorm của phần backbone đóng băng tiếp tục ở eval mode theo chính sách đã chốt.
  - [x] `start_epoch=5` là offset hiển thị/log; Fine-Tuning vẫn chạy tối đa 20 epochs (early stopping) trong phase.
- [x] Quy trình dữ liệu thống nhất:
  - [x] Mỗi run tạo DataLoader riêng sau khi thiết lập seed.
  - [x] Train DataLoader nhận generator riêng; Validation/Test không shuffle.
  - [x] Notebook inject transforms vào `data_setup`; fallback module dữ liệu giữ độc lập.
  - [x] Augmentation chỉ áp dụng cho Train; mean/std lấy từ preprocessing đã cấu hình.
- [x] Quy tắc đánh giá thống nhất: chọn checkpoint bằng Validation; Test chỉ dùng đánh giá cuối, không dùng để chọn model hay điều chỉnh hyperparameter.
- [x] Hợp đồng `evaluate.py` thống nhất một lượt inference Test, metrics/predictions JSON-ready, confidence tối đa theo mẫu, không lưu logits, và chi phí tách `train_time_phase_s` với `train_time_total_s`.
- [x] Hợp đồng package thống nhất 14 exports phẳng theo `00` mục 2.8.6; kế hoạch cho phép test module import trực tiếp trong các phase và kiểm tra toàn bộ exports khi tích hợp.

## 4. Chốt artifacts, reuse, resume và vận hành Colab

- [x] Mỗi run có thư mục artifact riêng và danh sách file đầu ra thống nhất:
  - [x] `config.json`, `history.json`.
  - [x] `best_model.pth`, `last_checkpoint.pth`.
  - [x] `test_metrics.json` chỉ được tạo sau khi đánh giá Test hoàn tất.
- [x] Chính sách Git rõ ràng: checkpoint `.pth` không commit; JSON/figures theo quy ước ở `00`.
- [x] `RUN_ID` chọn tường minh một trong sáu runs; mỗi phiên có thể chỉ chạy hoặc resume một run, không mặc định train toàn bộ sáu runs liên tiếp.
- [x] Phân biệt rõ:
  - [x] `RUN_MODE="train"`: bắt đầu một run mới, không âm thầm ghi đè artifacts.
  - [x] `RESUME=True`: tiếp tục run chưa hoàn tất từ checkpoint phù hợp, không tự khởi động lại nếu checkpoint thiếu/sai config.
  - [x] `RUN_MODE="reuse"`: nạp run đã hoàn thành, không tự train; chỉ đánh giá Test nếu thiếu riêng `test_metrics.json`.
- [x] Resume chỉ khôi phục tại ranh giới epoch đã lưu; epoch đang chạy khi bị ngắt có thể phải chạy lại; RNG không được khôi phục.
- [x] Khi dùng Colab, artifact root nằm trên Drive đã mount; `last_checkpoint.pth` và history được ghi sau mỗi epoch hoàn tất. Lỗi ghi phải làm dừng run và được báo rõ.
- [x] Không xóa artifacts cũ; checkpoint legacy không được hỗ trợ tương thích ngược, nhưng dữ liệu hiện có phải được giữ nguyên trừ khi có xác nhận/sao lưu rõ ràng.
- [x] Có tiêu chí kiểm thử mô phỏng ngắt phiên, resume từ epoch đã lưu, và xác nhận run hoàn tất không bị train lại.

## 5. Chốt chính sách legacy và phạm vi thay đổi

- [x] Đã quy định phải rà soát legacy APIs/classes và call sites trong workspace trước khi loại bỏ hoặc deprecated; việc rà soát thực tế được thực hiện ở phase refactor liên quan.
- [x] Chỉ loại bỏ API cũ sau khi call sites liên quan đã chuyển sang API chuẩn và có kiểm thử phù hợp.
- [x] Không xóa hoặc ghi đè artifacts cũ như một phần ngầm định của “clean start”.
- [x] Những khác biệt được xác định là API/tiện ích, không phải lỗi hành vi, được ghi đúng mức độ; không biến chúng thành yêu cầu xóa nếu chưa có quyết định.

## 6. Chốt bộ đặc tả test

- [x] `docs/define&designt/tests/00_test_plan.md` và các file test module phản ánh cùng hợp đồng với `01`–`05`; V16 đã được sửa theo quy tắc lấy stats từ transform hoặc preset weights đã cấu hình.
- [x] Mã test trong từng phase của kế hoạch khớp danh mục mã trong test plan:
  - [x] Data: D1–D15 và S1–S3.
  - [x] Model: T1–T14c và S1–S2.
  - [x] Engine: E1–E12.
  - [x] Utils: U1–U11.
  - [x] Evaluate: V1–V24.
- [x] Test chậm, phụ thuộc mạng/dữ liệu/trọng số, được đánh dấu riêng; test thường chạy nhanh và không cần mạng.
- [x] Cổng Phase 1 yêu cầu smoke test thực thi được và pytest collect thành công sau khi có test scaffold; không xem `conftest.py` đơn lẻ là đủ.

## 7. Kiểm tra nhất quán tài liệu và lỗi đã biết

- [x] Đã rà soát và xử lý các mâu thuẫn đã biết giữa `00`–`05`, kế hoạch và `docs/define&designt/tests/`; các đường dẫn lỗi thời đã được đồng bộ.
- [x] Đã ghi các lỗi tích hợp code đã biết vào nơi theo dõi phù hợp, không nhầm chúng với quyết định Define & Design. Tối thiểu bao gồm lời gọi `compute_confusion_matrix` trong notebook phải dùng keyword arguments đúng chiều `y_true`, `y_pred`.
- [x] Các tuyên bố về code hiện tại đã được phân biệt với yêu cầu mục tiêu và sẽ được xác minh bằng test ở các phase triển khai.
- [x] Các đường dẫn/tên notebook, artifacts, runs và tài liệu được rà soát, đồng bộ về vị trí hiện tại.

## 8. Cổng hoàn tất Phase 0

- [x] Phạm vi sáu runs, cấu hình, schema, artifacts, train/resume/reuse, preprocessing, đánh giá và API package thống nhất xuyên tài liệu.
- [x] Bộ test specs và mã test trong kế hoạch khớp nhau.
- [x] Các mâu thuẫn tài liệu đã biết đã được xử lý; không còn mâu thuẫn chặn Phase 1.
- [x] Kết luận Phase 0 chỉ xác nhận tài liệu thiết kế đã sẵn sàng; **không** xác nhận code đã triển khai đúng.

**Kết quả Phase 0:** ☐ Chưa đạt  ☒ Đạt
**Ngày rà soát:** 2026-11-04
**Ghi chú / quyết định còn mở:** Không còn mục tài liệu nào chặn Phase 1. Việc rà soát call sites legacy và xác minh code được thực hiện trong các phase triển khai tương ứng.

---

Sau khi đạt cổng này, bắt đầu **Phase 1 — Hạ tầng kiểm thử và cấu hình dự án** theo [`implementation_plan.md`](./implementation_plan.md).
