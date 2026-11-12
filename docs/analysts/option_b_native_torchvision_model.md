# Phân tích hướng triển khai B — Model torchvision kiểu gốc, giữ orchestration hiện tại

> **Ngày:** 2026-11-11  
> **Nhánh đề xuất:** `toan3` (đã tách từ `toan2` sau commit `5b674b1`)  
> **Phạm vi:** Phân tích triển khai; chưa thay đổi code.

## 1. Mục tiêu và chốt phạm vi

**Lựa chọn B:** ưu tiên code sát `docs/yeucaugoc.md` (tải/adapt mô hình pre-trained theo cách trực tiếp từ `torchvision.models`) để dễ thuyết trình, **chấp nhận không dùng lại artifacts/checkpoint cũ** và **train lại 3 run** theo format mới.

**Nguyên tắc bắt buộc:** giữ các chức năng hiện tại ở tầng điều phối/hạ tầng:
- `RUN_SPECS`, `RUN_ID`/`RUN_IDS`, `RUN_PLAN`, `DEFAULT_RUN_ACTION`, `FORCE_RETRAIN` (notebook)
- Vòng lặp train/val, chọn best, checkpoint/resume: `going_modular/engine.py`
- Lưu/nạp artifacts & tiện ích: `going_modular/utils.py` (config/history/best/last checkpoint, seed/device, dated output/log helpers)
- Data/eval/logs: `going_modular/data_setup.py`, `going_modular/evaluate.py`, TensorBoard, cấu trúc `artifacts/`, `outputs/`, `logs/`

**Cách tiếp cận:** **chỉ đổi implementation và format `state_dict` ở tầng model/builder**, giữ “lớp cấu hình/builder” và contract gọi từ notebook/engine; **không** đưa toàn bộ logic (train loop, resume, lưu artifact, đánh giá) về notebook thuần.

## 2. Hiện trạng (baseline) và khác biệt với hướng B

### 2.1. Hiện trạng
- `going_modular/model_builder.py` định nghĩa `MyModel(nn.Module)` (wrapper): `self.backbone` (torchvision model với `fc`/`classifier` thay bằng `Identity`) + `self.head` (Linear/Dropout+Linear cho CIFAR-10), `forward = head(backbone(x))`.
- `build_model(cfg: ModelConfig) -> nn.Module` trả `MyModel`; `get_param_groups`, `set_train_mode`, `ARCH_SPECS`, `count_params` hoạt động trên cấu trúc wrapper (`model.head`, `model.backbone.layer4`, `model.backbone.bn1`, `model.backbone.features`, ...).
- `state_dict` keys có prefix `backbone.*`, `head.*` (khác torchvision gốc).
- Notebook: `build_model_for_run` dựng model theo `action` (`train` mới dùng ImageNet/baseline; `resume/reuse` dựng `weights=None` rồi nạp checkpoint của run), Cell 11 in `model_selected.head`, `torchinfo_summary` khám phá kiến trúc 2 model.
- `engine.py`, `utils.py`, `evaluate.py` thao tác theo `nn.Module`/`state_dict` tổng quát (phù hợp để giữ nguyên khi đổi model, miễn interface ổn định).
- Tests phụ thuộc cấu trúc hiện tại: `tests/test_model_builder.py` (truy cập `model.backbone.*`, `model.head`, `ARCH_SPECS` head, param counts VGG16 wrapper `14,965,578/250,890`), `tests/test_engine.py` (E6 kiểm tra `model.backbone.bn1.running_mean`), `tests/test_smoke.py` export `MyModel`.

### 2.2. Mục tiêu “kiểu gốc” (sát `yeucaugoc.md`)
- Tải model từ `torchvision.models` (ưu tiên API hiện hành `weights=VGG16_Weights.IMAGENET1K_V1` / `ResNet18_Weights.IMAGENET1K_V1` thay vì `pretrained=True` đã lỗi thời, nhưng ngữ nghĩa tương đương).
- Adapt trực tiếp: thay lớp phân loại cuối (`resnet18.fc`, `vgg16.classifier[6]`) cho 10 lớp CIFAR-10; đóng băng backbone (feature extraction) hoặc mở khóa N block cuối (fine-tune) trên chính đối tượng torchvision.
- Cấu trúc/model object gần ví dụ gốc; `torchinfo.summary` hiển thị kiến trúc torchvision tự nhiên.

### 2.3. Khác biệt cốt lõi
| Khía cạnh | Hiện tại (wrapper) | Hướng B (kiểu gốc) |
|---|---|---|
| Kiểu trả về | `MyModel` (`backbone`, `head`) | `torchvision.models` đã adapt (`ResNet`/`VGG`) |
| Head ResNet-18 | `backbone.fc=Identity` + `head=Linear(512,10)` | `model.fc = Linear(512,10)` |
| Head VGG16 | `backbone.classifier=Identity` + `head=Linear(25088,10)` (wrapper) | Thay `model.classifier[6] = Linear(4096,10)` (giữ `features/avgpool/classifier[0..5]`) — *khuyến nghị, xem 4.1* |
| `state_dict` keys | `backbone.*`, `head.*` | `conv1/layer*/fc.*`, `features.*`, `classifier.*` |
| Truy cập nội bộ | `model.backbone.bn1`, `model.head` | `model.bn1`, `model.fc` / `model.classifier[6]` |
| Tương thích artifacts cũ | Có | **Không** (phải train lại) |

## 3. Chức năng giữ nguyên (không thay đổi bản chất)

| Thành phần | Trạng thái | Ghi chú |
|---|---|---|
| `engine.train/train_step/evaluate_step` | Giữ nguyên logic | API/state_dict-generic; `verbose`, `train_mode_fn`, `resume`, `checkpoint_dir`, `expected_phase_config` vẫn dùng như hiện tại. |
| `utils.py` save/load/checkpoint/history/config/metrics, seed/device, dated dirs | Giữ nguyên | Chỉ nội dung checkpoint (`state_dict`) đổi format theo model mới. |
| `data_setup.py`, `evaluate.py` | Giữ nguyên | Nhận `nn.Module` bất kỳ. |
| Notebook: `RUN_SPECS/PLAN/FORCE_RETRAIN`, vòng lặp per-run, artifact guards theo action, TensorBoard, dataloader per-run, eval/compare flow | Giữ hành vi | Cần chỉnh điểm gọi/builder và hiển thị head (xem 5.2). |
| `ModelConfig` (fields, `to_dict/from_dict`), `build_model(cfg)` contract | **Giữ contract** | Cho phép thay implementation bên trong. |

## 4. Quyết định thiết kế cần chốt trước khi code

### 4.1. Cách thay head VGG16 (ảnh hưởng lớn đến param count & state_dict)
- **Phương án G1 (khuyến nghị, sát gốc):** giữ nguyên `features`, `avgpool`, `classifier` (Dropout/ReLU/Linear 25088→4096→4096), **chỉ thay `classifier[6]`** thành `Linear(4096, 10)` (có thể kèm Dropout nếu spec yêu cầu).  
  - Ưu: sát `yeucaugoc.md`/torchvision transfer-learning chuẩn; `state_dict`/cấu trúc quen thuộc; feature extraction trainable ~40,970 (head Linear 4096×10+10) thay vì ~250,890.
- **Phương án G2:** thay toàn bộ `classifier` bằng head nhỏ gọn. Ít sát ví dụ gốc, key/param khác torchvision chuẩn.

> **Cần chốt:** G1 hay G2. (Tài liệu này giả định **G1** trừ khi nêu khác; mọi số liệu VGG16 trong tests/docs phải cập nhật theo phương án được chọn.)

### 4.2. `MyModel` class và export
- Giữ class `MyModel` chỉ như alias/tương thích ngược (rủi ro gây nhầm “wrapper”) hay **xóa** và để `build_model` trả model torchvision gốc?
- `__init__.py`/`tests/test_smoke.py` hiện export/kiểm tra `MyModel` (contract 13 symbols). Cần quyết định: bỏ `MyModel` khỏi contract hay thay bằng symbol khác (ưu tiên **bỏ** nếu không còn wrapper, đồng thời cập nhật contract/docs).

### 4.3. `state_dict` và artifacts cũ
- **Chốt B:** không map key, không load checkpoint cũ. Dọn hoặc bỏ qua `artifacts/<run>/{best_model.pth,last_checkpoint.pth}` (và metrics gắn với model cũ) trước khi chạy format mới; train lại `v0_vgg16`, `v1_resnet18`, `v1_resnet18_finetune`.
- `history/config` cũ: có thể không tương thích ngữ nghĩa (param counts, kiến trúc mô tả) — khuyến nghị coi là thuộc format cũ, không dùng lẫn.

### 4.4. API/weights
- Dùng `weights=...` (torchvision hiện hành) thay `pretrained=True` trong implementation; có thể ghi chú trong notebook/doc là tương đương pretrained ImageNet theo `yeucaugoc.md`.

## 5. Phạm vi thay đổi theo thành phần

### 5.1. `going_modular/model_builder.py` (trọng tâm)
- **Xây dựng model:** trả về instance torchvision đã adapt (không wrapper `backbone/head` riêng) hoặc wrapper mỏng tương thích key gốc — ưu tiên **model gốc**.
  - ResNet-18: `models.resnet18(weights=...)`; thay `model.fc = Linear(in_features, num_classes)` (hoặc Dropout+Linear theo `cfg.dropout`).
  - VGG16 (G1): `models.vgg16(weights=...)`; thay `model.classifier[6] = Linear(in_features, num_classes)` (và chèn Dropout nếu cần, giữ nhất quán key).
- **Freeze/unfreeze:** thao tác trực tiếp trên `model.parameters()`, `model.fc`/`model.classifier[6]`, `model.layer1..4`, `model.features` (blocks). Logic thứ tự: validate → tạo/load → (freeze backbone) → thay head → unfreeze `unfreeze_last_n` block cuối.
- **`ARCH_SPECS`/registry:** cập nhật `get_head`/`set_head`/`blocks`/`num_blocks` trỏ đúng cấu trúc gốc (vd `get_head: lambda m: m.fc`, blocks ResNet `layer1..4`; VGG16 blocks theo `features` kết thúc `MaxPool2d` nhưng unfreeze áp lên `model.features`/submodules tương ứng).
- **`get_param_groups`:** xác định head params từ head thực tế (`fc` hoặc `classifier[6]`/head module) thay vì `model.head`.
- **`set_train_mode`:** giữ hành vi BatchNorm freeze đúng với param bị đóng băng (duyệt `model.modules()` như hiện tại là ổn, nhưng đường dẫn truy cập trong tests đổi từ `model.backbone.bn1` sang `model.bn1`).
- **`count_params`:** giữ hàm; **giá trị kỳ vọng thay đổi** (đặc biệt VGG16 theo G1).
- **`_vgg16_conv_blocks`/helper:** điều chỉnh theo `model.features` (không còn `model.backbone.features`).
- **`_resolve_weights`/validation:** giữ; `build_model` phải `resolve` weights trước `MyModel.__init__` kiểu cũ (hiện đã tách) — đảm bảo validation không tải nặng ngoài ý muốn (tests patch `_resolve_weights`).
- **Exports:** xem xét bỏ `MyModel` nếu không còn dùng; giữ `ModelConfig`, `build_model`, `count_params`, `get_param_groups`, `set_train_mode`, `ARCH_SPECS`.

### 5.2. `notebooks/pretrained_models.ipynb`
- **Cell 11:**
  - Cập nhật `build_model_for_run`/hiển thị: in head đúng vị trí mới (`model.fc` cho ResNet, `model.classifier[6]` hoặc `model.classifier` cho VGG16) thay `model_selected.head`.
  - `torchinfo_summary` sẽ tự phản ánh cấu trúc gốc (tốt cho thuyết trình, sát ví dụ); kiểm tra `depth`, `row_settings` vẫn hiển thị rõ các lớp.
  - Giữ nguyên luồng `action/train|resume|reuse`, fine-tune nạp baseline khi train mới, không nạp baseline khi resume (engine nạp checkpoint run).
- **Các cell train/eval:** không đổi logic gọi `engine.train`, `evaluate`, save/load (đã generic) — chỉ chạy lại sau khi model đổi.
- **Markdown:** cập nhật mô tả “wrapper/backbone+head” nếu còn, làm rõ code sát `torchvision` gốc + giữ orchestration.

### 5.3. Tests
- **`tests/test_model_builder.py`:**
  - Đổi truy cập: `model.backbone.*` → `model.*`, `model.head` → `model.fc`/`model.classifier[6]` (hoặc helper theo arch), `m_pretrained.backbone.conv1.weight` → `m_pretrained.conv1.weight`.
  - Cập nhật `expected` trong `test_t4_param_counts` theo cấu trúc mới (ResNet-18 thường giữ tương đương 11,181,642/5,130; VGG16 thay đổi đáng kể theo G1).
  - Điều chỉnh test kiểm tra head type khi dropout (Sequential vị trí mới), và các giả định `get_head`.
  - Giữ/điều chỉnh mock `_resolve_weights` cho T8c–T8e theo implementation mới.
- **`tests/test_engine.py`:** E6 đổi `model.backbone.bn1` → `model.bn1` (hoặc tra cứu BN đầu theo kiến trúc).
- **`tests/test_smoke.py`:** cập nhật contract exports nếu bỏ/đổi `MyModel`; giữ kiểm tra các API cốt lõi (`build_model`, `train`, `evaluate_step`, `create_train_val_test_dataloaders`, `set_seed`, `get_device`, ...).
- **Kỳ vọng:** chạy `pytest tests/ -m "not slow"` trước; nhóm `slow` (tải ImageNet weights) cần mạng/cache.

### 5.4. Docs và metadata
- `docs/define&designt/02_model.md`, `00_lab2_description.md`: mô tả kiến trúc/model trả về, API (`MyModel`?), state_dict keys, param counts VGG16/ResNet, cách adapt head (G1/G2).
- `docs/analysts/notebook_status_checklist.md`, `docs/plans/run-orchestration-checklist-plan.md`, `docs/plans/phase8.md`, `docs/plans/implementation_plan.md`, `README.md`: cập nhật ghi chú model format/state_dict, bỏ ngụ ý tương thích checkpoint cũ, nêu rõ cần train lại 3 run.
- `docs/yeucaugoc.md`/phần đối chiếu: có thể bổ sung ghi chú “implementation dùng `weights=...` tương đương pretrained, code adapt theo kiểu trực tiếp”.

### 5.5. Artifacts/outputs/logs (vận hành)
- **Không tương thích:** checkpoint/model `.pth` và các file gắn chặt format cũ trong `artifacts/<v0_vgg16|v1_resnet18|v1_resnet18_finetune>/` không dùng để `resume/reuse/load`.
- **Hành động đề xuất (khi bắt đầu chạy format mới):** đặt `FORCE_RETRAIN=True` hoặc dọn artifacts của 3 run (tùy chính sách), rồi chạy lại theo thứ tự `v0_vgg16`, `v1_resnet18`, `v1_resnet18_finetune` (fine-tune phụ thuộc baseline mới). `test_metrics.json`/outputs cũ nên xem là thuộc kết quả format cũ.
- Không commit `.pth`, không xóa dữ liệu ngoài ý muốn trên Drive/Colab.

## 6. Lộ trình triển khai đề xuất (trên `toan3`)

1. **Chốt quyết định 4.1–4.3** (VGG16 G1/G2, xử lý `MyModel`/contract, chính sách dọn artifacts).
2. **Refactor `model_builder.py`**: implement model gốc + giữ `ModelConfig/build_model/get_param_groups/set_train_mode/count_params/ARCH_SPECS` tương thích; tự kiểm tra nhanh shape/grad/param.
3. **Cập nhật tests** (`test_model_builder.py`, `test_engine.py`, `test_smoke.py`) theo cấu trúc/param mới; chạy `pytest tests/test_model_builder.py tests/test_engine.py tests/test_smoke.py -m "not slow"`.
4. **Cập nhật notebook** (hiển thị head, markdown Section 2/3, đảm bảo `build_model_for_run` + `train/resume/reuse` đúng); compile code cells.
5. **Cập nhật docs** (02_model, 00, README, checklist, plans liên quan).
6. **Xác nhận vận hành:** dọn/force retrain artifacts 3 run, chạy lại `v0_vgg16` → `v1_resnet18` → `v1_resnet18_finetune`, đánh giá và so sánh (ngoài phạm vi edit code thuần, cần môi trường dữ liệu/GPU).
7. Commit theo đơn vị hợp lý, push `toan3`, mở PR/merge khi nghiệm thu.

## 7. Rủi ro và cách giảm thiểu

| Rủi ro | Ảnh hưởng | Giảm thiểu |
|---|---|---|
| VGG16 param/head sai kỳ vọng (G1 vs G2) | Test/docs/so sánh sai | Chốt 4.1 trước; tính `count_params` thực tế làm chuẩn cập nhật. |
| `get_param_groups`/unfreeze nhầm block (đặc biệt VGG16) | Fine-tune sai, LR group sai | Test tường minh trainable ids theo block/head; so sánh với spec `unfreeze_last_n`. |
| BN freeze lệch (`set_train_mode`) | Train/resume kém ổn định | Giữ rule “BN có params frozen → eval”; test E6/T9/T10 cập nhật path. |
| Vô tình load checkpoint cũ | Lỗi key mismatch/runtime | Guard/raise rõ khi detect format không khớp (tùy chọn), tài liệu hóa “phải train lại”. |
| Regression engine/utils/eval | Mất resume/reuse | Không đổi engine/utils; chạy test engine + smoke + notebook compile. |
| Notebook/Colab path & Drive | Mất artifact khi retrain | Chỉ dọn run cần thiết, có `FORCE_RETRAIN`/xác nhận, không động vào `data/`. |

## 8. Tiêu chí hoàn thành (Definition of Done) cho hướng B

- `build_model` trả model torchvision đã adapt (sát gốc), thay head/freeze/unfreeze đúng cho `v0_vgg16`, `v1_resnet18`, `v1_resnet18_finetune`.
- Giữ nguyên hành vi `RUN_PLAN`/`train|resume|reuse`/`FORCE_RETRAIN`, `engine` checkpoint-resume, lưu `config/history/best/last`, TensorBoard, outputs/metrics, đánh giá test cuối.
- `pytest tests/test_model_builder.py tests/test_engine.py tests/test_smoke.py -m "not slow"` pass; notebook code cells compile.
- Docs/spec phản ánh format mới, param counts đúng, và nêu rõ: artifacts cũ không tương thích, cần train lại 3 run.
- (Nghiệm thu thực nghiệm) 3 run huấn luyện/đánh giá thành công với format mới, fine-tune khởi tạo từ baseline `v1_resnet18` mới.

## 9. Câu hỏi cần xác nhận

1. Chốt **VGG16 head**: **G1** (thay `classifier[6]`, giữ 4096 features, sát gốc) hay **G2** (thay gọn classifier)?
2. Xử lý **`MyModel`**: xóa khỏi API/contract hay giữ tên tương thích?
3. Có cần thêm **cơ chế phát hiện/migrate** checkpoint cũ (báo lỗi rõ ràng) hay chỉ tài liệu hóa + dọn tay/`FORCE_RETRAIN`?
4. Có cần cập nhật **`06_pytorch_transfer_learning.ipynb`** tham chiếu hay chỉ notebook chính và docs lab?
