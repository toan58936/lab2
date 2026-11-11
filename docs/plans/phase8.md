# Phase 8 — Chuẩn hóa cấu hình Notebook, GitHub và Google Drive

## Mục tiêu

Điều chỉnh code notebook để khai báo đúng nguồn code, dữ liệu, artifacts và cấu hình sáu run; sau đó kiểm chứng cấu hình mà không khởi chạy các thao tác thực nghiệm:

- Notebook và mã `going_modular/` được lấy từ phiên bản GitHub đã xác định.
- CIFAR-10 được quản lý trên Google Drive; có thể copy/giải nén vào Colab runtime để đọc nhanh.
- Checkpoint và kết quả cần lưu lâu dài được ghi về Drive, tách biệt theo từng run.
- Cấu hình có thể chọn đúng một trong sáu run và xác định rõ chế độ train/reuse/resume.

**Phạm vi Phase 8 này là chỉnh sửa code notebook và kiểm chứng cấu hình. Không mount Drive, clone GitHub, tải dữ liệu/weights hoặc chạy huấn luyện trong lượt triển khai code này.** Việc vận hành notebook trên Colab và huấn luyện thật thuộc các bước tiếp theo do người dùng thực hiện.

## Trạng thái sau lần chỉnh sửa notebook

- **Đã triển khai trong code:** sửa syntax cell imports; tách các đường dẫn; thêm cấu hình clone repository trên Colab và in revision; khai báo sáu `RUN_ID`/`RUN_SPECS`; kiểm tra cấu trúc CIFAR-10 và stage từ thư mục/archive bằng `download=False`; tạo bản đồ đường dẫn artifact theo run; thêm cảnh báo dừng sau bước cấu hình.
- **Đã kiểm tra local:** notebook JSON hợp lệ; 26 code cells compile; cả sáu cấu hình resolve trong thư mục tạm; logic stage thư mục và archive CIFAR-10 chạy với fixture giả trong thư mục tạm. Không có cell notebook nào được thực thi.
- **Chưa nghiệm thu:** chưa chạy bootstrap thật trên Colab/Drive; chưa xác nhận Drive write permission/dataset thật; các cell huấn luyện hiện có vẫn là pipeline cố định cho ba luồng và chưa được nối với `RUN_ID`/`RESUME`. Các mục còn thiếu được giữ `[ ]`; không đánh dấu toàn Phase hoàn tất.

## Bố trí tài nguyên mục tiêu

```text
GitHub (branch/commit đã chốt)
└── lab2/
    ├── going_modular/
    ├── notebooks/pretrained_models.ipynb
    ├── requirements.txt
    └── ...

Colab runtime (tạm thời)
├── /content/lab2/                 # Clone từ GitHub; import code từ đây
├── /content/cifar10_data/         # Bản làm việc của CIFAR-10
└── /content/runs/                 # TensorBoard logs tạm hoặc đồng bộ theo chính sách

Google Drive (lưu bền vững)
└── MyDrive/lab2/
    ├── data/
    │   ├── cifar-10-python.tar.gz     # Hoặc thư mục cifar-10-batches-py/
    └── artifacts/
        ├── v0_vgg16/
        ├── v1_resnet18/
        ├── v1_resnet18_finetune/      # run fine-tune duy nhất
        └── figures/
```

Trong mỗi thư mục run, quy trình huấn luyện sau này dự kiến lưu `config.json`, `history.json`, `best_model.pth`, `last_checkpoint.pth`, và `test_metrics.json` sau khi run hoàn tất. Phase này chỉ kiểm tra cấu hình thư mục/quyền truy cập, không tạo checkpoint hay metrics thực nghiệm.

## 1. Xác lập baseline và nguồn cấu hình

- [x] Ghi nhận các gap notebook trước khi sửa: setup dùng tên đường dẫn chưa khai báo, chưa có bootstrap GitHub, dữ liệu Drive chưa được stage, và cấu hình run chưa điều khiển các cell huấn luyện.
- [x] Khai báo repository `https://github.com/toan58936/lab2.git`, branch `main`, tùy chọn pin `GITHUB_COMMIT`, và in revision checkout trong cell setup. Việc xác minh checkout thật trên Colab vẫn đang chờ.
- [ ] Quy định notebook và thông số run nằm trong nội dung được quản lý phiên bản Git, không dùng bản notebook chỉnh sửa riêng trên Drive làm nguồn cấu hình không truy vết được.
- [ ] Không sửa hoặc xóa dataset/artifacts đang có trong repository/Drive.

## 2. Tách đường dẫn code, dữ liệu và kết quả

**File chính:** `notebooks/pretrained_models.ipynb` (cell setup/paths); cập nhật `README.md` nếu hướng dẫn hiện hành không còn khớp.

- [x] Thay giả định hiện tại rằng `going_modular/` phải tồn tại trong `MyDrive/lab2` bằng code bootstrap clone repository vào `/content/lab2` nếu thư mục chưa có.
- [x] Thêm kiểm tra thư mục đích là Git checkout, tùy chọn checkout commit cụ thể, in revision và đặt checkout vào `sys.path`. Chưa thực thi bootstrap hoặc xác minh origin/branch thực tế trên Colab.
- [x] Đặt các biến có vai trò riêng và tên rõ ràng, ví dụ:
  - `CODE_ROOT`: checkout GitHub trong Colab runtime.
  - `DRIVE_ROOT`: `/content/drive/MyDrive/lab2` (Drive folder name is case-sensitive).
  - `DRIVE_DATA_ROOT`: thư mục dữ liệu CIFAR-10 trên Drive.
  - `DATA_DIR`: thư mục dữ liệu mà DataLoader đọc.
  - `ARTIFACT_ROOT`: thư mục artifacts trên Drive.
  - `RUNS_ROOT`: nơi lưu TensorBoard logs theo chính sách đã chốt.
- [x] Thêm lệnh mount Drive trước khi thiết lập các đường dẫn Drive.
- [ ] Kiểm tra rõ Drive mounted/đọc-ghi được; hiện chưa có write-probe và chưa được xác minh trên runtime Colab.
- [x] Không dùng một biến `ROOT` chung cho source code, dataset, artifacts và logs.

## 3. Cấu hình CIFAR-10 trên Drive

**File liên quan:** notebook cell setup/dữ liệu; `going_modular/data_setup.py` chỉ thay đổi nếu kiểm thử chứng minh thật sự cần đổi hợp đồng.

- [x] Chốt vị trí nguồn dữ liệu trên Drive là `MyDrive/lab2/data/` (khớp tên thư mục hiển thị trên Drive).
- [x] Hỗ trợ code nhận dataset đã upload theo một trong hai dạng:
  - archive `cifar-10-python.tar.gz`; hoặc
  - thư mục đã giải nén `cifar-10-batches-py/` với các batch và `batches.meta`.
- [x] Kiểm tra file CIFAR-10 cần thiết trước khi tạo DataLoader; báo lỗi rõ khi thiếu/sai và không fallback sang download.
- [x] Thêm code copy thư mục/archive từ Drive sang `/content/cifar10_data` mà không sửa/xóa nguồn; kiểm thử cả hai dạng bằng fixture giả local.
- [x] Truyền `data_dir` tường minh và `download=False` cho DataLoader; xác minh code path tĩnh, chưa chạy DataLoader với CIFAR-10 thật.
- [x] Giữ nguyên seed, stratified split và quy tắc train/validation/test trong factory hiện có.
- [ ] Ghi rõ trong README quy trình upload dataset, cấu trúc thư mục mong đợi và sự khác nhau giữa bản nguồn Drive với bản runtime.
- [ ] Chạy staging thật từ Drive/Colab và kiểm tra một batch với `download=False` (thực hiện ở bước vận hành Colab của người dùng, không thuộc lượt sửa code).

## 4. Cấu hình artifacts và logs trên Drive

**File chính:** notebook cells cấu hình run/lưu kết quả.

- [x] Đặt `ARTIFACT_ROOT` trên Drive, tách khỏi `CODE_ROOT`, `DATA_DIR` và `RUNS_ROOT`.
- [x] Resolve thư mục và các đường dẫn artifact riêng theo `RUN_ID`; chưa khẳng định các cell huấn luyện cũ đã sử dụng những đường dẫn này.
- [ ] Kiểm tra Drive có thể tạo/ghi/xóa một probe file riêng do Phase tạo; hiện chưa có probe trong code và chưa kiểm tra runtime Drive.
- [x] Khai báo trong checklist cấu trúc và ý nghĩa artifacts dự kiến:
  - `config.json`: cấu hình run và môi trường.
  - `history.json`: metric train/validation theo epoch.
  - `best_model.pth`: weights được chọn theo validation.
  - `last_checkpoint.pth`: trạng thái resume tại ranh giới epoch.
  - `test_metrics.json`: metrics/predictions sau đánh giá cuối.
- [x] Chọn `RUNS_ROOT = MyDrive/lab2/runs` trên Colab; trên local dùng `<CODE_ROOT>/runs`. Chưa kiểm tra thao tác ghi trên Drive.
- [ ] Không tạo artifacts huấn luyện giả trong Phase này; chỉ xác minh đường dẫn và quyền truy cập.

## 5. Khai báo và kiểm tra cấu hình sáu run

**File chính:** notebook cell cấu hình run; thông số chuẩn theo `docs/plans/implementation_plan.md` và đặc tả cấu hình tổng thể.

- [x] Khai báo danh sách allowlist `RUN_ID` gồm 3 giá trị:
  - `v0_vgg16`
  - `v1_resnet18`
  - `v1_resnet18_finetune` (run fine-tune duy nhất)
- [x] Mỗi ID ánh xạ tường minh sang kiến trúc, weights khi khởi tạo, freeze/unfreeze, epochs, batch size, LR theo nhóm tham số, seed và dependency nếu có.
- [x] Với `v1_resnet18_finetune`, khai báo nguồn khởi tạo là `best_model.pth` của `v1_resnet18`; optimizer là cấu hình mới cho phase fine-tune.
- [x] Cell cấu hình chỉ resolve và in một run; bản thân cell này không gọi train, inference Test hoặc lặp qua sáu run.
- [x] Kiểm thử local cả sáu ID trong môi trường tạm; xác nhận arch, batch size, epochs, LR và artifact paths được resolve đúng, không gọi train/evaluate.
- [ ] Hoàn thiện kiểm tra collision/artifacts, đầy đủ field của từng spec và dependency cho mọi mode. Hiện mới validate ID/mode và kiểm tra checkpoint dependency khi `RUN_MODE="train"`.
- [x] Mỗi lần chuẩn bị chỉ chọn một run; hiển thị rõ tham số, source checkpoint (nếu có), data path và artifact path trước khi cho phép tiếp tục.

## 6. Khai báo rõ `RUN_MODE` và `RESUME` cho phase huấn luyện sau

- [x] Khai báo và validate `RUN_MODE` (`train`/`reuse`) cùng quy tắc `RESUME=True` chỉ hợp lệ trong `train`.
- [ ] Chỉ khai báo intent hiện tại; chưa triển khai các bảo đảm runtime:
  - `RESUME=False`: dừng nếu run artifacts đã có; không ghi/xóa.
  - `RESUME=True`: yêu cầu checkpoint tồn tại và phase config khớp.
  - `RUN_MODE="reuse"`: xác nhận đủ artifacts và không huấn luyện.
- [x] Khai báo dependency checkpoint baseline cho run fine-tune; kiểm tra file khi chọn run train. Kiểm tra trên Drive thật còn chờ.
- [x] Đã thêm cảnh báo dừng sau phần cấu hình; không thực thi notebook trong lượt sửa code.
- [ ] Tách hoặc đặt cổng rõ cho cell thực nghiệm để chạy tuần tự phần setup/config không vô tình bắt đầu huấn luyện.

## 7. Sửa lỗi và đối chiếu API notebook

**Files:** `notebooks/pretrained_models.ipynb`, các module `going_modular/` chỉ khi thật sự cần.

- [x] Sửa code cell import đang có `try/except` rỗng và các lỗi cú pháp khác được phát hiện khi compile cell.
- [ ] Đối chiếu mọi lời gọi notebook với API hiện tại: `build_model`, `train`, `save/load config/history/checkpoint`, `run_inference` và các hàm evaluate.
- [ ] Thay các lời gọi legacy còn tồn tại trước khi notebook được dùng làm entry point cho phase huấn luyện.
- [ ] Giữ đánh giá Test sau khi run hoàn tất; không đưa Test vào quá trình chọn cấu hình/checkpoint.
- [ ] Không refactor ngoài phạm vi cấu hình notebook/Drive nếu không có lỗi trực tiếp ngăn cản Phase 8.

## 8. Kiểm chứng notebook ở chế độ setup-only

- [x] Parse notebook JSON và xác nhận schema notebook hợp lệ.
- [x] Compile toàn bộ code cells sau khi xử lý riêng IPython magic nếu có; không còn lỗi syntax.
- [x] Kiểm thử cấu hình resolve cả sáu `RUN_ID` trong môi trường tạm; xác nhận thông số/paths và không gọi train/evaluate.
- [x] Kiểm thử code stage CIFAR-10 bằng fixture giả trong thư mục tạm cho cả thư mục giải nén và archive; không tải dataset thật.
- [ ] Chạy lại smoke/full fast suite sau lần sửa notebook này. Gần nhất trước đó, suite đầy đủ báo `130 passed in 55.79s`; đây là baseline gần nhất, không phải kết quả sau chỉnh sửa notebook.
- [ ] Trên Colab, xác nhận clone đúng revision GitHub, import từ checkout, mount Drive, nhận diện CIFAR-10 đã upload/stage, ghi được probe và resolve artifact path.
- [ ] Nếu người dùng kiểm tra DataLoader trên Colab, chỉ lấy một batch từ dữ liệu Drive đã có với `download=False`; không tải CIFAR-10 hoặc pretrained weights như một phần của lượt sửa code này.
- [x] Static/local checks xác nhận cell cấu hình không gọi huấn luyện hoặc tạo checkpoint/metrics. Các cell thực nghiệm bên dưới vẫn chứa lệnh train cũ; không chạy tuần tự qua chúng trong Phase 8 hiện tại.
- [ ] Sau khi các mục code và runtime phù hợp được xác minh, ghi lại revision/commands/results. Chưa đánh dấu hoàn tất toàn Phase ở trạng thái hiện tại.

## Ngoài phạm vi Phase 8

- Chạy huấn luyện bất kỳ run nào, kể cả smoke training trên tập nhỏ.
- Resume run thật, đánh giá trên tập Test, chọn/tinh chỉnh hyperparameter dựa trên metrics.
- Tạo `best_model.pth`, `last_checkpoint.pth`, `test_metrics.json`, bảng/biểu đồ kết quả thực nghiệm.
- Tuyên bố hoàn tất sáu runs hoặc nghiệm thu kết quả Phase 8 thực nghiệm.
- Thay đổi kiến trúc mô hình, schema dữ liệu/artifacts hoặc chiến lược thí nghiệm đã chốt.
- Xóa hoặc ghi đè dataset/artifacts đã tồn tại trên Drive.

## Cổng nghiệm thu Phase 8 — Notebook configuration

- [ ] Chạy trên Colab để xác nhận bootstrap/import từ GitHub và revision thực tế.
- [ ] Xác nhận CIFAR-10 thật từ Drive có thể nhận diện/stage, không fallback sang download.
- [ ] Kiểm tra Drive read/write và artifact/log paths.
- [x] Sáu cấu hình resolve đúng với fixture tạm; điều này không chứng minh các cell huấn luyện đã dùng cấu hình đó.
- [x] Notebook JSON/Python syntax hợp lệ; logic stage archive/folder qua fixture giả.
- [ ] Hoàn tất API audit và cổng dừng trước cell train/evaluate; hiện các cell cũ vẫn có thể train nếu chạy.
- [x] Trong lượt sửa code không có run nào được huấn luyện/đánh giá và không tạo checkpoint/metrics thực nghiệm.

## Phase tiếp theo — Thực nghiệm sáu run

Sau khi các mục Phase 8 còn mở được xử lý/được người dùng xác nhận trên Colab, chuyển sang phase riêng cho train/resume/reuse và đánh giá đủ sáu run theo thứ tự/phụ thuộc trong `docs/plans/implementation_plan.md`. Chỉ ở phase thực nghiệm đó mới tạo checkpoint, history, test metrics, bảng so sánh và báo cáo kết quả.

> **Lưu ý đồng bộ kế hoạch:** `implementation_plan.md` hiện đang gọi Phase 8 là “Huấn luyện và đánh giá đầy đủ (6 runs)”. Khi áp dụng checklist này, cần đổi roadmap để Phase 8 là cấu hình notebook/Drive và chuyển nội dung huấn luyện sáu run sang phase kế tiếp; không được để hai tài liệu mang hai phạm vi Phase 8 khác nhau.
