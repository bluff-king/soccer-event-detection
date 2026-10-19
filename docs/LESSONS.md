# Bài học từ các lỗi đã gặp

Bản chữ, mỗi mục tự đủ nghĩa. Bản có hình (dữ liệu thật, sai và đúng đặt cạnh nhau) ở `docs/loi/`, sinh lại bằng
`python scripts/build_lessons.py`.

## 2026-10-13 · Ablation cửa sổ độ trễ theo lớp thật ra train trên nhãn cũ
[trang có hình](loi/2026-10-13_class-delays-dung-chung-du-lieu.html)

- **Triệu chứng:** `imp_class_delays` (vòng 1, Colab) báo Penalty F1 = 0 và evt F1 thấp hơn baseline; README kết
  luận "chưa kết luận được".
- **Nguyên nhân gốc:** nhãn có độ trễ được tạo một lần khi `prepare` vào `data/processed`. Config chỉ đổi
  `delay_per_class` mà không đổi `processed_dir`, nên train vẫn đọc nhãn [-5, 15]; chỉ phần quy đổi thời điểm sự kiện
  dùng độ trễ mới. Thí nghiệm không thử cái nó ghi.
- **Chẩn đoán sai đã sửa:** tôi từng viết "độ trễ lệch đẩy dự đoán Penalty ra ngoài ±30 s nên F1 = 0". Tái hiện trên
  xác suất của `v2_baseline` bác bỏ: độ trễ −27.5 s cho Penalty F1 test 0.235, cao hơn 0.118 với +5 s. Vì sao Colab
  ra 0 vẫn **chưa phân định** (ứng viên: checkpoint epoch 0 chưa học Penalty); Vũ kiểm được bằng `history.json` /
  `metrics.json` của run này trên Drive.
- **Bài học:** thiết lập nào "nướng" vào dữ liệu đã xử lý thì phải đi cùng dữ liệu và được kiểm khi đọc
  (`manifest.json → labeling`, `check_labeling`). Giải thích một con số phải tái hiện, không chỉ suy luận.

## 2026-10-13 · Id trận dùng `\` trên Windows làm mọi phép so khớp trả về rỗng
[trang có hình](loi/2026-10-13_id-tran-windows.html)

- **Triệu chứng:** trên Windows, bảng ngôn ngữ khớp 0/1100 hiệp, bộ lọc trận Kaggle trùng valid/test loại 0 trận
  (Linux: 89), nguồn `asr_views` chỉ ra một biến thể Whisper. Không có lỗi nào được báo.
- **Nguyên nhân gốc:** id trận tạo bằng `str(Path)` (và `os.path.join` trong gói SoccerNet) dùng `\` trên Windows,
  rồi được so như chuỗi với dữ liệu dùng `/`.
- **Bài học:** id dùng để so khớp chuỗi phải chuẩn hoá (`as_posix()`), kể cả id lấy từ thư viện ngoài. Phép lọc chống
  rò rỉ phải có số kỳ vọng để đối chiếu.

## 2026-10-13 · Chọn checkpoint theo F1 mức đoạn làm evt F1 tụt còn 0.389
[trang có hình](loi/2026-10-13_chon-checkpoint-muc-doan.html)

- **Triệu chứng:** `v2_baseline` có F1 mức đoạn ngang baseline nhưng evt F1 valid 0.389 (692 FP Goal, phần lớn là bình
  luận sau bàn thắng).
- **Nguyên nhân gốc:** checkpoint được chọn theo F1 mức đoạn trên tập con valid xáo trộn, không phải theo F1 mức sự
  kiện trên cả trận; một checkpoint muộn, tự tin hơn thắng ở proxy nhưng thua ở mục tiêu. Chọn theo evt F1 trên 45
  trận nguyên vẹn: 0.389 → 0.522 valid, test 0.403 → 0.497.
- **Chẩn đoán sai trên đường đi:** "lưới ngưỡng dừng ở 0.95" — mở rộng lưới không đổi gì.
- **Bài học:** chỉ số chọn model phải là chỉ số mục tiêu, đo đúng như đánh giá cuối. Loại trừ giả thuyết bằng phép
  đo trước khi sửa.

## 2026-10-14 · Hai job trên một GPU 4 GB: chậm hàng chục lần, và một chẩn đoán sai
[trang có hình](loi/2026-10-14_hai-job-mot-gpu.html)

- **Triệu chứng:** (1) 500 bước đầu mất ~10,500 s thay vì ~120 s; (2) ModernBERT train ~6 mẫu/s thay vì ~91.
- **Nguyên nhân gốc:** hai tiến trình train cùng trên GPU 4 GB, cả hai tràn sang bộ nhớ chung (WDDM). (1) `TaskStop`
  chỉ dừng lớp shell ngoài, runner hàng đợi cũ vẫn sống → hai runner; (2) một job khác của Vũ bắt đầu một phút sau.
- **Chẩn đoán sai:** tôi kết luận "phân mảnh bộ nhớ do ModernBERT bỏ padding" và giảm batch; batch nhỏ hơn mà bộ nhớ
  không giảm mới lộ ra là sai. Đã hoàn tác.
- **Bài học:** trên GPU nhỏ mỗi lúc một job (runner giữ khoá và chờ `nvidia-smi` trống). Dừng tiến trình nền phải
  kiểm cả cây tiến trình. Tốc độ tụt thì liệt kê ai đang dùng GPU trước khi nghĩ tới nguyên nhân trong model.

## 2026-10-15 · Đổi seed lệch 0.073 evt F1: dừng sớm lúc lr còn ở đỉnh
[trang có hình](loi/2026-10-15_seed-dung-som.html)

- **Triệu chứng:** cùng cấu hình `v2_delays_views_hardneg`, test evt F1 0.576 (seed 42) và 0.503 (seed 1) — lớn hơn
  phần lớn khác biệt giữa các cấu hình đang so.
- **Nguyên nhân gốc:** lịch lr tính cho 5 epoch nhưng dừng sớm sau 4 lần đánh giá không cải thiện, nên không run nào
  xong epoch đầu; checkpoint được chọn khi lr ở đỉnh, nơi evt F1 nhảy ±0.05 giữa hai lần đánh giá → chọn trúng đỉnh
  nhiễu. Tập trận dùng để chọn checkpoint còn đổi theo seed.
- **Bài học:** đo phương sai giữa seed của chính recipe trước khi so cấu hình. Chọn checkpoint trên tín hiệu nhiễu
  cần lr đã giảm (vòng 3: 2 epoch, lr về 0, không dừng sớm, 3 seed, tập chọn cố định).
