# Phát hiện sự kiện bóng đá từ lời bình luận (Commentary Event Detection)

## Bối cảnh
Phát hiện các sự kiện quan trọng trong trận bóng (bàn thắng, thẻ phạt, penalty) chỉ từ transcript lời bình luận
(sinh ra bằng ASR), thay vì xử lý video — rẻ hơn và dễ triển khai hơn. Đầu ra dùng để tự động tạo highlight.

## Dữ liệu (công khai)
- **SoccerNet-Echoes**: transcript ASR lời bình luận các trận SoccerNet (có timestamp từng segment).
- **SoccerNet Labels-v2**: nhãn sự kiện có timestamp (`Labels-v2.json` mỗi trận). Chỉ dùng file nhãn, KHÔNG cần video.
- Repo baseline tham khảo: dự án **Event-Detection** trên SoccerNet-Echoes (transformer phân loại chuỗi, backbone
  `xlm-roberta-base`, cửa sổ ngữ cảnh 3 segment, có tính độ trễ phản ứng của bình luận viên).
  → Tìm repo này trên GitHub (gợi ý: các repo liên quan SoccerNet-Echoes). Nếu không tìm thấy, tự cài đặt lại
  pipeline theo mô tả ở đây và ghi rõ trong README.

## Vấn đề đã biết từ lần chạy trước
- **Precision thấp**: mô hình báo nhầm sự kiện nhiều (false positive).
- **Dữ liệu quá ít**: khuyến nghị khoảng 10k mẫu mỗi nhãn, nhưng SoccerNet chỉ có vài trăm đến vài nghìn sự kiện
  mỗi lớp (Penalty ít nhất). → Mở rộng dữ liệu là ưu tiên số 1.

## Mở rộng dữ liệu (làm trước khi huấn luyện)
Khảo sát, kiểm chứng (link, license, số lượng mẫu thực tế mỗi lớp) và tích hợp các nguồn sau.
Báo cáo bảng thống kê trước khi dùng:
1. **SoccerNet-Caption / MatchTime / SoccerReplay-1988**: lời bình luận dạng text gắn loại sự kiện và timestamp,
   số trận nhiều hơn. Kiểm tra trùng trận với SoccerNet-Echoes để không rò rỉ giữa train và test.
2. **Kaggle "Football Events"** (text tường thuật hàng trăm nghìn sự kiện, nhiều goal và card): văn bản theo
   khuôn mẫu, nên phải **bỏ các từ khóa lộ nhãn và tỉ số** để mô hình không học vẹt. Chỉ dùng cho pre-train hoặc
   bổ sung, không dùng để đánh giá.
3. **Dữ liệu tổng hợp bằng LLM**: paraphrase các câu sự kiện thật sang văn nói của bình luận viên, có nhiễu kiểu ASR
   (sai chính tả, thiếu dấu câu, câu bị cắt). Ưu tiên lớp ít mẫu (Penalty). Đánh dấu nguồn từng mẫu.
4. **Hard negatives**: các câu nhắc tới "goal/card/penalty" nhưng không phải sự kiện xảy ra (nói về replay, nhắc
   bàn thắng cũ, "almost a goal", "should be a card"). Đây là nguồn chính gây false positive.

Nguyên tắc: **tập validation/test chỉ gồm transcript ASR thật của SoccerNet-Echoes**, chia theo trận.
Dữ liệu ngoài chỉ thêm vào tập train. Ablation bắt buộc: chỉ Echoes, Echoes + từng nguồn, Echoes + tất cả.

## Chẩn đoán precision
- Phân tích các mẫu false positive theo nhóm: lệch thời gian (dự đoán đúng sự kiện nhưng sai cửa sổ), replay hoặc
  nhắc lại, nhầm lớp, nhiễu nhãn.
- Đánh giá ở mức sự kiện, không chỉ mức segment: nhiều FP thực chất là cùng một sự kiện bị báo ở nhiều cửa sổ
  liền nhau → cần gộp (NMS theo thời gian).
- Vẽ đường precision–recall theo threshold cho từng lớp, chọn ngưỡng trên tập validation.
- Thử pipeline 2 tầng: tầng 1 lọc ứng viên (recall cao), tầng 2 xác minh (mô hình lớn hơn hoặc LLM) để tăng precision.

## Bài toán
Mỗi mẫu = một cửa sổ transcript (segment hiện tại + ngữ cảnh lân cận) → 1 trong 4 lớp:
`No-Event`, `Goal`, `Card`, `Penalty`.
- Gộp các loại thẻ (Yellow card, Red card, Yellow->red card) thành `Card`.
- Gán nhãn có tính **độ trễ**: bình luận viên thường nói về sự kiện sau thời điểm thật vài giây → cửa sổ gán nhãn
  lệch về sau (delay-aware). Độ trễ là hyperparameter, cần phân tích phân bố độ trễ thực tế.
- Chia train/val/test **theo trận** (không trộn ngẫu nhiên segment) để tránh rò rỉ ngữ cảnh.

## Việc cần làm
1. **Pipeline dữ liệu**: tải transcript + nhãn, căn chỉnh thời gian, sinh mẫu sliding-window, thống kê
   (phân bố lớp, phân bố độ trễ transcript ↔ ground truth, ví dụ nhiễu nhãn). Lưu thành file dữ liệu sạch.
2. **Baseline**: tái hiện `xlm-roberta-base`, 5 epoch, mixed precision. Mốc tham chiếu công bố:
   accuracy ≈ 72.4%, F1-highlight ≈ 0.67 trên validation.
3. **Cải tiến** (thử ít nhất 2 hướng, so sánh công bằng với baseline):
   - Xử lý mất cân bằng lớp: class weighting / focal loss / sampling.
   - Thay backbone (vd. `xlm-roberta-large`, `mdeberta-v3-base`, mô hình hiện đại hơn), điều chỉnh max length.
   - Kích thước/vị trí context window, chiến lược gán nhãn theo độ trễ.
   - Hậu xử lý: temporal smoothing, gộp các dự đoán liên tiếp thành 1 sự kiện, tuning threshold.
4. **Đánh giá** ở 2 mức:
   - Mức đoạn: accuracy, precision, recall, macro-F1, confusion matrix.
   - Mức highlight: F1 cho nhóm (Goal, Card, Penalty) và khả năng truy hồi sự kiện trên toàn trận
     (một sự kiện được tính là bắt được nếu có dự đoán đúng lớp trong khoảng ±T giây).
5. **Service suy luận**: API (FastAPI) — đầu vào: transcript một trận (list segment có start/end/text);
   đầu ra: danh sách sự kiện `{type, timestamp, confidence}`. Có Dockerfile và ví dụ gọi API.

## Ràng buộc môi trường
- Môi trường cloud **không có GPU**: viết code chạy được trên CPU với tập nhỏ (smoke test vài trận, vài trăm bước)
  để kiểm chứng toàn bộ pipeline. Huấn luyện thật sẽ chạy trên **Google Colab T4** → cung cấp notebook
  `notebooks/train_colab.ipynb` chạy được từ đầu đến cuối (clone repo, cài đặt, tải dữ liệu, train, đánh giá, lưu kết quả).
- Mọi cấu hình qua file YAML/CLI, có seed cố định để tái lập.
- Ghi rõ cấu hình huấn luyện, hyperparameter, thời gian train ước tính và lý do chọn.

## Tiêu chí nghiệm thu
- [ ] `pytest` pass (test căn chỉnh nhãn, sinh cửa sổ, chia theo trận không trùng, metric).
- [ ] Một lệnh chạy smoke test toàn pipeline trên CPU thành công.
- [ ] Notebook Colab đầy đủ, có bảng so sánh baseline vs các cải tiến (điền khi chạy thật).
- [ ] Service chạy được bằng `docker run`, có ví dụ request/response.
- [ ] README: mô tả bài toán, phân tích dữ liệu (biểu đồ), phương pháp, kết quả, hướng dẫn chạy.

## Không được làm
- Không đưa vào repo bất kỳ code/tài liệu nội bộ nào của doanh nghiệp. Chỉ dùng dữ liệu và code công khai.
- Không commit dữ liệu lớn hay checkpoint mô hình (dùng `.gitignore`).
