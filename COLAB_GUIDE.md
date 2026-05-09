# Hướng dẫn chạy trên Google Colab

## Yêu cầu

- Tài khoản Google (Drive + Colab)
- Máy local đã train xong Stage 1 + Stage 3 (có checkpoint)
- Ollama đã cài trên máy local nếu muốn chạy CPU inference

---

## Lần đầu — Chuẩn bị trên máy local

### Bước 1: Đóng gói code + data

```cmd
cd D:\Projects\defect_detection
python scripts\pack_for_colab.py
```

Tạo ra 2 file:
- `code.zip` (~2 MB) — toàn bộ source code
- `data_tsfabric_T1.zip` (~312 MB) — data train

### Bước 2: Upload lên Google Drive

1. Mở [drive.google.com](https://drive.google.com)
2. Tạo thư mục `defect_detection/`
3. Upload vào thư mục đó:
   - `code.zip`
   - `data_tsfabric_T1.zip`
4. Tạo thư mục con `defect_detection/checkpoints/`
5. Upload checkpoint từ `D:\Projects\defect_detection\outputs\checkpoints\`:
   - `stage1_tsfabric_T1_bank.npy` (81 MB)
   - `stage1_tsfabric_T1_meta.json`
   - Thư mục `stage3_tsfabric_T1_lora/` (1.1 GB)

---

## Mở Colab và chạy

### Bước 3: Mở notebook

1. Mở [colab.research.google.com](https://colab.research.google.com)
2. File → Upload notebook → chọn `scripts/colab_run.ipynb`
3. **Bắt buộc:** Runtime → Change runtime type → **T4 GPU**

### Bước 4: Chạy từng cell theo thứ tự

| Cell | Tên | Phải chạy lại mỗi session? |
|------|-----|---------------------------|
| 1 | Kiểm tra GPU | ✅ Luôn |
| 2 | Mount Google Drive | ✅ Luôn |
| 3 | Giải nén code + data | ⏭ Tự skip nếu đã có |
| 4 | Cài packages | ✅ Luôn (~3-5 phút) |
| **5** | **Auto-patch** | ✅ **Luôn — fix tất cả lỗi tương thích** |
| 6 | Copy checkpoints từ Drive | ✅ Luôn |
| 7 | Train (nếu cần) | ⏭ Bỏ qua nếu đã có checkpoint |
| 8 | Lưu checkpoint về Drive | ⏭ Chỉ sau khi train |
| **9** | **Chạy Gradio UI** | ✅ **Luôn** |

> **Quan trọng:** Cell 5 (Auto-patch) **phải chạy mỗi session** — fix các lỗi tương thích giữa InternVL2 và transformers version mới trên Colab.

### Bước 5: Lấy link Gradio

Sau khi Cell 9 chạy xong, copy link dạng:
```
https://xxxxxxxxxxxxxxxx.gradio.live
```
Link này dùng được trong **1 tuần**, chia sẻ được cho người khác.

---

## Sử dụng Gradio UI

### Phân tích ảnh
1. Upload ảnh vải vào ô **"Upload ảnh vải"**
2. Chọn **Category** (mặc định: `tsfabric_T1`)
3. Nhấn **🔬 Phân tích**
4. Xem kết quả: Verdict (Pass/Fail), điểm từng stage, mask overlay

### Chat
Sau khi phân tích xong, gõ câu hỏi vào ô chat:
- *"Lỗi này có nghiêm trọng không?"*
- *"What caused this defect?"*
- *"Cần xử lý như thế nào?"*

AI trả lời dựa trên kết quả phân tích của ảnh vừa upload.

### Ảnh test mẫu
Để chắc chắn thấy kết quả **Fail**, dùng ảnh từ `data/fail/train/tsfabric_T1/`. Tải về:

```python
from google.colab import files
files.download('data/fail/train/tsfabric_T1/000003.jpeg')
```

---

## Các lỗi thường gặp

### "No module named 'configs'"
Thiếu PYTHONPATH. Thêm vào đầu cell:
```python
import sys, os
sys.path.insert(0, '/content/defect_detection')
os.environ['PYTHONPATH'] = '/content/defect_detection'
```

### "meta tensor" / "all_tied_weights_keys"
Cell 5 (Auto-patch) chưa chạy hoặc transformers version sai. Chạy:
```python
!pip install transformers==4.44.0 "torchao>=0.16.0" -q
```
Sau đó chạy lại Cell 5.

### "gr.Divider() not found"
Cell 5 chưa patch app.py. Chạy lại Cell 5.

### Stage 1 Pass nhưng ảnh có lỗi
Threshold quá cao. Hạ xuống:
```python
import json
meta = json.load(open('outputs/checkpoints/stage1_tsfabric_T1_meta.json'))
meta['threshold'] = meta['threshold'] * 0.85
json.dump(meta, open('outputs/checkpoints/stage1_tsfabric_T1_meta.json', 'w'))
```

### RAM OOM khi train Stage 1 trên Colab
```python
!sed -i 's/"batch_size": 4/"batch_size": 2/' configs/stage1_config.py
!sed -i 's/"max_samples": 10000/"max_samples": 5000/' configs/stage1_config.py
!sed -i 's/GREEDY_LIMIT = 50_000/GREEDY_LIMIT = 20_000/' stage1_anomaly/memory_bank.py
```

### Colab bị rate limit / hết GPU quota
- Free tier: ~4-6 giờ GPU/ngày, reset lúc ~12h trưa VN
- Dùng tài khoản Google khác: share thư mục Drive → mount trên Colab account mới
- Checkpoint đã lưu trên Drive → không mất khi đổi account

---

## Lưu ý quan trọng

- **Không đóng tab Colab** khi đang train — session sẽ disconnect sau ~90 phút nếu không có tương tác
- **Luôn chạy Cell 8** sau khi train xong để lưu checkpoint về Drive
- **Stage 2 không cần train** — SAM2 zero-shot cho kết quả chấp nhận được, verdict vẫn đúng nhờ Stage 1
- **Cell 4 phải chạy mỗi session** — Colab xóa packages khi runtime reset
