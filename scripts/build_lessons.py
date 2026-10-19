"""Build the lesson pages docs/loi/*.html from the real runs (self-contained: base64 images, inline CSS, offline).

    python scripts/build_lessons.py [1 2 3 4 5]

Every number and figure is computed here from files of the actual runs (outputs/local/..., data/processed_v2, logs);
the source of each figure is printed under it. docs/LESSONS.md is the text version (for tools); these pages are for
people.
"""

from __future__ import annotations

import base64
import html
import io
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

OUT = Path("docs/loi")
# colour = claim: WRONG (purple) marks what is wrong, RED only what needs a human decision, GREEN what is right
WRONG, RED, GREEN, GREY, ORANGE, BLUE = "#8e3fa8", "#d1453b", "#2e8b57", "#9aa0a6", "#e08a1e", "#3f6fb5"

CSS = """
body{font-family:Segoe UI,Arial,sans-serif;max-width:1180px;margin:24px auto;padding:0 16px;color:#1f2328;
background:#fff;line-height:1.5}
h1{font-size:24px;margin-bottom:4px}h2{font-size:18px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}
.meta{color:#666;font-size:13px}.pair{display:flex;gap:16px;flex-wrap:wrap}.pair>div{flex:1;min-width:320px}
.box{border:2px solid #ccc;border-radius:8px;padding:10px}.bad{border-color:#8e3fa8}.good{border-color:#2e8b57}
.tag{display:inline-block;font-weight:600;padding:1px 8px;border-radius:4px;color:#fff;font-size:13px}
.tbad{background:#8e3fa8}.tred{background:#d1453b}.tgood{background:#2e8b57}.tunk{background:#8a6d00}
.root{background:#fff4e5;border-left:4px solid #e08a1e;padding:8px 12px;margin:8px 0}
.unk{background:#fff1f0;border:2px solid #d1453b;padding:8px 12px;border-radius:6px}
img{max-width:100%}.src{color:#666;font-size:12px}table{border-collapse:collapse;font-size:13px;width:100%}
td,th{border:1px solid #ddd;padding:3px 6px;text-align:left;vertical-align:top}th{background:#f6f8fa}
td.diff{background:#f1e4f7;font-weight:600;color:#5b2370}code,pre{background:#f6f8fa;border-radius:4px}
pre{padding:8px;overflow-x:auto;font-size:12px}mark{background:#fff3b0}
"""


def fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def page(title: str, date: str, body: str, path: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>{CSS}</style></head><body><h1>{html.escape(title)}</h1>
<div class="meta">{date} · soccer-event-detection · trang sinh bởi <code>scripts/build_lessons.py</code> từ dữ liệu
của lần chạy thật</div>{body}</body></html>""", encoding="utf-8")
    print("wrote", path)


def esc(x) -> str:
    return html.escape(str(x))


# --------------------------------------------------------------------------------------------- lesson 1
def lesson1() -> None:
    from sed.config import load_config
    from sed.data.align import label_segments, mentions_any
    from sed.data.prepare import load_events, load_split
    from sed.evaluate import _meta
    from sed.metrics import event_metrics
    from sed.postprocess import detect_events, tune_thresholds
    from sed.data.soccernet_labels import LabelStore
    from sed.data.echoes import EchoesCorpus

    game = "england_epl/2016-2017/2017-05-13 - 14-30 Manchester City 2 - 1 Leicester"
    cfg = load_config("configs/base.yaml", ["data.label_source=official"])
    d = cfg["data"]
    events = LabelStore(d["mirror_root"], d["official_root"], "official").load(game)
    segs = EchoesCorpus(d["echoes_root"], "auto_en", d["lang_csv"]).load(game, 2)
    ev = next(e for e in events if e.cls == "Penalty" and e.half == 2)
    used = label_segments(segs, events, 2, -5, 15)
    meant = label_segments(segs, events, 2, -5, 15,
                           per_class={"Goal": (-5, 15), "Card": (-15, 15), "Penalty": (-60, 5)})
    idx = [i for i, s in enumerate(segs) if -75 <= s.start - ev.time <= 25]

    def timeline(labels, title, frame, mark_wrong=False):
        fig, ax = plt.subplots(figsize=(6.2, 2.3))
        for i in idx:
            s = segs[i]
            x0, x1 = s.start - ev.time, s.end - ev.time
            c = ORANGE if labels[i] == "Penalty" else (BLUE if labels[i] == "Goal" else GREY)
            ax.barh(0, x1 - x0, left=x0, height=0.5, color=c, edgecolor="white")
            if "Penalty" in mentions_any(s.text):
                ax.plot((x0 + x1) / 2, 0.42, marker="v", color="black", ms=7)
            if mark_wrong and used[i] != meant[i]:  # circle the wrongly labelled segments
                ax.add_patch(plt.Rectangle((x0 - 0.3, -0.33), x1 - x0 + 0.6, 0.66, fill=False, ec=WRONG, lw=1.6))
        ax.axvline(0, color="black", lw=1)
        ax.text(0.5, 0.55, "cú sút (nhãn)", fontsize=8)
        ax.set_xlim(-75, 25)
        ax.set_yticks([])
        ax.set_xlabel("thời điểm segment − thời điểm nhãn Penalty (s)")
        ax.set_title(title, fontsize=10, color=frame)
        for sp in ax.spines.values():
            sp.set_color(frame)
            sp.set_linewidth(2)
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=GREY),
                           plt.Line2D([], [], marker="v", color="black", ls="")],
                  labels=["gán Penalty", "gán No-Event", "câu có chữ 'penalty'"], fontsize=7, loc="lower left",
                  ncol=3, bbox_to_anchor=(0, -0.62))
        return fig_b64(fig)

    rows = []
    for i in idx:
        s = segs[i]
        diff = used[i] != meant[i]
        kw = "Penalty" in mentions_any(s.text)
        rows.append(f"<tr><td>{s.start - ev.time:+.1f}</td><td>{'<mark>' if kw else ''}{esc(s.text)}"
                    f"{'</mark>' if kw else ''}</td><td class=\"{'diff' if diff else ''}\">{used[i]}</td>"
                    f"<td class=\"{'diff' if diff else ''}\">{meant[i]}</td></tr>")

    # reproduction of the offset effect on real probabilities (model trained on [-5, 15] labels)
    rcfg = yaml.safe_load(open("outputs/local/v2_baseline/config.yaml"))
    rep = {}
    vv, pv = load_split(rcfg, "valid"), np.load("outputs/local/v2_baseline/probs_valid.npy")
    gv = [g for g in load_events(rcfg, "valid") if g.cls == "Penalty"]
    for name, off in (("+5 s (đúng với nhãn [-5,15])", 5.0), ("−27.5 s (độ trễ của config lỗi)", -27.5)):
        offs = {"Goal": 5.0, "Card": 5.0, "Penalty": off}
        thr = tune_thresholds(_meta(vv), pv, gv, 30.0, 1, 30.0, offs)
        for split in ("valid", "test"):
            v, p = load_split(rcfg, split), np.load(f"outputs/local/v2_baseline/probs_{split}.npy")
            g = [x for x in load_events(rcfg, split) if x.cls == "Penalty"]
            rep[(name, split)] = event_metrics(detect_events(_meta(v), p, thr, 1, 30.0, offs), g, 30.0)["Penalty"]["f1"]
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    names = list(dict.fromkeys(k[0] for k in rep))
    for j, split in enumerate(("valid", "test")):
        ax.bar(np.arange(2) + j * 0.36, [rep[(n, split)] for n in names], width=0.34, label=split,
               color=[BLUE, "#7fa3d6"][j])
    ax.set_xticks(np.arange(2) + 0.18, names, fontsize=8)
    ax.set_ylabel("Penalty F1 mức sự kiện")
    ax.legend(fontsize=8)
    ax.set_title("Cùng xác suất (v2_baseline, train trên [-5,15]), đổi độ trễ quy đổi", fontsize=9)
    rep_img = fig_b64(fig)

    body = f"""
<h2>Triệu chứng</h2>
<p>Ablation <code>imp_class_delays</code> trên Colab (cửa sổ độ trễ riêng cho từng lớp, Penalty [-60, 5] s) báo
Penalty F1 = 0.000 và evt F1 thấp hơn baseline, nên README vòng 1 kết luận "chưa kết luận được".</p>
<h2>Sai và đúng trên cùng một mẫu thật</h2>
<p>Hiệp 2 trận <b>{esc(game.split('/')[-1])}</b>, quả penalty của Mahrez (nhãn tại {ev.time:.1f} s). Cửa sổ độ trễ
được áp khi <i>prepare</i> dữ liệu, nên run lỗi train trên nhãn bên trái chứ không phải bên phải.</p>
<div class="pair"><div class="box bad"><span class="tag tbad">SAI — nhãn run thực sự dùng</span><br>
<img src="{timeline(used, 'Cửa sổ chung [-5, 15] s (ô tím = gán sai)', WRONG, True)}"></div>
<div class="box good"><span class="tag tgood">ĐÚNG — nhãn config tưởng là đang dùng</span><br>
<img src="{timeline(meant, 'Cửa sổ Penalty [-60, 5] s', GREEN)}"></div></div>
<div class="root"><b>Nguyên nhân gốc:</b> nhãn có độ trễ được tạo một lần trong <code>data/processed</code>; config
<code>imp_class_delays</code> chỉ đổi <code>delay_per_class</code> mà không đổi <code>processed_dir</code>, nên train
vẫn đọc nhãn [-5, 15] — chỉ phần quy đổi thời điểm sự kiện dùng độ trễ mới.</div>
<table><tr><th>t − nhãn (s)</th><th>câu ASR (tô = có chữ "penalty")</th><th>nhãn đã dùng</th><th>nhãn định dùng</th></tr>
{''.join(rows)}</table>
<p class="src">Nguồn: Echoes <code>whisper</code> (auto_en) hiệp 2 + Labels-v2 chính thức;
nhãn tính bằng <code>sed.data.align.label_segments</code>. Ô tím = hai cách gán nhãn khác nhau; chữ tô vàng = câu có chữ "penalty".</p>
<h2>Điều chưa xác định được</h2>
<div class="unk"><span class="tag tred">CẦN VŨ PHÂN XỬ</span> Lúc sửa lỗi, tôi viết rằng độ trễ lệch đẩy mọi dự đoán
Penalty ra ngoài ±30 s nên F1 = 0. <b>Tái hiện bác bỏ điều đó</b>: với độ trễ −27.5 s, Penalty F1 còn cao hơn
(bình luận penalty diễn ra trước cú sút, nên đẩy thời điểm về sau lại khớp hơn).
<img src="{rep_img}"><br>
Hai khả năng còn lại cho con số 0 trên Colab: (a) checkpoint epoch 0 (dừng sớm) chưa học được Penalty, như README
vòng 1 đã đoán; (b) một khác biệt khác của môi trường Colab. Máy không phân định được vì output Colab không có ở đây. <b>Vũ kiểm được</b>:
mở <code>MyDrive/soccer-event-detection/outputs/imp_class_delays/history.json</code> — nếu checkpoint được chọn là
epoch 0 và <code>metrics.json → valid.event.Penalty</code> có 0 dự đoán (tp + fp = 0) thì là (a).</div>
<p class="src">Nguồn hình: <code>outputs/local/v2_baseline/probs_*.npy</code>, ngưỡng tune lại trên valid cho mỗi
cách quy đổi, NMS 30 s, dung sai ±30 s. Valid: {rep[(names[0], 'valid')]:.3f} → {rep[(names[1], 'valid')]:.3f};
test: {rep[(names[0], 'test')]:.3f} → {rep[(names[1], 'test')]:.3f}.</p>
<h2>Cách phát hiện</h2>
<p>Khi chạy lại cửa sổ theo lớp trên máy local, số mẫu Penalty train phải tăng từ 576 lên 1,715 — đếm trong
<code>manifest.json</code> của thư mục dùng chung thì không đổi. Đọc code thấy <code>delay_per_class</code> chỉ được
đọc trong <code>prepare</code> và <code>class_offsets</code>.</p>
<h2>Bài học</h2>
<ul><li>Thiết lập nào được "nướng" vào dữ liệu đã xử lý thì phải đi cùng dữ liệu: nay mỗi <code>processed_dir</code>
ghi <code>labeling</code> vào manifest và <code>load_split</code> báo lỗi nếu config khác.</li>
<li>Giải thích một con số bằng suy luận chưa đủ: phải tái hiện. Giải thích đầu tiên của tôi sai.</li></ul>
<p>Sửa trong commit <code>2f09a27</code>.</p>
"""
    page("Ablation cửa sổ độ trễ theo lớp thật ra train trên nhãn cũ", "2026-10-13", body,
         OUT / "2026-10-13_class-delays-dung-chung-du-lieu.html")


# --------------------------------------------------------------------------------------------- lesson 2
def lesson2() -> None:
    import csv

    from sed.config import load_config
    from sed.data.echoes import EchoesCorpus, load_lang_table
    from sed.sources.kaggle_events import overlapping_kaggle_games

    cfg = load_config("configs/base.yaml")
    d = cfg["data"]
    corp = EchoesCorpus(d["echoes_root"], "auto_en", d["lang_csv"])
    games = corp.list_games()  # forward slashes (fixed)
    win = [g.replace("/", "\\") for g in games]  # what the unfixed code produced on Windows
    table = load_lang_table(d["lang_csv"])
    hit_fixed = sum(f"{g}/{h}" in table for g in games for h in (1, 2))
    hit_win = sum(f"{g}/{h}" in table for g in win for h in (1, 2))
    with (Path(d["raw_root"]) / "football-events" / "ginf.csv").open(newline="", encoding="utf-8", errors="replace") as f:
        ginf = list(csv.DictReader(f))
    held = json.load(open("data/processed/manifest.json"))["split_games"]
    held = held["valid"] + held["test"]
    ov_fixed = len(overlapping_kaggle_games(ginf, held))
    ov_win = len(overlapping_kaggle_games(ginf, [g.replace("/", "\\") for g in held]))
    ex = games[0]
    labels = ["Bảng ngôn ngữ\n(hiệp khớp được)", "Trận Kaggle trùng\nvalid/test bị loại"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.6))
    for ax, lab, a, b, tot in zip(axes, labels, (hit_win, ov_win), (hit_fixed, ov_fixed), (len(games) * 2, None)):
        ax.bar([0, 1], [a, b], color=[WRONG, GREEN])
        ax.set_xticks([0, 1], ["id có '\\' (lỗi)", "id có '/' (sửa)"], fontsize=8)
        ax.set_title(lab, fontsize=9)
        for x, v in zip((0, 1), (a, b)):
            ax.text(x, v, str(v), ha="center", va="bottom", fontsize=9)
    img = fig_b64(fig)
    body = f"""
<h2>Triệu chứng</h2>
<p>Trên Windows: nguồn <code>asr_views</code> chỉ lấy được một biến thể Whisper (<code>whisper_v1_en</code>); manifest
báo <code>kaggle_games_removed_overlap_valid_test: 0</code> trong khi README vòng 1 (Colab, Linux) ghi 89.</p>
<h2>Sai và đúng</h2>
<pre>id trận do list_games() trả trên Windows (lỗi):  <mark>{esc(ex.replace('/', chr(92)))}</mark>
khoá trong SN-echoes-lang.csv:                   {esc(ex)}/1
gói SoccerNet getListGames() cũng trả:           <mark>{esc(ex.replace('/', chr(92)))}</mark></pre>
<img src="{img}">
<div class="root"><b>Nguyên nhân gốc:</b> id trận được tạo bằng <code>str(Path)</code> (và <code>os.path.join</code>
trong gói SoccerNet) nên dùng <code>\\</code> trên Windows, rồi được so như chuỗi với dữ liệu viết bằng
<code>/</code>: mọi phép so khớp âm thầm trả về "không tìm thấy" thay vì báo lỗi.</div>
<p class="src">Nguồn: tính lại trong script này trên đúng dữ liệu đã tải — {len(games)} trận Echoes,
<code>SN-echoes-lang.csv</code>, <code>football-events/ginf.csv</code>, danh sách trận valid/test của
<code>data/processed/manifest.json</code>. Cột tím là kết quả khi id dùng <code>\\</code>.</p>
<h2>Hậu quả nếu không phát hiện</h2>
<ul><li>89 trận Kaggle trùng với valid/test lọt vào tập train (rò rỉ) khi chạy trên Windows.</li>
<li>Biến thể ASR "khuyến nghị theo ngôn ngữ" không được dùng: mọi hiệp rơi về mặc định v2.</li>
<li>Không có lỗi nào được báo — chỉ số liệu hơi lệch.</li></ul>
<h2>Cách phát hiện</h2><p>So số trong manifest với số đã ghi trong README (89 → 0), và một nguồn dữ liệu mới cho ra
ít biến thể hơn số thư mục có trên đĩa.</p>
<h2>Bài học</h2><ul><li>Id dùng để so khớp chuỗi phải chuẩn hoá (<code>as_posix()</code>), không dùng
<code>str(Path)</code>; chuẩn hoá cả dữ liệu từ thư viện ngoài.</li>
<li>Phép lọc chống rò rỉ phải có số kỳ vọng để đối chiếu (test <code>test_asr_views.py</code> kiểm id dùng
<code>/</code>).</li></ul><p>Sửa trong commit <code>654c67d</code>.</p>"""
    page("Id trận dùng '\\' trên Windows làm mọi phép so khớp trả về rỗng", "2026-10-13", body,
         OUT / "2026-10-13_id-tran-windows.html")


# --------------------------------------------------------------------------------------------- lesson 3
def lesson3() -> None:
    old = json.load(open("outputs/local/_archive/v2_baseline_segment_selection/history.json"))
    new = json.load(open("outputs/local/v2_baseline/history.json"))
    mo = json.load(open("outputs/local/_archive/v2_baseline_segment_selection/metrics.json"))
    mn = json.load(open("outputs/local/v2_baseline/metrics.json"))
    so = json.load(open("outputs/local/_archive/v2_baseline_segment_selection/summary.json"))
    sn = json.load(open("outputs/local/v2_baseline/summary.json"))

    fig, ax = plt.subplots(figsize=(6.2, 3))
    st = [r["step"] for r in old]
    ax.plot(st, [r["highlight_f1"] for r in old], "-o", color=WRONG, ms=4, label="F1 mức đoạn (tiêu chí cũ)")
    b = max(old, key=lambda r: r["highlight_f1"])
    ax.plot(b["step"], b["highlight_f1"], "o", ms=13, mfc="none", mec=WRONG, mew=2)
    ax.annotate(f"chọn step {b['step']}", (b["step"], b["highlight_f1"]), xytext=(8, -22),
                textcoords="offset points", color=WRONG, fontsize=8)
    ax.set_xlabel("bước tối ưu")
    ax.set_ylabel("F1 trên tập chọn")
    ax.set_title("SAI: chọn theo F1 mức đoạn (tập con xáo trộn)", color=WRONG, fontsize=10)
    ax.legend(fontsize=8)
    img_old = fig_b64(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3))
    st = [r["step"] for r in new]
    ax.plot(st, [r["event_f1"] for r in new], "-o", color=GREEN, ms=4, label="F1 mức sự kiện, 45 trận nguyên vẹn")
    ax.plot(st, [r["highlight_f1"] for r in new], "--", color=GREY, label="F1 mức đoạn (cùng run, chỉ để so)")
    b = max(new, key=lambda r: r["event_f1"])
    ax.plot(b["step"], b["event_f1"], "o", ms=13, mfc="none", mec=GREEN, mew=2)
    ax.annotate(f"chọn step {b['step']}", (b["step"], b["event_f1"]), xytext=(8, -22),
                textcoords="offset points", color=GREEN, fontsize=8)
    ax.set_xlabel("bước tối ưu")
    ax.set_title("ĐÚNG: chọn theo F1 mức sự kiện", color=GREEN, fontsize=10)
    ax.legend(fontsize=8)
    img_new = fig_b64(fig)

    groups = ["replay_recall", "keyword_no_event", "time_offset", "wrong_class", "duplicate", "other"]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    x = np.arange(len(groups))
    for j, (m, lab, c) in enumerate(((mo, "chọn theo mức đoạn", WRONG), (mn, "chọn theo mức sự kiện", GREEN))):
        ax.bar(x + j * 0.4, [m["valid"]["fp_diagnosis"]["groups"].get(g, {}).get("count", 0) for g in groups],
               width=0.38, color=c, label=lab)
    ax.set_xticks(x + 0.2, groups, fontsize=8)
    ax.set_ylabel("số FP mức sự kiện (valid)")
    ax.legend(fontsize=8)
    img_fp = fig_b64(fig)
    gfp = (mo["valid"]["event"]["Goal"].get("fp"), mn["valid"]["event"]["Goal"].get("fp"))
    body = f"""
<h2>Triệu chứng</h2>
<p><code>v2_baseline</code> (nhãn chính thức) có F1 mức đoạn ngang baseline nhãn mirror, nhưng evt F1 valid chỉ
<b>{so['evt_f1']:.3f}</b> (baseline mirror 0.510): {gfp[0]} FP Goal, precision {so['evt_precision']:.2f}.</p>
<h2>Sai và đúng trên cùng recipe, cùng dữ liệu</h2>
<div class="pair"><div class="box bad"><img src="{img_old}"></div><div class="box good"><img src="{img_new}"></div></div>
<div class="root"><b>Nguyên nhân gốc:</b> checkpoint được chọn theo F1 mức đoạn trên tập con valid đã xáo trộn
(mọi mẫu dương + 30k mẫu âm), trong khi mục tiêu là F1 mức sự kiện trên cả trận. Một checkpoint muộn, tự tin hơn
thắng ở mức đoạn nhưng bắn sự kiện liên tục vào bình luận sau bàn thắng.</div>
<p>Kết quả cuối: evt F1 valid {so['evt_f1']:.3f} → <b>{sn['evt_f1']:.3f}</b>, test {so['test_evt_f1']:.3f} →
<b>{sn['test_evt_f1']:.3f}</b>; FP Goal valid {gfp[0]} → {gfp[1]}.</p>
<img src="{img_fp}">
<p class="src">Nguồn: <code>outputs/local/_archive/v2_baseline_segment_selection/</code> (run chọn theo mức đoạn) và
<code>outputs/local/v2_baseline/</code> (chạy lại, chọn theo mức sự kiện): <code>history.json</code>,
<code>metrics.json → valid.fp_diagnosis</code>.</p>
<h2>Một chẩn đoán sai trên đường đi</h2>
<p><span class="tag tbad">SAI</span> Tôi đoán trước là lưới ngưỡng dừng ở 0.95 nên không chọn được ngưỡng cao hơn.
Mở rộng lưới tới 0.999 rồi chấm lại (<code>scripts/reevaluate.py</code>): ngưỡng tối ưu vẫn 0.85, F1 không đổi.
Kiểm tiếp thì thấy thời điểm Goal của nhãn chính thức và mirror trùng nhau trong 1 s cho cả 1,430 bàn thắng, và FP
rải đều trên mọi trận — loại trừ dữ liệu, còn lại cách chọn checkpoint.</p>
<h2>Bài học</h2><ul><li>Chỉ số chọn model phải là chỉ số mục tiêu, đo cùng cách đánh giá cuối (cả pipeline sự kiện,
trận nguyên vẹn), không phải một proxy rẻ hơn.</li>
<li>Khi một con số tụt, loại trừ từng giả thuyết bằng phép đo (lưới ngưỡng, nhãn, phân bố FP theo trận) trước khi
sửa.</li></ul><p>Sửa trong commit <code>9e0f92b</code>.</p>"""
    page("Chọn checkpoint theo F1 mức đoạn làm evt F1 tụt còn 0.389", "2026-10-13", body,
         OUT / "2026-10-13_chon-checkpoint-muc-doan.html")


# --------------------------------------------------------------------------------------------- lesson 4
def _train_curve(log: Path):
    pts = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\[train\] epoch=\d+ step=(\d+)/\d+ .* (\d+)s$", line)
        if m:
            pts.append((int(m.group(1)), int(m.group(2))))
    return pts


def lesson4() -> None:
    slow = _train_curve(Path("outputs/local/logs/bb_modernbert.log"))
    clean = _train_curve(Path("outputs/local/logs/bb_ettin.log"))
    fig, ax = plt.subplots(figsize=(7.5, 3))
    ax.plot([t / 3600 for _, t in slow], [s for s, _ in slow], color=WRONG, label="ModernBERT 14/10, GPU dùng chung")
    ax.plot([t / 3600 for _, t in clean], [s for s, _ in clean], color=GREEN,
            label="Ettin 15/10 (cùng kiến trúc, 8-bit AdamW), GPU riêng")
    ax.set_xlabel("giờ từ lúc bắt đầu train")
    ax.set_ylabel("bước tối ưu")
    ax.legend(fontsize=8)
    ax.set_title("Tốc độ train: cùng recipe, khác ở chỗ có job khác trên GPU", fontsize=10)
    img = fig_b64(fig)
    rs = lambda p: 32 * (p[-1][0] - p[0][0]) / max(p[-1][1] - p[0][1], 1)  # noqa: E731
    q = [ln for ln in Path("outputs/local/logs/queue.log").read_text(encoding="utf-8").splitlines()
         if "start #" in ln or "done #" in ln or "GPU" in ln]
    qtxt = "\n".join(esc(re.sub(r" > outputs.*", "", ln)) for ln in q[-22:])
    body = f"""
<h2>Triệu chứng</h2>
<p>Hai lần, train chậm đi hàng chục lần mà không có lỗi nào: (1) đêm 13/10, <code>v2_baseline</code> mất ~10,500 s cho
500 bước đầu (bình thường ~120 s); (2) 14/10, <code>bb_modernbert</code> chạy {rs(slow):.0f} mẫu/s trong khi
benchmark cùng cấu hình đo 91 mẫu/s.</p>
<h2>Sai và đúng</h2><img src="{img}">
<div class="root"><b>Nguyên nhân gốc:</b> hai tiến trình train cùng ở trên GPU 4 GB, cả hai tràn sang bộ nhớ
chung của Windows (WDDM) nên chậm hàng chục lần. Lần (1): <code>TaskStop</code> chỉ dừng lớp shell bên ngoài,
runner hàng đợi cũ vẫn sống và đọc cùng file hàng đợi → hai runner. Lần (2): một job khác của Vũ
(<code>py_chay_hangloat.py</code>, 3 tiến trình) bắt đầu 09:50:40, một phút sau khi ModernBERT bắt đầu (09:49:46).</div>
<p class="src">Nguồn: dòng <code>[train]</code> trong <code>outputs/local/logs/bb_modernbert.log</code> và
<code>bb_ettin.log</code> (Ettin {rs(clean):.0f} mẫu/s; tốc độ ở đây tính cả thời gian đánh giá giữa chừng, nên
thấp hơn 91 mẫu/s của benchmark chỉ đo bước train); thời điểm tạo tiến trình lấy từ
<code>Get-CimInstance Win32_Process</code> lúc điều tra. Log của lần (1) đã bị ghi đè khi chạy lại; số 10,500 s ghi
lại từ lúc điều tra.</p>
<h2>Chẩn đoán sai</h2>
<p><span class="tag tbad">SAI</span> Với lần (2) tôi kết luận là "phân mảnh bộ nhớ do ModernBERT bỏ padding" và đổi
batch 32 → 16 × accumulation 2. Ettin với batch 16 vẫn ~2.3 s/bước, bộ nhớ vẫn 3.9 GB — batch nhỏ hơn mà bộ nhớ không
giảm là dấu hiệu sai. Liệt kê <code>nvidia-smi --query-compute-apps</code> mới thấy 3 tiến trình lạ. Thay đổi batch
đã hoàn tác.</p>
<h2>Dòng thời gian hàng đợi (trích log thật)</h2><pre>{qtxt}</pre>
<h2>Bài học</h2><ul><li>Trên GPU nhỏ, mỗi lúc chỉ một job: runner nay giữ khoá (một runner mỗi file) và chờ tới khi
<code>nvidia-smi</code> không còn tiến trình nào trước mỗi job.</li>
<li>Dừng tiến trình nền phải kiểm cả cây tiến trình con (<code>Get-CimInstance Win32_Process</code>), không tin vào
lệnh dừng lớp ngoài.</li><li>Khi tốc độ tụt, liệt kê ai đang dùng GPU trước khi nghĩ tới nguyên nhân bên trong
model.</li></ul><p>Sửa trong commit <code>839e4a1</code> (một runner) và <code>d9a65f2</code> (chờ GPU rảnh).</p>"""
    page("Hai job trên một GPU 4 GB: train chậm hàng chục lần, và một chẩn đoán sai", "2026-10-14", body,
         OUT / "2026-10-14_hai-job-mot-gpu.html")


# --------------------------------------------------------------------------------------------- lesson 5
def lesson5() -> None:
    runs = {"seed 42": "outputs/local/v2_delays_views_hardneg", "seed 1": "outputs/local/v2_delays_views_hardneg_s1"}
    cfg = yaml.safe_load(open(runs["seed 42"] + "/config.yaml"))
    hist = {k: json.load(open(v + "/history.json")) for k, v in runs.items()}
    summ = {k: json.load(open(v + "/summary.json")) for k, v in runs.items()}
    spe = round(hist["seed 42"][0]["step"] / hist["seed 42"][0]["epoch"])
    total = spe * cfg["train"]["epochs"]
    warm = int(total * cfg["train"]["warmup_ratio"])
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for (k, h), c in zip(hist.items(), (BLUE, ORANGE)):
        ax.plot([r["step"] for r in h], [r["event_f1"] for r in h], "-o", ms=4, color=c, label=f"{k}")
        b = max(h, key=lambda r: r["event_f1"])
        ax.plot(b["step"], b["event_f1"], "o", ms=13, mfc="none", mec=c, mew=2)
    ax.axvline(spe, color="black", ls=":", lw=1)
    ax.text(spe, 0.08, "hết epoch 1 ", fontsize=8, ha="right")
    ax2 = ax.twinx()
    s = np.arange(0, spe + 1, 50)
    lr = np.where(s < warm, s / warm, (total - s) / (total - warm)) * float(cfg["train"]["lr"])
    ax2.plot(s, lr, color=GREY, lw=1)
    ax2.set_ylabel("learning rate", color=GREY)
    ax.set_xlabel("bước tối ưu")
    ax.set_ylabel("evt F1 trên 45 trận chọn")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("SAI: dừng sớm khi lr còn ở đỉnh — chọn trúng đỉnh nhiễu", color=WRONG, fontsize=10)
    img_bad = fig_b64(fig)
    v3 = sorted(Path("outputs/local/v3").glob("views_hardneg_xlmr_s*/history.json"))
    good = ""
    if v3:
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        for p, c in zip(v3, (BLUE, ORANGE, GREEN)):
            h = json.load(open(p))
            ax.plot([r["step"] for r in h], [r["event_f1"] for r in h], "-o", ms=4, color=c, label=p.parent.name)
        ax.set_xlabel("bước tối ưu")
        ax.set_ylabel("evt F1 trên 45 trận chọn")
        ax.legend(fontsize=8)
        ax.set_title("ĐÚNG: 2 epoch, lr giảm về 0, không dừng sớm", color=GREEN, fontsize=10)
        good = f'<div class="box good"><img src="{fig_b64(fig)}"></div>'
        vs = [json.load(open(p.parent / "summary.json")) for p in v3 if (p.parent / "summary.json").exists()]
        if vs:
            t = np.array([x["test_evt_f1"] for x in vs])
            good += (f"<p>v3, {len(vs)} seed: test evt F1 {t.mean():.3f} ± {t.std(ddof=1) if len(t) > 1 else 0:.3f} "
                     f"(min {t.min():.3f}, max {t.max():.3f}).</p>")
    else:
        good = '<div class="box"><p>Chưa có run v3 (đang chạy) — trang sẽ được dựng lại khi có.</p></div>'
    body = f"""
<h2>Triệu chứng</h2>
<p>Cùng cấu hình <code>v2_delays_views_hardneg</code>, chỉ đổi seed: test evt F1 {summ['seed 42']['test_evt_f1']:.3f}
(seed 42) và {summ['seed 1']['test_evt_f1']:.3f} (seed 1). Chênh {summ['seed 42']['test_evt_f1'] -
summ['seed 1']['test_evt_f1']:.3f} — lớn hơn phần lớn khác biệt giữa các cấu hình đang được so.</p>
<h2>Sai và đúng</h2><div class="pair"><div class="box bad"><img src="{img_bad}"></div>{good}</div>
<div class="root"><b>Nguyên nhân gốc:</b> lịch lr tính cho 5 epoch nhưng dừng sớm sau 4 lần đánh giá không cải thiện,
nên không run nào xong nổi epoch đầu ({spe} bước): checkpoint được chọn khi lr còn ở đỉnh, nơi evt F1 nhảy ±0.05
giữa hai lần đánh giá liền nhau. Chọn "max" trên một đường nhiễu là chọn đỉnh nhiễu.</div>
<p class="src">Nguồn: <code>history.json</code> của <code>outputs/local/v2_delays_views_hardneg</code> và
<code>..._s1</code>; đường lr tính lại từ config (warmup {cfg['train']['warmup_ratio']}, tuyến tính trên {total}
bước).</p>
<h2>Hậu quả</h2><p>Các kết luận vòng 2 dựa trên chênh lệch 0.01–0.05 không đứng vững; chỉ những chênh lệch lớn (cửa
sổ theo lớp cho Penalty, tỉ lệ âm 2) là chắc. Vòng 3 chạy lại 4 cấu hình chính × 3 seed.</p>
<h2>Bài học</h2><ul><li>Trước khi so cấu hình, đo phương sai giữa các seed của chính recipe đó.</li>
<li>Dừng sớm và chọn checkpoint trên một tín hiệu nhiễu cần lr đã giảm (lịch khớp với số bước thực sự chạy).</li>
<li>Tập dùng để chọn checkpoint không được đổi theo seed train (nay cố định bằng <code>select_seed</code>).</li></ul><p>Recipe mới trong commit <code>38d3ed1</code>.</p>"""
    page("Đổi seed lệch 0.073 evt F1: dừng sớm lúc lr còn ở đỉnh", "2026-10-15", body,
         OUT / "2026-10-15_seed-dung-som.html")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for k in todo:
        {"1": lesson1, "2": lesson2, "3": lesson3, "4": lesson4, "5": lesson5}[k]()
