"""Autonomous exploration for arbn_teacher: analyse -> decide -> update, then write a Chinese summary.

  setsid nohup .venv/bin/python analysis/auto_explore.py > analysis/auto/run.log 2>&1 < /dev/null &

Resumable: every evaluation is cached in analysis/auto/ (re-running skips finished work).

Stage 0  baseline (current default: no_geom aug, original + flip, T3A 50, LAME 5), val 3 seeds x 7 short streams,
         test time of the default, measured cost of one extra view per input size.
Stage 1  prediction views: record logits / features of 14 views once per (seed, stream), simulate ~150 view sets
         offline (each with its own T3A + LAME). "safe" pick = best mean accuracy within the time budget whose
         scanline+wave accuracy also improves (the test stream has almost no fisheye-like images; a val-only fisheye
         gain from crops lost on public); "max" pick = best mean accuracy within the budget, no guard.
Stage 2  coordinate search on parameters / structure, starting from the safe pick. A candidate is accepted only if,
         paired with the current best on the same seed + stream: mean gain >= +0.10 pt, wins >= 12/21, and the
         scanline+wave accuracy does not drop by more than 0.10 pt.
Stage 3  official run.py --method arbn_best (subprocess) on val and test for the final configs: time <= budget,
         peak VRAM <= 9.2 GB, label-free health check; otherwise fall back to the previous accepted config.
"""
import itertools
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402
from tta.arbn_tent import lame  # noqa: E402
from tta.t3a import T3A  # noqa: E402

ROOT = E.ROOT
AUTO = ROOT / "analysis" / "auto"
AUTO.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "out" / "_cache" / "auto_views"
CACHE.mkdir(parents=True, exist_ok=True)
PY = sys.executable
SEEDS = (0, 1, 2)
STREAMS = list(E.ORDERS.items())
TIME_BUDGET = 48.0          # seconds on the local RTX 4090 (3090 is roughly 1.5-1.8x slower; limit 90 s)
VRAM_LIMIT = 9.2 * 1024     # MiB (official limit 10 GB)
LOG = open(AUTO / "progress.log", "a")


def log(*a):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + "\n"); LOG.flush()


def key(cfg):
    return json.dumps(cfg, sort_keys=True)


# ------------------------------------------------------------------------------------------------- evaluation
EVAL_FILE = AUTO / "evals.csv"


def evaluate(cfg):
    """3 seeds x 7 short streams. Returns per-run rows (cached)."""
    k = key(cfg)
    if EVAL_FILE.exists():
        old = pd.read_csv(EVAL_FILE)
        hit = old[old.cfg == k]
        if len(hit) == len(SEEDS) * len(STREAMS):
            return hit
    rows = []
    kw = dict(cfg)
    if kw.get("upscales") is not None:
        kw["upscales"] = tuple(kw["upscales"])
    for seed in SEEDS:
        for s, idx in STREAMS:
            pred, _ = E.stream(lambda m: AdaptiveRBNTeacher(m, **kw), idx, seed=seed)
            ok = pred == E.y[idx]; d = E.dom[idx]
            starts = [i for i in range(1, len(d)) if d[i] != d[i - 1]]
            early = np.zeros(len(d), bool)
            for st in starts:
                early[st:st + 128] = True
            rows.append(dict(cfg=k, seed=seed, stream=s, acc=ok.mean(),
                             **{f"n_{dd}": int((d == dd).sum()) for dd in E.DOMS},
                             **{f"c_{dd}": int(ok[d == dd].sum()) for dd in E.DOMS},
                             c_early=int(ok[early].sum()), n_early=int(early.sum())))
    new = pd.DataFrame(rows)
    pd.concat([pd.read_csv(EVAL_FILE), new] if EVAL_FILE.exists() else [new]).to_csv(EVAL_FILE, index=False)
    return new


def summarize(rows):
    r = rows
    dom = {d: r[f"c_{d}"].sum() / r[f"n_{d}"].sum() for d in E.DOMS}
    off = r[r.stream == "official"].acc.mean()
    o = r[r.stream != "official"].groupby("seed").acc
    sw = (r.c_scanline.sum() + r.c_wave.sum()) / (r.n_scanline.sum() + r.n_wave.sum())
    return dict(all_mean=r.acc.mean(), official=off, orders_mean=o.mean().mean(), orders_worst=o.min().mean(),
                scan_wave=sw, after_shift=r.c_early.sum() / r.n_early.sum(), **dom)


def paired(rows, base_rows):
    a = rows.set_index(["seed", "stream"]).acc
    b = base_rows.set_index(["seed", "stream"]).acc
    d = a - b
    return d.mean(), d.min(), int((d > 0).sum())


# ------------------------------------------------------------------------------------------------- timing
def harness_test_seconds(cfg, tag):
    """Official run.py --method arbn_best in a subprocess -> (seconds, peak VRAM MiB, test_pred path)."""
    cfg_path = AUTO / f"cfg_{tag}.json"
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    out = ROOT / "out" / f"auto_{tag}"
    env = dict(os.environ, ARBN_CONFIG=str(cfg_path))
    torch.cuda.empty_cache()
    res = {}
    for split in ("val", "test"):
        subprocess.run([PY, "run.py", "--method", "arbn_best", "--split", split, "--data", str(ROOT / "data"),
                        "--ckpt", str(ROOT / "source_resnet50_160.pth"), "--out", str(out)],
                       cwd=ROOT / "starter", env=env, check=True, capture_output=True, text=True)
        res[split] = json.load(open(out / f"{split}_run_log.json"))
    return res, out / "test_pred.csv"


def view_costs(sizes):
    """Seconds added to the test stream (192 batches) by one extra eval forward at each input size."""
    m = deepcopy(E.BASE).cuda().eval()
    x = ((E.X[:64].cuda().float() / 255) - E.MEAN) / E.STD
    out = {}
    with torch.no_grad():
        for s in sizes:
            xi = x if s == 160 else F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
            for _ in range(5):
                m(xi)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(30):
                m(xi)
            torch.cuda.synchronize()
            out[s] = (time.time() - t) / 30 * 192
    del m; torch.cuda.empty_cache()
    return out


def health(test_pred, ref):
    r = subprocess.run([PY, str(ROOT / "analysis/health_check.py"), str(test_pred), str(ref)], cwd=ROOT,
                       capture_output=True, text=True).stdout
    return "HEALTHY" in r, r.strip()


# ------------------------------------------------------------------------------------------------- stage 1
SCALES = [128, 176, 192, 208, 224, 256]
VIEW_FNS = {"F": lambda x: torch.flip(x, dims=[3])}
for s in SCALES:
    VIEW_FNS[f"U{s}"] = lambda x, s=s: F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
    VIEW_FNS[f"UF{s}"] = lambda x, s=s: torch.flip(F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False), dims=[3])


class ViewRec(AdaptiveRBNTeacher):
    """Default method; at answer time also records every view (eval mode, no grad, deterministic views)."""

    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        self.rec = []

    def predict_views(self, x, s_out, s_feats):
        with torch.no_grad():
            was = self.model.training
            self.model.eval()
            r = {"O": (s_out.half().cpu(), s_feats.half().cpu())}
            for v, fn in VIEW_FNS.items():
                o = self.model(fn(x)); r[v] = (o.half().cpu(), self._feats.half().cpu())
            self.model.train(was)
        self.rec.append(r)
        return super().predict_views(x, s_out, s_feats)


def record_views(seed, sname, idx):
    f = CACHE / f"{seed}_{sname}.pt"
    if not f.exists():
        _, m = E.stream(lambda mm: ViewRec(mm), idx, seed=seed)
        torch.save(m.rec, f)
    return torch.load(f)


def simulate(rec, views, y):
    t3a = T3A(E.BASE.fc, 50, 20)
    P = []
    for b in rec:
        lg = torch.stack([b[v][0].float() for v in views]).mean(0).cuda()
        ft = torch.stack([b[v][1].float() for v in views]).mean(0).cuda()
        P.append(lame(t3a(ft, lg), ft, 5).argmax(1).cpu())
    return torch.cat(P).numpy() == y


def views_to_cfg(views):
    ups = sorted({int(v[2:]) if v.startswith("UF") else int(v[1:]) for v in views if v.startswith("U")})
    flips = {int(v[2:]) for v in views if v.startswith("UF")}
    return dict(flip="F" in views, upscales=ups, upscale_flip=bool(ups) and flips == set(ups))


def stage1(base_time, costs):
    out_f = AUTO / "stage1_sim.csv"
    combos = []
    for fl in (True, False):
        for n in (0, 1, 2):
            for sc in itertools.combinations(SCALES, n):
                for flips in itertools.product((False, True), repeat=n):
                    if n and len(set(flips)) > 1:
                        continue          # the method flips either all or none of the upscaled views
                    v = ["O"] + (["F"] if fl else [])
                    for s, f in zip(sc, flips):
                        v += [f"U{s}"] + ([f"UF{s}"] if f else [])
                    combos.append(v)
    combos = [c for c in combos if c[1:] or True]
    def est_time(v):
        return base_time + sum(costs[int(x[2:]) if x.startswith("UF") else int(x[1:])] for x in v if x.startswith("U")) \
            - (0 if "F" in v else costs[160])
    done = pd.read_csv(out_f) if out_f.exists() else pd.DataFrame()
    have = set(done.views) if len(done) else set()
    todo = [c for c in combos if "+".join(c) not in have and est_time(c) <= TIME_BUDGET + 4]
    log(f"stage 1: {len(combos)} view sets, {len(todo)} to simulate within budget (+4 s margin)")
    recs = {}
    for seed in SEEDS:
        for s, idx in STREAMS:
            recs[(seed, s)] = record_views(seed, s, idx)
        log(f"  views recorded for seed {seed}")
    rows = []
    for i, c in enumerate(todo):
        for (seed, s), rec in recs.items():
            idx = E.ORDERS[s]
            ok = simulate(rec, c, E.y[idx]); d = E.dom[idx]
            rows.append(dict(views="+".join(c), seed=seed, stream=s, acc=ok.mean(), est_time=est_time(c),
                             **{f"n_{dd}": int((d == dd).sum()) for dd in E.DOMS},
                             **{f"c_{dd}": int(ok[d == dd].sum()) for dd in E.DOMS}))
        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            pd.concat([done, pd.DataFrame(rows)]).to_csv(out_f, index=False)
            log(f"  simulated {i + 1}/{len(todo)}")
    del recs
    S = pd.read_csv(out_f)
    agg = []
    for v, g in S.groupby("views"):
        sw = (g.c_scanline.sum() + g.c_wave.sum()) / (g.n_scanline.sum() + g.n_wave.sum())
        o = g[g.stream != "official"].groupby("seed").acc
        agg.append(dict(views=v, est_time=g.est_time.iloc[0], all_mean=g.acc.mean(),
                        official=g[g.stream == "official"].acc.mean(), orders_worst=o.min().mean(), scan_wave=sw,
                        fisheye=g.c_fisheye.sum() / g.n_fisheye.sum()))
    A = pd.DataFrame(agg).sort_values("all_mean", ascending=False)
    A.to_csv(AUTO / "stage1_ranking.csv", index=False)
    ref = A[A.views == "O+F"].iloc[0]
    ok_budget = A[A.est_time <= TIME_BUDGET]
    safe = ok_budget[ok_budget.scan_wave >= ref.scan_wave + 0.003]
    safe_pick = (safe if len(safe) else ok_budget).iloc[0]
    max_pick = ok_budget.iloc[0]
    log(f"stage 1: O+F all_mean {ref.all_mean:.4f} scan_wave {ref.scan_wave:.4f}")
    log(f"  safe pick {safe_pick.views}: all_mean {safe_pick.all_mean:.4f}, scan_wave {safe_pick.scan_wave:.4f}, "
        f"est {safe_pick.est_time:.1f}s")
    log(f"  max pick  {max_pick.views}: all_mean {max_pick.all_mean:.4f}, scan_wave {max_pick.scan_wave:.4f}, "
        f"est {max_pick.est_time:.1f}s")
    return A, ref, safe_pick, max_pick


# ------------------------------------------------------------------------------------------------- stage 2
CANDIDATES = [
    ("lr", [0.05, 0.15, 0.2]),
    ("w_ent", [0.5, 1.5]),
    ("w_cons", [0.5, 2.0]),
    ("mt", [0.99, 0.995]),
    ("t3a_k", [20, 100, None]),
    ("lame_k", [3, 8, None]),
    ("alpha_max", [0.3, 0.8]),
    ("gain", [0.5, 2.0]),
    ("aug", ["soft", "full"]),
]


def stage2(base_cfg):
    best_cfg, best_rows = dict(base_cfg), evaluate(base_cfg)
    hist = []
    for name, values in CANDIDATES:
        for v in values:
            cand = dict(best_cfg); cand[name] = v
            if key(cand) == key(best_cfg):
                continue
            rows = evaluate(cand)
            sm, sb = summarize(rows), summarize(best_rows)
            dm, dmin, wins = paired(rows, best_rows)
            d_sw = sm["scan_wave"] - sb["scan_wave"]
            accept = dm >= 0.001 and wins >= 12 and d_sw >= -0.001
            hist.append(dict(param=name, value=v, d_mean=dm, d_min=dmin, wins=wins, d_scan_wave=d_sw,
                             all_mean=sm["all_mean"], official=sm["official"], accepted=accept))
            log(f"stage 2: {name}={v}: paired {dm:+.4f} (min {dmin:+.4f}, wins {wins}/21), scan_wave {d_sw:+.4f}, "
                f"all_mean {sm['all_mean']:.4f} -> {'ACCEPT' if accept else 'reject'}")
            pd.DataFrame(hist).to_csv(AUTO / "stage2_history.csv", index=False)
            if accept:
                best_cfg, best_rows = cand, rows
    return best_cfg, best_rows, pd.DataFrame(hist)


# ------------------------------------------------------------------------------------------------- summary
def fmt(x):
    return f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x)


def write_summary(base, s1, s2hist, finals, notes):
    L = []
    L.append("# 自動探索總結（arbn_teacher）\n")
    L.append(f"產生時間：{time.strftime('%Y-%m-%d %H:%M')}。所有 val 數字都是 3 個 seed × 7 條短 stream（官方順序＋6 種 domain 順序）。\n")
    L.append("## 目標與判斷原則\n")
    L.append("- 目標：val 至少 0.94，同時不犧牲 test 代表性（test 幾乎沒有 fisheye 類型的圖，所以 scanline+wave 必須一起進步）。")
    L.append(f"- 每個改動都先分析再採用：和目前最佳同 seed、同 stream 成對比較，平均 ≥ +0.1 分、21 組至少贏 12 組、scanline+wave 不退步。")
    L.append(f"- 時間預算：test stream ≤ {TIME_BUDGET:.0f} 秒（4090；換算 3090 約 ×1.5–1.8，上限 90 秒），VRAM ≤ 9.2 GB，test 健康檢查必須通過。\n")
    L.append("## 階段 0：基準（目前預設，public 0.91243）\n")
    L.append("| 指標 | 值 |\n|---|---|")
    for k, v in base.items():
        L.append(f"| {k} | {fmt(v)} |")
    L.append("\n## 階段 1：預測視角（離線模擬）\n")
    A, ref, safe, mx = s1
    L.append("前 12 名（預算內）：\n")
    L.append("| 視角 | 預估 test 秒數 | 平均 | 官方 val | 6 順序最差 | scanline+wave | fisheye |\n|---|---|---|---|---|---|---|")
    for _, r in A[A.est_time <= TIME_BUDGET].head(12).iterrows():
        L.append(f"| {r.views} | {r.est_time:.1f} | {r.all_mean:.4f} | {r.official:.4f} | {r.orders_worst:.4f} | {r.scan_wave:.4f} | {r.fisheye:.4f} |")
    L.append(f"\n- 基準 O+F：平均 {ref.all_mean:.4f}，scanline+wave {ref.scan_wave:.4f}")
    L.append(f"- **安全選擇**（scanline+wave 也要進步）：`{safe.views}`，平均 {safe.all_mean:.4f}，scanline+wave {safe.scan_wave:.4f}")
    L.append(f"- **val 最高選擇**（不設限制）：`{mx.views}`，平均 {mx.all_mean:.4f}，scanline+wave {mx.scan_wave:.4f}\n")
    L.append("## 階段 2：參數與結構（從安全選擇出發，逐一成對比較）\n")
    if len(s2hist):
        L.append("| 參數 | 值 | 成對平均 | 最差一組 | 勝場 | scanline+wave 變化 | 平均 | 官方 val | 採用 |\n|---|---|---|---|---|---|---|---|---|")
        for _, r in s2hist.iterrows():
            L.append(f"| {r.param} | {r.value} | {r.d_mean:+.4f} | {r.d_min:+.4f} | {r.wins}/21 | {r.d_scan_wave:+.4f} | {r.all_mean:.4f} | {r.official:.4f} | {'✅' if r.accepted else '—'} |")
    L.append("\n## 階段 3：最終版本（官方 run.py，獨立 process）\n")
    L.append("| 版本 | 設定 | val（seed 0） | 3 seed × 7 平均 | 官方 val 3 seed | 6 順序最差 | scanline+wave | test 秒數 | VRAM MiB | 健康檢查 | 檔案 |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for f in finals:
        L.append(f"| {f['name']} | `{json.dumps(f['cfg'])}` | {f['val_acc']:.4f} | {f['sum']['all_mean']:.4f} | {f['sum']['official']:.4f} | "
                 f"{f['sum']['orders_worst']:.4f} | {f['sum']['scan_wave']:.4f} | {f['seconds']} | {f['vram']:.0f} | "
                 f"{'✅' if f['healthy'] else '❌'} | `{f['file']}` |")
    L.append("\n## 使用的方法（最終版本的完整流程）\n")
    L.append("""1. **Robust BN + 自適應更新速度（arbn）**：所有 BN 換成保留 EMA 統計量的 Robust BN；每批先只跑 stem+layer1，量這批統計量和記住的統計量的對稱 KL 距離 s，更新速度 α = clip(gain·s, 0.05, alpha_max)。domain 切換時統計量快速跟上，穩定時不受類別組成干擾。
2. **TENT（entropy minimization）**：student 看原圖的輸出算 entropy，SGD（momentum 0.9、nesterov）只更新所有 BN 的 γ、β。
3. **Mean teacher 一致性**：teacher 是 student 的 EMA（動量 mt），看原圖給軟標籤；student 看強增強圖（no_geom：顏色抖動＋模糊＋雜訊＋翻轉，**不做旋轉/仿射**，因為幾何變換會把 scanline 的掃描線變斜，診斷顯示它讓 26.7% 原本答對的 scanline 圖變錯），用對稱 cross-entropy 對齊 teacher。兩個 loss 分開 backward 以節省 VRAM，每批一次 SGD。
4. **多視角作答**：原圖＋水平翻轉（＋最終選出的整張圖放大視角），在作答時的模型狀態下 eval 模式計算，平均 logits 與特徵；不寫入 BN 統計、不做梯度。放大整張圖改變物體在網路中的尺度，能救回原圖連前三名都猜不到的難圖，而且不像中央裁切會丟掉邊緣資訊。
5. **T3A**：每類保留 entropy 最低的 K 個特徵（以分類層權重初始化），用類別代表特徵取代分類層打分。
6. **LAME**：同一批內以特徵 kNN（k 個鄰居）建圖，做 Laplacian 平滑，讓相似圖片的預測一致。
7. 作答在參數更新之前完成，這批學到的東西從下一批生效。""")
    L.append("\n## 試過但沒採用（之前的實驗，見 analysis/README.md）\n")
    L.append("中央裁切視角（val +，public −0.5）、image bank / 完整 RoTTA 記憶庫（test 長 stream 崩潰；低 lr 時穩定但 public 下降）、把最終答案教回 student、用 student 自己的翻轉平均當目標、解凍 layer4、momentum 重置（public −0.5）、分組增強、局部特徵重新池化。\n")
    L.append("## 備註\n")
    for n in notes:
        L.append(f"- {n}")
    L.append("\n## 如何重現\n")
    L.append("```bash\ncd starter\n../.venv/bin/python run.py --method arbn_best --split test --data ../data --ckpt ../source_resnet50_160.pth --out ../out/arbn_best\n```\n`arbn_best` 讀取 `starter/best_config.json`（已寫入最終的安全版本設定）。")
    (ROOT / "analysis" / "AUTO_SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    log("summary written to analysis/AUTO_SUMMARY.md")


# ------------------------------------------------------------------------------------------------- main
def main():
    notes = []
    base_cfg = {}
    log("=== stage 0: baseline")
    base_rows = evaluate(base_cfg)
    base_sum = summarize(base_rows)
    log("baseline", {k: round(v, 4) for k, v in base_sum.items()})
    t_file = AUTO / "timing.json"
    if t_file.exists():
        T = json.load(open(t_file))
    else:
        res, _ = harness_test_seconds({}, "baseline")
        costs = view_costs([160] + SCALES)
        T = dict(base_time=res["test"]["seconds"], costs={str(k): v for k, v in costs.items()})
        json.dump(T, open(t_file, "w"), indent=1)
    base_time = T["base_time"]; costs = {int(k): v for k, v in T["costs"].items()}
    log(f"baseline test time {base_time}s; extra view cost per size (s): " + ", ".join(f"{k}:{v:.1f}" for k, v in costs.items()))

    log("=== stage 1: prediction views")
    s1 = stage1(base_time, costs)
    A, ref, safe, mx = s1
    safe_cfg = views_to_cfg(safe.views.split("+"))
    max_cfg = views_to_cfg(mx.views.split("+"))
    safe_rows = evaluate(safe_cfg)
    ss = summarize(safe_rows)
    dm, dmin, wins = paired(safe_rows, base_rows)
    log(f"stage 1 safe pick verified online: all_mean {ss['all_mean']:.4f} (sim {safe.all_mean:.4f}), paired {dm:+.4f}, wins {wins}/21")

    log("=== stage 2: parameters / structure (from the safe pick)")
    best_cfg, best_rows, hist = stage2(safe_cfg)
    log("stage 2 final config:", best_cfg, {k: round(v, 4) for k, v in summarize(best_rows).items()})

    log("=== stage 3: official runs")
    finals = []
    accepted = [safe_cfg] + [None]
    chain = [best_cfg]
    if key(best_cfg) != key(safe_cfg):
        chain.append(safe_cfg)
    chain.append({})
    ref_file = ROOT / "out" / "arbn_teacher_aug_no_geom" / "test_pred.csv"
    chosen = None
    for i, cfg in enumerate(chain):
        res, tp = harness_test_seconds(cfg, f"safe{i}")
        healthy, hc = health(tp, ref_file)
        ok = res["test"]["seconds"] <= TIME_BUDGET and res["test"]["peak_vram_mib"] <= VRAM_LIMIT and healthy
        log(f"official run safe{i}: val {res['val'].get('acc')}, test {res['test']['seconds']}s, "
            f"VRAM {res['test']['peak_vram_mib']} MiB, healthy {healthy} -> {'OK' if ok else 'fall back'}")
        finals.append(dict(name=f"安全版本 safe{i}" + ("（最終）" if ok else "（未通過）"), cfg=cfg, val_acc=res["val"]["acc"],
                           sum=summarize(evaluate(cfg)), seconds=res["test"]["seconds"], vram=res["test"]["peak_vram_mib"],
                           healthy=healthy, file=str(tp.relative_to(ROOT))))
        if ok:
            chosen = cfg
            break
        notes.append(f"safe{i} 未通過最終檢查：{hc.splitlines()[0] if hc else ''}，test {res['test']['seconds']}s，VRAM {res['test']['peak_vram_mib']}")
    if chosen is not None:
        json.dump(chosen, open(ROOT / "starter" / "best_config.json", "w"), indent=1)
        log("best_config.json written:", chosen)
    # the val-maximising view set (no scanline+wave guard), with the stage-2 parameters
    max_full = dict(best_cfg); max_full.update(max_cfg)
    if key(max_full) != key(chosen or {}):
        res, tp = harness_test_seconds(max_full, "max")
        healthy, hc = health(tp, ref_file)
        finals.append(dict(name="val 最高版本 max", cfg=max_full, val_acc=res["val"]["acc"], sum=summarize(evaluate(max_full)),
                           seconds=res["test"]["seconds"], vram=res["test"]["peak_vram_mib"], healthy=healthy,
                           file=str(tp.relative_to(ROOT))))
        log(f"official run max: val {res['val'].get('acc')}, test {res['test']['seconds']}s, VRAM {res['test']['peak_vram_mib']}, healthy {healthy}")
    final_sum = summarize(evaluate(chosen if chosen is not None else {}))
    notes.append(f"最終安全版本 3 seed × 7 stream 平均 {final_sum['all_mean']:.4f}，官方 val 3 seed 平均 {final_sum['official']:.4f}"
                 f"（目標 0.94：{'達成' if final_sum['official'] >= 0.94 else '未達成'}）。")
    notes.append("val 的 fisheye domain 在 test 中幾乎不存在；scanline+wave 欄位比總分更能預測 public。")
    notes.append("所有 test 檔案都只做了無標籤健康檢查，真正分數要靠 Kaggle public 確認。")
    write_summary(base_sum, s1, hist, finals, notes)
    log("=== done")


if __name__ == "__main__":
    main()
