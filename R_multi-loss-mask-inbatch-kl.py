# train_2gpu_teacher_mask_K9_inbatch_tqdm_kl_distill_anneal.py
import os
import json
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, evaluation

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-Embedding-8B"

    train_path: str = "/hits/basement/nlp/qianz/sts_embedding/dataset/augmented_dataset/t4.jsonl"
    val_path: str = "dataset/original_dataset_jsonl/val.jsonl"
    output_dir: str = "Result_output/multi-loss-inbatch-kl-best"

    teacher_adapter_dir: str = "Result_output/multi-loss-inbatch-kl-best_1/best_model"

    student_device: str = "cuda:0"
    teacher_device: str = "cuda:1"

    sim_prompt: str = (
        "Instruct: Given a movie or story synopsis, produce an embedding that captures STORY EQUIVALENCE.\n"
        "Focus on: theme,Course of Action, core goal, key conflict/obstacle, and outcome trajectory.\n"
        "Ignore: writing style, names, locations,or superficial genre terms.\n"
        "text: "
    )

    # Train
    batch_size: int = 8
    num_epochs: int = 5
    lr: float = 5e-5
    grad_accum: int = 1
    temperature: float = 0.09  # for CE / InfoNCE
    margin: float = -0.05       # teacher mask rule
    seed: int = 42
    max_seq_length: int = 1024

    # Mixed precision
    use_amp: bool = True
    prefer_bf16: bool = True

    # memory
    use_gradient_checkpointing: bool = True

    # eval/log
    eval_every_steps: int = 5
    eval_batch_size: int = 2
    log_every_steps: int = 5

    # Student LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    target_modules: List[str] = field(
        default_factory=lambda: "q_proj k_proj v_proj o_proj gate_proj up_proj down_proj".split()
    )

    # In-batch negatives (use other samples' positives as negatives)
    use_inbatch_neg: bool = True

    # ---- KL distillation: logits distribution distillation ----
    use_kl_distill: bool = True
    kl_temperature: float = 0.15   # soften distill distribution; try 0.07~0.2
    kl_fp32: bool = True           # compute KL in fp32 for stability

    # ---- KL weight annealing (退火) ----
    # weight will linearly decay from kl_weight_start -> kl_weight_end
    # between [kl_anneal_start_ratio, kl_anneal_end_ratio] of total steps.
    kl_weight_start: float = 0.7   # try 0.3~1.0
    kl_weight_end: float = 0.05     # typically 0.0~0.1
    kl_anneal_start_ratio: float = 0.0
    kl_anneal_end_ratio: float = 1.0


cfg = Config()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text(x) -> Optional[str]:
    if x is None:
        return None
    x = str(x).strip()
    return x if x else None


def add_sim_prefix(t: str) -> str:
    return cfg.sim_prompt + t


PAD_NEG_RAW = "[PAD_NEG]"
PAD_NEG = add_sim_prefix(PAD_NEG_RAW)


class MultiNegJsonlDatasetK9(Dataset):
    """
    固定 9 个负样本字段（1+9格式）：
      neg_theme_1..3, neg_structure_1..3, neg_outcome_1..3
    缺失的用 PAD_NEG 补位（训练时永远 mask 掉，不影响 loss）
    """
    def __init__(self, path: str):
        self.samples: List[Dict[str, Any]] = []
        total = used = skipped = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                total += 1
                line = line.strip()
                if not line:
                    skipped += 1
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                a = clean_text(obj.get("anchor") or obj.get("query") or obj.get("a"))
                p = clean_text(obj.get("positive") or obj.get("pos") or obj.get("p"))
                if a is None or p is None:
                    skipped += 1
                    continue

                n_t1 = clean_text(obj.get("neg_theme_1"))
                n_t2 = clean_text(obj.get("neg_theme_2"))
                n_t3 = clean_text(obj.get("neg_theme_3"))

                n_s1 = clean_text(obj.get("neg_structure_1"))
                n_s2 = clean_text(obj.get("neg_structure_2"))
                n_s3 = clean_text(obj.get("neg_structure_3"))

                n_o1 = clean_text(obj.get("neg_outcome_1"))
                n_o2 = clean_text(obj.get("neg_outcome_2"))
                n_o3 = clean_text(obj.get("neg_outcome_3"))

                if (
                    n_t1 is None and n_t2 is None and n_t3 is None and
                    n_s1 is None and n_s2 is None and n_s3 is None and
                    n_o1 is None and n_o2 is None and n_o3 is None
                ):
                    skipped += 1
                    continue

                negs = [
                    add_sim_prefix(n_t1) if n_t1 is not None else PAD_NEG,
                    add_sim_prefix(n_t2) if n_t2 is not None else PAD_NEG,
                    add_sim_prefix(n_t3) if n_t3 is not None else PAD_NEG,
                    add_sim_prefix(n_s1) if n_s1 is not None else PAD_NEG,
                    add_sim_prefix(n_s2) if n_s2 is not None else PAD_NEG,
                    add_sim_prefix(n_s3) if n_s3 is not None else PAD_NEG,
                    add_sim_prefix(n_o1) if n_o1 is not None else PAD_NEG,
                    add_sim_prefix(n_o2) if n_o2 is not None else PAD_NEG,
                    add_sim_prefix(n_o3) if n_o3 is not None else PAD_NEG,
                ]

                self.samples.append({"a": add_sim_prefix(a), "p": add_sim_prefix(p), "negs": negs})
                used += 1

        print(f"[train dataset] total={total}, used={used}, skipped={skipped} (K=9)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    a = [x["a"] for x in batch]
    p = [x["p"] for x in batch]
    negs_by_k = [[x["negs"][j] for x in batch] for j in range(9)]
    pad_mask = [[(x["negs"][j] != PAD_NEG) for j in range(9)] for x in batch]  # [B,9]
    return {"a": a, "p": p, "negs_by_k": negs_by_k, "pad_mask": pad_mask}


def read_triplets(val_path: str):
    anchors, positives, negatives = [], [], []
    total = used = skipped = 0
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            line = line.strip()
            if not line:
                skipped += 1
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            a = clean_text(obj.get("anchor") or obj.get("query") or obj.get("a"))
            p = clean_text(obj.get("positive") or obj.get("pos") or obj.get("p"))
            n = clean_text(obj.get("negative") or obj.get("neg") or obj.get("b") or obj.get("n"))
            if a is None or p is None or n is None:
                skipped += 1
                continue

            anchors.append(add_sim_prefix(a))
            positives.append(add_sim_prefix(p))
            negatives.append(add_sim_prefix(n))
            used += 1

    print(f"[val] total={total}, used={used}, skipped={skipped}")
    return anchors, positives, negatives


def attach_adapter(st_model: SentenceTransformer, adapter_dir: str) -> None:
    from peft import PeftModel
    tm = st_model._first_module()
    am = getattr(tm, "auto_model", None)
    if am is None:
        raise RuntimeError("Could not find st_model._first_module().auto_model")
    tm.auto_model = PeftModel.from_pretrained(am, adapter_dir).eval()


def inject_lora(st_model: SentenceTransformer):
    from peft import LoraConfig, get_peft_model, TaskType
    tm = st_model._first_module()
    base = getattr(tm, "auto_model", None)
    if base is None:
        raise RuntimeError("Could not find st_model._first_module().auto_model")

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    tm.auto_model = get_peft_model(base, peft_config)
    try:
        tm.auto_model.print_trainable_parameters()
    except Exception:
        pass


def assert_trainable(st_model: SentenceTransformer):
    trainable = sum(p.numel() for p in st_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in st_model.parameters())
    print(f"[DEBUG] trainable params: {trainable} / {total} = {trainable/total:.8f}")
    if trainable == 0:
        raise RuntimeError("No trainable params. LoRA target_modules mismatch.")


def enable_ckpt_input_grads(auto_model: Any):
    m = auto_model
    if hasattr(m, "base_model"):
        try:
            m = m.base_model
        except Exception:
            pass
    if hasattr(m, "model"):
        try:
            m = m.model
        except Exception:
            pass

    if hasattr(m, "enable_input_require_grads"):
        m.enable_input_require_grads()
        return

    try:
        emb = auto_model.get_input_embeddings()
        emb.weight.requires_grad_(True)
    except Exception:
        pass


def tokenize_to_device(st_model: SentenceTransformer, texts: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
    feats = st_model.tokenize(texts)
    for k, v in feats.items():
        if torch.is_tensor(v):
            feats[k] = v.to(device, non_blocking=True)
    return feats


def st_sentence_embedding(st_model: SentenceTransformer, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
    return st_model(feats)["sentence_embedding"]


def cosine_mat(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return x @ y.transpose(0, 1)


@torch.no_grad()
def run_eval(student: SentenceTransformer, val_eval: evaluation.TripletEvaluator):
    student.eval()
    res = val_eval(student, output_path=None)
    student.train()
    return res


def extract_primary_score(
    res: Union[float, int, Dict[str, Any]],
    evaluator: Any,
) -> Tuple[str, float]:
    metric_key = getattr(evaluator, "primary_metric", None) or "score"

    if isinstance(res, dict):
        if metric_key in res and isinstance(res[metric_key], (float, int, np.number)):
            return metric_key, float(res[metric_key])
        cand = [k for k in res.keys() if "max_accuracy" in k or k.endswith("_accuracy")]
        for k in cand:
            v = res.get(k, None)
            if isinstance(v, (float, int, np.number)):
                return k, float(v)
        for k, v in res.items():
            if isinstance(v, (float, int, np.number)):
                return k, float(v)
        raise RuntimeError(f"Evaluator returned dict but no numeric metrics found: keys={list(res.keys())}")

    if isinstance(res, (float, int, np.number)):
        return metric_key, float(res)

    raise RuntimeError(f"Unsupported eval result type: {type(res)}")


def save_best(student: SentenceTransformer, best_dir: str, meta: Dict[str, Any]):
    os.makedirs(best_dir, exist_ok=True)
    student.save(best_dir)
    with open(os.path.join(best_dir, "best_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def mask_logits(logits: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    return torch.where(keep_mask, logits, torch.full_like(logits, -1e9))


def kl_distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, fp32: bool, T: float) -> torch.Tensor:
    """
    KL( softmax(teacher/T) || softmax(student/T) ) with standard T^2 scaling.
    """
    if fp32:
        s = student_logits.float()
        t = teacher_logits.float()
    else:
        s = student_logits
        t = teacher_logits

    s = s / T
    t = t / T
    logp_s = F.log_softmax(s, dim=1)
    p_t = F.softmax(t, dim=1)
    return F.kl_div(logp_s, p_t, reduction="batchmean") * (T * T)


def get_kl_weight(global_step_1based: int, total_steps: int) -> float:
    """
    Linear anneal from cfg.kl_weight_start -> cfg.kl_weight_end
    between [start_ratio, end_ratio] of total_steps.
    Steps are 1-based.
    """
    if total_steps <= 0:
        return cfg.kl_weight_end

    start_step = max(1, int(total_steps * cfg.kl_anneal_start_ratio))
    end_step = max(start_step, int(total_steps * cfg.kl_anneal_end_ratio))

    if global_step_1based <= start_step:
        return float(cfg.kl_weight_start)
    if global_step_1based >= end_step:
        return float(cfg.kl_weight_end)

    # linear interpolation
    t = (global_step_1based - start_step) / float(end_step - start_step)
    return float(cfg.kl_weight_start + t * (cfg.kl_weight_end - cfg.kl_weight_start))


def main():
    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    assert torch.cuda.is_available(), "CUDA required"
    assert torch.cuda.device_count() >= 2, "Need 2 GPUs visible"

    student_dev = torch.device(cfg.student_device)
    teacher_dev = torch.device(cfg.teacher_device)

    use_bf16 = cfg.use_amp and cfg.prefer_bf16 and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_amp = cfg.use_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    # data
    train_ds = MultiNegJsonlDatasetK9(cfg.train_path)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    total_steps = cfg.num_epochs * len(train_dl)
    print(f"[INFO] total_steps (micro-steps) = {total_steps} | grad_accum={cfg.grad_accum}")

    val_a, val_p, val_n = read_triplets(cfg.val_path)
    val_eval = evaluation.TripletEvaluator(val_a, val_p, val_n, name="val", batch_size=cfg.eval_batch_size)

    # student
    print("[LOAD] student ->", student_dev)
    student = SentenceTransformer(cfg.model_name, trust_remote_code=True, device=str(student_dev))
    student.max_seq_length = cfg.max_seq_length
    inject_lora(student)
    assert_trainable(student)
    student.train()

    # checkpointing
    if cfg.use_gradient_checkpointing:
        am = student._first_module().auto_model
        if hasattr(am, "gradient_checkpointing_enable"):
            am.gradient_checkpointing_enable()
        if hasattr(am, "config"):
            am.config.use_cache = False
        enable_ckpt_input_grads(am)
        print("[DEBUG] gradient checkpointing ON (+ input grads enabled)")

    # teacher
    print("[LOAD] teacher ->", teacher_dev)
    teacher = SentenceTransformer(cfg.model_name, trust_remote_code=True, device=str(teacher_dev))
    teacher.max_seq_length = cfg.max_seq_length
    attach_adapter(teacher, cfg.teacher_adapter_dir)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    try:
        teacher = teacher.half()
    except Exception:
        pass

    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=cfg.lr)

    os.makedirs(cfg.output_dir, exist_ok=True)
    global_step = 0

    best_score = -1e18
    best_dir = os.path.join(cfg.output_dir, "best_model")
    best_meta = {"epoch": None, "step": None, "metric": None, "score": None}

    for epoch in range(cfg.num_epochs):
        pbar = tqdm(train_dl, desc=f"epoch {epoch+1}/{cfg.num_epochs}", dynamic_ncols=True)
        for batch in pbar:
            global_step += 1
            kl_w = get_kl_weight(global_step, total_steps) if cfg.use_kl_distill else 0.0

            a_txt = batch["a"]
            p_txt = batch["p"]
            negs_by_k = batch["negs_by_k"]
            pad_mask_cpu = batch["pad_mask"]  # [B,9] python bool
            pad_mask = torch.tensor(pad_mask_cpu, dtype=torch.bool, device=student_dev)  # [B,9]

            # tokenize to devices
            a_s = tokenize_to_device(student, a_txt, student_dev)
            p_s = tokenize_to_device(student, p_txt, student_dev)
            n_s = [tokenize_to_device(student, negs_by_k[j], student_dev) for j in range(9)]

            a_t = tokenize_to_device(teacher, a_txt, teacher_dev)
            p_t = tokenize_to_device(teacher, p_txt, teacher_dev)
            n_t = [tokenize_to_device(teacher, negs_by_k[j], teacher_dev) for j in range(9)]

            # ---- teacher forward (cuda:1): masks + teacher sims for KL ----
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                ea = st_sentence_embedding(teacher, a_t)
                ep = st_sentence_embedding(teacher, p_t)
                en = [st_sentence_embedding(teacher, n_t[j]) for j in range(9)]

                t_pos_sim = F.cosine_similarity(ea, ep, dim=-1)  # [B]
                t_neg_sim = torch.stack([F.cosine_similarity(ea, en[j], dim=-1) for j in range(9)], dim=1)  # [B,9]

                keep_fields = (t_neg_sim <= (t_pos_sim.unsqueeze(1) + cfg.margin))  # [B,9]

                if cfg.use_inbatch_neg and ea.size(0) > 1:
                    t_ap = cosine_mat(ea, ep)  # [B,B]
                    keep_inbatch_full = (t_ap <= (t_pos_sim.unsqueeze(1) + cfg.margin))  # [B,B]
                    eye_t = torch.eye(ea.size(0), dtype=torch.bool, device=t_ap.device)
                    keep_inbatch_full = keep_inbatch_full & (~eye_t)
                else:
                    t_ap = None
                    keep_inbatch_full = None

            # move masks to student + mask PAD
            keep_fields = keep_fields.to(student_dev)
            keep_fields = keep_fields & pad_mask
            keep_ratio_fields = keep_fields.float().mean().item()

            if keep_inbatch_full is not None:
                keep_inbatch_full = keep_inbatch_full.to(student_dev)

            # move teacher sims to student for KL (detached)
            if cfg.use_kl_distill:
                t_pos_sim_s = t_pos_sim.detach().to(student_dev, non_blocking=True)  # [B]
                t_neg_sim_s = t_neg_sim.detach().to(student_dev, non_blocking=True)  # [B,9]
                t_ap_s = t_ap.detach().to(student_dev, non_blocking=True) if t_ap is not None else None
            else:
                t_pos_sim_s = t_neg_sim_s = t_ap_s = None

            # ---- student forward (cuda:0): CE + KL distill ----
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                sa = st_sentence_embedding(student, a_s)
                sp = st_sentence_embedding(student, p_s)
                sn = [st_sentence_embedding(student, n_s[j]) for j in range(9)]

                s_pos_sim = F.cosine_similarity(sa, sp, dim=-1)  # [B]
                s_neg_sim = torch.stack([F.cosine_similarity(sa, sn[j], dim=-1) for j in range(9)], dim=1)  # [B,9]

                # CE logits
                pos_logits_ce = (s_pos_sim / cfg.temperature).unsqueeze(1)  # [B,1]
                neg_logits_ce = (s_neg_sim / cfg.temperature)               # [B,9]
                neg_logits_ce = mask_logits(neg_logits_ce, keep_fields)

                if cfg.use_inbatch_neg and sa.size(0) > 1:
                    s_ap = cosine_mat(sa, sp)  # [B,B]
                    eye_s = torch.eye(sa.size(0), dtype=torch.bool, device=s_ap.device)

                    inbatch_ce = s_ap[~eye_s].view(sa.size(0), sa.size(0) - 1) / cfg.temperature
                    inbatch_keep = keep_inbatch_full[~eye_s].view(sa.size(0), sa.size(0) - 1)
                    inbatch_ce = mask_logits(inbatch_ce, inbatch_keep)

                    logits_ce = torch.cat([pos_logits_ce, neg_logits_ce, inbatch_ce], dim=1)
                    keep_ratio_inbatch = inbatch_keep.float().mean().item()
                else:
                    s_ap = None
                    inbatch_keep = None
                    logits_ce = torch.cat([pos_logits_ce, neg_logits_ce], dim=1)
                    keep_ratio_inbatch = float("nan")

                target = torch.zeros(logits_ce.size(0), dtype=torch.long, device=logits_ce.device)
                loss_ce = F.cross_entropy(logits_ce, target)

                # KL distill on sims (pos + negs + inbatch), same masks
                if cfg.use_kl_distill and kl_w > 0.0:
                    pos_kd_s = s_pos_sim.unsqueeze(1)  # [B,1]
                    neg_kd_s = mask_logits(s_neg_sim, keep_fields)

                    pos_kd_t = t_pos_sim_s.unsqueeze(1)  # [B,1]
                    neg_kd_t = mask_logits(t_neg_sim_s, keep_fields)

                    if cfg.use_inbatch_neg and sa.size(0) > 1:
                        eye2 = torch.eye(sa.size(0), dtype=torch.bool, device=student_dev)

                        inbatch_kd_s = s_ap[~eye2].view(sa.size(0), sa.size(0) - 1)
                        inbatch_kd_s = mask_logits(inbatch_kd_s, inbatch_keep)

                        inbatch_kd_t = t_ap_s[~eye2].view(sa.size(0), sa.size(0) - 1)
                        inbatch_kd_t = mask_logits(inbatch_kd_t, inbatch_keep)

                        logits_kd_s = torch.cat([pos_kd_s, neg_kd_s, inbatch_kd_s], dim=1)
                        logits_kd_t = torch.cat([pos_kd_t, neg_kd_t, inbatch_kd_t], dim=1)
                    else:
                        logits_kd_s = torch.cat([pos_kd_s, neg_kd_s], dim=1)
                        logits_kd_t = torch.cat([pos_kd_t, neg_kd_t], dim=1)

                    loss_kd = kl_distill_loss(
                        student_logits=logits_kd_s,
                        teacher_logits=logits_kd_t.detach(),
                        fp32=cfg.kl_fp32,
                        T=cfg.kl_temperature,
                    )
                else:
                    loss_kd = torch.zeros((), device=student_dev)

                loss = (loss_ce + kl_w * loss_kd) / cfg.grad_accum

            # backward
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if global_step % cfg.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # logging
            loss_item = float(loss.item() * cfg.grad_accum)
            ce_item = float(loss_ce.detach().item())
            kd_item = float(loss_kd.detach().item()) if cfg.use_kl_distill else 0.0

            if cfg.use_inbatch_neg and cfg.batch_size > 1:
                pbar.set_postfix(
                    loss=f"{loss_item:.4f}",
                    ce=f"{ce_item:.4f}",
                    kd=f"{kd_item:.4f}",
                    klw=f"{kl_w:.3f}",
                    keepK9=f"{keep_ratio_fields:.3f}",
                    keepIB=f"{keep_ratio_inbatch:.3f}",
                    step=global_step,
                )
            else:
                pbar.set_postfix(
                    loss=f"{loss_item:.4f}",
                    ce=f"{ce_item:.4f}",
                    kd=f"{kd_item:.4f}",
                    klw=f"{kl_w:.3f}",
                    keepK9=f"{keep_ratio_fields:.3f}",
                    step=global_step,
                )

            if cfg.log_every_steps and global_step % cfg.log_every_steps == 0:
                if cfg.use_inbatch_neg and cfg.batch_size > 1:
                    print(
                        f"[train] step={global_step} loss={loss_item:.4f} ce={ce_item:.4f} kd={kd_item:.4f} klw={kl_w:.3f} "
                        f"keepK9={keep_ratio_fields:.3f} keepIB={keep_ratio_inbatch:.3f}"
                    )
                else:
                    print(
                        f"[train] step={global_step} loss={loss_item:.4f} ce={ce_item:.4f} kd={kd_item:.4f} klw={kl_w:.3f} "
                        f"keepK9={keep_ratio_fields:.3f}"
                    )

            # eval + save best
            if cfg.eval_every_steps and global_step % cfg.eval_every_steps == 0:
                res = run_eval(student, val_eval)
                metric_key, score = extract_primary_score(res, val_eval)
                print("[val] raw:", res)
                if score > best_score:
                    best_score = score
                    best_meta = {"epoch": epoch + 1, "step": global_step, "metric": metric_key, "score": best_score}
                    save_best(student, best_dir, best_meta)
                    print(f"[best] saved -> {best_dir} | {metric_key}={best_score:.6f} @ epoch={epoch+1} step={global_step}")

        # epoch end eval + save best
        res = run_eval(student, val_eval)
        metric_key, score = extract_primary_score(res, val_eval)
        print(f"[val] epoch_end={epoch+1} raw:", res)

        if score > best_score:
            best_score = score
            best_meta = {"epoch": epoch + 1, "step": global_step, "metric": metric_key, "score": best_score}
            save_best(student, best_dir, best_meta)
            print(f"[best] saved -> {best_dir} | {metric_key}={best_score:.6f} @ epoch={epoch+1} step={global_step}")

    final_dir = os.path.join(cfg.output_dir, "final_model")
    student.save(final_dir)
    print(f"[done] final saved -> {final_dir}")
    print(f"[done] best  saved -> {best_dir} | meta={best_meta}")


if __name__ == "__main__":
    main()
