"""Tier-4 Wave B QLoRA training, inference, and EntailmentGate evaluation harness."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import polars as pl
import structlog

from tacet.core.graph import Triple, WorldGraph
from tacet.core.symbolic import Rule, RuleEngine
from tacet.distill.distill import mine_rules
from tacet.distill.export import Arm, export_tier4_dataset
from tacet.eval.benchmark import BenchmarkConfig, generate
from tacet.eval.metrics import entailment_rate, grounded_fraction, proof_coverage_tier4
from tacet.eval.unseen import UnseenBenchmark, UnseenBenchmarkConfig, generate_unseen_benchmark
from tacet.tiers.base import TierResult
from tacet.tiers.gate import EntailmentGate

log = structlog.get_logger(__name__)

BASE_RELATIONS: frozenset[str] = frozenset(
    {"parent_of", "manages", "works_at", "lives_in", "member_of"}
)
DERIVED_RELATIONS: frozenset[str] = frozenset(
    {"ancestor_of", "superior_of", "colleague_of", "primary_language", "favourite_food"}
)
DEDUCTIVE_RELATIONS: frozenset[str] = frozenset({"ancestor_of", "superior_of", "colleague_of"})

DEFAULT_MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_MODEL_REVISION = "70d244cc86cc"

_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_PROOF_LINE_RE = re.compile(r"^\s*(FACT|DERIVED)\s+.+$")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class Predictor(Protocol):
    """Protocol for a Wave B model predictor producing raw completion strings."""

    def predict(self, prompts: Sequence[str]) -> list[str]: ...


_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class QLoRAConfig:
    """Hyperparameters locked in Wave B preregistration."""

    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = DEFAULT_MODEL_REVISION
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    max_seq_len: int = 2560
    max_new_tokens: int = 1536
    eval_batch_size: int = 16
    max_context_hops: int = 2
    max_context_facts: int = 60


def extract_local_context(
    graph: WorldGraph,
    head: str,
    target_relation: str,
    *,
    max_hops: int = 2,
    max_facts: int = 60,
) -> list[Triple]:
    """Extract deterministic k-hop base facts around ``head`` with zero label leakage."""
    collected: list[Triple] = []
    seen_triples: set[Triple] = set()
    visited_nodes: set[str] = {head}
    queue: deque[tuple[str, int]] = deque([(head, 0)])

    while queue and len(collected) < max_facts:
        node, depth = queue.popleft()
        if depth >= max_hops:
            continue
        hop_triples: list[Triple] = []
        next_nodes: list[str] = []

        for rel, tails in sorted(graph.out_relations(node).items()):
            if rel not in BASE_RELATIONS or rel in DERIVED_RELATIONS or rel == target_relation:
                continue
            for tail in sorted(tails):
                hop_triples.append((node, rel, tail))
                if tail not in visited_nodes:
                    next_nodes.append(tail)

        for rel, sources in sorted(graph.in_relations(node).items()):
            if rel not in BASE_RELATIONS or rel in DERIVED_RELATIONS or rel == target_relation:
                continue
            for src in sorted(sources):
                hop_triples.append((src, rel, node))
                if src not in visited_nodes:
                    next_nodes.append(src)

        for triple in sorted(set(hop_triples)):
            if triple in seen_triples:
                continue
            seen_triples.add(triple)
            collected.append(triple)
            if len(collected) >= max_facts:
                break

        for nxt in sorted(set(next_nodes)):
            if nxt not in visited_nodes:
                visited_nodes.add(nxt)
                queue.append((nxt, depth + 1))

    for h, rel, t in collected:
        if rel == target_relation or rel in DERIVED_RELATIONS or rel not in BASE_RELATIONS:
            raise ValueError(f"leaked disallowed relation in local context: {(h, rel, t)!r}")
    return collected


def compact_proof_lines(proof: Sequence[str]) -> list[str]:
    """Deduplicate proof lines and collapse excess whitespace without losing steps."""
    seen: set[str] = set()
    compacted: list[str] = []
    for raw_line in proof:
        norm = _WS_RE.sub(" ", str(raw_line).strip())
        if norm and norm not in seen:
            seen.add(norm)
            compacted.append(norm)
    return compacted


def build_sft_prompt(
    head: str,
    relation: str,
    context_facts: Sequence[Triple],
    *,
    arm: str = "B",
) -> str:
    """Render deterministic prompt from non-leaking base facts for Tier-4 SFT/eval."""
    facts_block = "\n".join(f"  {h} -{r}-> {t}" for h, r, t in context_facts) or "  (none)"
    if arm == "A":
        fmt_hint = 'Output only JSON: {"answers": [<strings>]}.'
    elif arm == "C":
        fmt_hint = (
            "First write grounded proof lines (FACT/DERIVED) if entailed, then output JSON: "
            '{"answers": [<strings>], "abstain": <true|false>}.'
        )
    else:
        fmt_hint = (
            "First write grounded proof lines (FACT/DERIVED), then output JSON: "
            '{"answers": [<strings>]}.'
        )
    return (
        f"Query: ({head}, {relation}, ?)\n"
        f"BaseFacts:\n{facts_block}\n"
        f"Instruction: {fmt_hint}\n"
        "Response:"
    )


def format_sft_target(record: Mapping[str, Any], arm: Arm) -> str:
    """Format canonical training completion string for Arm A / B / C."""
    answers = [str(a) for a in record.get("answers", [])]
    abstain = bool(record.get("abstain", False))
    proof = compact_proof_lines(record.get("proof", []))

    if arm == "A":
        return json.dumps({"answers": answers}, ensure_ascii=False)
    if arm == "C":
        payload = json.dumps({"answers": answers, "abstain": abstain}, ensure_ascii=False)
        if proof and not abstain:
            return "\n".join(proof) + "\n" + payload
        return payload
    payload = json.dumps({"answers": answers}, ensure_ascii=False)
    if proof:
        return "\n".join(proof) + "\n" + payload
    return payload


def parse_model_output_detailed(text: str) -> tuple[list[str], list[str], bool, bool]:
    """Parse proof lines, answers, abstain flag, and whether valid JSON was parsed."""
    cleaned = _THINK_BLOCK_RE.sub("", text).strip()
    proof_lines = [line.strip() for line in cleaned.splitlines() if _PROOF_LINE_RE.match(line)]

    answers: list[str] = []
    abstain = False
    matches = _JSON_OBJ_RE.findall(cleaned)
    for raw_json in reversed(matches):
        try:
            obj = json.loads(raw_json)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("answers" in obj or "abstain" in obj):
            raw_ans = obj.get("answers", [])
            if isinstance(raw_ans, str):
                answers = [raw_ans] if raw_ans else []
            elif isinstance(raw_ans, list):
                answers = sorted({str(x).strip() for x in raw_ans if str(x).strip()})
            abstain = bool(obj.get("abstain", False)) or (len(answers) == 0 and "abstain" in obj)
            return answers, proof_lines, abstain, True

    return [], proof_lines, False, False


def parse_model_output(text: str) -> tuple[list[str], list[str], bool]:
    """Parse proof lines, answers list, and abstain flag from raw model text."""
    answers, proof_lines, abstain, _ = parse_model_output_detailed(text)
    return answers, proof_lines, abstain


def discover_train_rules(
    seed: int,
    train_entities: frozenset[str],
    eval_entities: frozenset[str],
    train_graph: WorldGraph,
) -> list[Rule]:
    """Mine rules strictly on ``train_graph`` and ``train_entities`` without eval leakage."""
    base = generate(BenchmarkConfig(seed=seed))
    teacher_facts: set[Triple] = set()
    complete_heads: dict[str, set[str]] = {"superior_of": set(), "colleague_of": set()}

    for h in sorted(train_entities):
        for rel in ("superior_of", "colleague_of"):
            complete_heads[rel].add(h)
            for t in base.oracle(h, rel):
                if t in eval_entities:
                    continue
                if t in train_entities:
                    teacher_facts.add((h, rel, t))

    mined: list[Rule] = []
    seen_names: set[str] = set()
    for rel in ("superior_of", "colleague_of"):
        rules = mine_rules(
            train_graph,
            teacher_facts,
            rel,
            min_confidence=0.95,
            min_support=3,
            complete_heads=complete_heads[rel],
            allowed_body=set(BASE_RELATIONS),
            allow_target_in_body=True,
            forbid_target_self_loop=True,
        )
        for m in rules:
            if m.rule.name not in seen_names:
                seen_names.add(m.rule.name)
                mined.append(m.rule)
    return mined


def build_engines_for_split(split: UnseenBenchmark, seed: int) -> tuple[RuleEngine, RuleEngine]:
    """Build materialised train and eval ``RuleEngine`` instances without eval rule discovery."""
    train_mined = discover_train_rules(
        seed,
        split.train_entities,
        split.eval_entities,
        split.train_graph,
    )
    all_rules = list(split.given_rules) + train_mined
    train_engine = RuleEngine(split.ontology, all_rules)
    train_engine.materialise(split.train_graph)

    eval_engine = RuleEngine(split.ontology, all_rules)
    eval_engine.materialise(split.eval_graph)
    return train_engine, eval_engine


def _build_train_truth(
    split: UnseenBenchmark,
    train_engine: RuleEngine,
) -> dict[tuple[str, str], list[str]]:
    merged: dict[tuple[str, str], set[str]] = {}
    for (h, r), answers in split.truth.items():
        if h in split.train_entities:
            merged.setdefault((h, r), set()).update(
                a for a in answers if a not in split.eval_entities
            )
    for h, r, t in train_engine.closure:
        if h in split.train_entities and t not in split.eval_entities:
            merged.setdefault((h, r), set()).add(t)
    return {k: sorted(v) for k, v in merged.items() if v}


def build_unanswerable_eval_queries(
    split: UnseenBenchmark,
    eval_engine: RuleEngine,
    *,
    max_queries: int = 60,
) -> list[tuple[str, str]]:
    """Collect deterministic empty-truth queries on ``eval_entities`` for abstain eval."""
    queries: list[tuple[str, str]] = []
    for head in sorted(split.eval_entities):
        for rel in ("ancestor_of", "superior_of", "colleague_of"):
            if split.truth.get((head, rel)):
                continue
            if eval_engine.query(head, rel).answered:
                continue
            queries.append((head, rel))
            if len(queries) >= max_queries:
                return queries
    return queries


def _f1_sets(pred: Iterable[str], gold: Iterable[str]) -> float:
    p_set = set(pred)
    g_set = set(gold)
    if not p_set and not g_set:
        return 1.0
    if not p_set or not g_set:
        return 0.0
    inter = len(p_set & g_set)
    if inter == 0:
        return 0.0
    prec = inter / len(p_set)
    rec = inter / len(g_set)
    return 2.0 * prec * rec / (prec + rec)


class _DictPredictorTier:
    tier_id = 4
    name = "qlora-tier4"

    def __init__(
        self, outputs: Mapping[tuple[str, str], tuple[list[str], list[str], bool]]
    ) -> None:
        self._outputs = outputs

    def can_handle(self, head: str, relation: str) -> bool:
        return (head, relation) in self._outputs

    def answer(self, head: str, relation: str) -> TierResult | None:
        entry = self._outputs.get((head, relation))
        if entry is None:
            return None
        answers, proof, abstain = entry
        if abstain or not answers:
            return None
        return TierResult(tier=4, answers=list(answers), confidence=1.0, proof=list(proof))


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        payload = "".join(
            json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        )
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(
            json.dumps(dict(obj), sort_keys=True, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


class MockPredictor:
    """Deterministic CPU predictor for ``--dry-run`` end-to-end verification."""

    def __init__(self, split: UnseenBenchmark, arm: Arm) -> None:
        self._split = split
        self._arm = arm
        self._eval_facts_by_query: dict[tuple[str, str], list[str]] = {}
        for h, r, t in split.eval_graph.triples():
            self._eval_facts_by_query.setdefault((h, r), []).append(t)

    def predict(self, prompts: Sequence[str]) -> list[str]:
        out: list[str] = []
        for prompt in prompts:
            first_line = prompt.splitlines()[0]
            # Format: "Query: (<head>, <relation>, ?)"
            inner = first_line[len("Query: (") : -len(", ?)")]
            head, relation = [part.strip() for part in inner.split(",", 1)]
            if relation == "ancestor_of":
                ctx = extract_local_context(self._split.eval_graph, head, relation, max_hops=1)
                answers = sorted({t for h, r, t in ctx if h == head and r == "parent_of"})
            elif relation == "superior_of":
                ctx = extract_local_context(self._split.eval_graph, head, relation, max_hops=1)
                answers = sorted({t for h, r, t in ctx if h == head and r == "manages"})
            else:
                answers = sorted(self._eval_facts_by_query.get((head, relation), []))

            if self._arm == "A":
                out.append(json.dumps({"answers": answers}))
            elif self._arm == "C":
                abstain = len(answers) == 0
                lines = [f"FACT {head} -{relation}-> {a}" for a in answers] if answers else []
                payload = json.dumps({"answers": answers, "abstain": abstain})
                out.append("\n".join([*lines, payload]))
            else:
                lines = [f"FACT {head} -{relation}-> {a}" for a in answers]
                payload = json.dumps({"answers": answers})
                out.append("\n".join([*lines, payload]))
        return out


class QLoRATrainerBackend:
    """4-bit NF4 QLoRA fine-tuning and greedy decoding on ``Qwen/Qwen3-1.7B``."""

    def __init__(self, cfg: QLoRAConfig, arm: Arm, seed: int) -> None:
        self.cfg = cfg
        self.arm = arm
        self.seed = seed
        self._model: Any = None
        self._tokenizer: Any = None
        self.truncated_train_examples: int = 0

    def _format_chat(self, prompt: str) -> str:
        assert self._tokenizer is not None
        messages = [{"role": "user", "content": prompt}]
        try:
            return str(
                self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        except TypeError:
            return prompt

    def fit(
        self, train_examples: Sequence[tuple[str, str]], adapter_dir: Path | None = None
    ) -> float:
        """Fine-tune a fresh 4-bit NF4 LoRA adapter on ``train_examples``."""
        import random

        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_id,
            revision=self.cfg.model_revision,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer

        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            revision=self.cfg.model_revision,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=compute_dtype,
        )
        try:
            model = prepare_model_for_kbit_training(
                model,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        except TypeError:
            model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora_cfg)
        model.train()
        self._model = model

        encoded: list[tuple[list[int], list[int]]] = []
        truncated_count = 0
        eos_id = tokenizer.eos_token_id
        for prompt, target in train_examples:
            prompt_text = self._format_chat(prompt)
            p_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            t_ids = tokenizer(target, add_special_tokens=False).input_ids + (
                [eos_id] if eos_id is not None else []
            )
            if len(p_ids) + len(t_ids) > self.cfg.max_seq_len:
                truncated_count += 1
            full_ids = (p_ids + t_ids)[: self.cfg.max_seq_len]
            labels = ([-100] * min(len(p_ids), len(full_ids))) + full_ids[
                min(len(p_ids), len(full_ids)) :
            ]
            if any(lbl != -100 for lbl in labels):
                encoded.append((full_ids, labels))
        self.truncated_train_examples = truncated_count
        log.info(
            "wave_b.train_tokenized",
            arm=self.arm,
            seed=self.seed,
            n_examples=len(encoded),
            truncated=truncated_count,
            max_seq_len=self.cfg.max_seq_len,
            compute_dtype=str(compute_dtype),
        )

        rng = random.Random(self.seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.learning_rate)
        steps_per_epoch = max(1, (len(encoded) + self.cfg.batch_size - 1) // self.cfg.batch_size)
        total_opt_steps = max(1, (steps_per_epoch * self.cfg.epochs) // max(1, self.cfg.grad_accum))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_opt_steps)
        scaler = torch.amp.GradScaler(
            "cuda", enabled=(torch.cuda.is_available() and compute_dtype == torch.float16)
        )

        pad_id = tokenizer.pad_token_id
        device = next(model.parameters()).device
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        micro_step = 0

        for epoch in range(self.cfg.epochs):
            order = list(range(len(encoded)))
            rng.shuffle(order)
            epoch_loss = 0.0
            epoch_steps = 0
            for i in range(0, len(order), self.cfg.batch_size):
                batch_idx = order[i : i + self.cfg.batch_size]
                max_len = max(len(encoded[j][0]) for j in batch_idx)
                input_rows = []
                attn_rows = []
                label_rows = []
                for j in batch_idx:
                    ids, lbls = encoded[j]
                    pad_len = max_len - len(ids)
                    input_rows.append(ids + [pad_id] * pad_len)
                    attn_rows.append([1] * len(ids) + [0] * pad_len)
                    label_rows.append(lbls + [-100] * pad_len)

                input_t = torch.tensor(input_rows, dtype=torch.long, device=device)
                attn_t = torch.tensor(attn_rows, dtype=torch.long, device=device)
                label_t = torch.tensor(label_rows, dtype=torch.long, device=device)

                with torch.amp.autocast(
                    "cuda", dtype=compute_dtype, enabled=torch.cuda.is_available()
                ):
                    raw_loss = model(input_ids=input_t, attention_mask=attn_t, labels=label_t).loss
                    loss = raw_loss / self.cfg.grad_accum
                epoch_loss += float(raw_loss.detach().item())
                epoch_steps += 1
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                micro_step += 1
                if micro_step % self.cfg.grad_accum == 0 or (i + self.cfg.batch_size) >= len(order):
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            log.info(
                "wave_b.train_epoch",
                arm=self.arm,
                seed=self.seed,
                epoch=epoch + 1,
                epochs=self.cfg.epochs,
                avg_loss=round(epoch_loss / max(1, epoch_steps), 4),
                elapsed_s=round(time.perf_counter() - t0, 2),
            )

        if adapter_dir is not None:
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
        return time.perf_counter() - t0

    def predict(self, prompts: Sequence[str]) -> list[str]:
        """Run deterministic left-padded batched greedy decoding with KV cache."""
        import torch

        assert self._model is not None and self._tokenizer is not None
        self._model.eval()
        if hasattr(self._model, "gradient_checkpointing_disable"):
            self._model.gradient_checkpointing_disable()
        if hasattr(self._model, "config"):
            self._model.config.use_cache = True
        if getattr(self._model, "generation_config", None) is not None:
            self._model.generation_config.use_cache = True

        device = next(self._model.parameters()).device
        bs = max(1, int(self.cfg.eval_batch_size))
        outputs: list[str] = []
        eval_t0 = time.perf_counter()
        orig_padding_side = getattr(self._tokenizer, "padding_side", "right")
        self._tokenizer.padding_side = "left"
        try:
            with torch.no_grad():
                for batch_idx, start in enumerate(range(0, len(prompts), bs)):
                    batch_texts = [self._format_chat(p) for p in prompts[start : start + bs]]
                    inputs = self._tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.cfg.max_seq_len,
                    ).to(device)
                    in_len = int(inputs["input_ids"].shape[1])
                    gen = self._model.generate(
                        **inputs,
                        do_sample=False,
                        use_cache=True,
                        max_new_tokens=self.cfg.max_new_tokens,
                        pad_token_id=self._tokenizer.pad_token_id,
                    )
                    for row_idx in range(len(batch_texts)):
                        decoded = self._tokenizer.decode(
                            gen[row_idx, in_len:],
                            skip_special_tokens=True,
                        )
                        outputs.append(decoded)
                    if (batch_idx + 1) % 4 == 0 or len(outputs) == len(prompts):
                        log.info(
                            "wave_b.eval_progress",
                            arm=self.arm,
                            seed=self.seed,
                            done=len(outputs),
                            total=len(prompts),
                            elapsed_s=round(time.perf_counter() - eval_t0, 2),
                        )
        finally:
            self._tokenizer.padding_side = orig_padding_side
        return outputs

    def cleanup(self) -> None:
        """Release GPU references between arms/seeds."""
        import gc

        import torch

        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _evaluate_run(
    *,
    arm: Arm,
    seed: int,
    split: UnseenBenchmark,
    eval_engine: RuleEngine,
    positive_queries: Sequence[tuple[str, str]],
    unanswerable_queries: Sequence[tuple[str, str]],
    raw_pos_outputs: Sequence[str],
    raw_neg_outputs: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed_map: dict[tuple[str, str], tuple[list[str], list[str], bool]] = {}
    pos_json_parsed: dict[tuple[str, str], bool] = {}
    pred_rows: list[dict[str, Any]] = []

    for (h, r), raw_text in zip(positive_queries, raw_pos_outputs, strict=True):
        answers, proof_lines, abstain, json_parsed = parse_model_output_detailed(raw_text)
        parsed_map[(h, r)] = (answers, proof_lines, abstain)
        pos_json_parsed[(h, r)] = json_parsed

    gate_strict = EntailmentGate(_DictPredictorTier(parsed_map), eval_engine, strict=True)
    gate_soft = EntailmentGate(_DictPredictorTier(parsed_map), eval_engine, strict=False)

    exact_hits = 0
    f1_scores: list[float] = []
    ded_exact_hits = 0
    ded_total = 0
    pos_parse_fails = 0
    ded_parse_fails = 0
    raw_for_entail: list[dict[str, Any]] = []
    ded_raw_for_entail: list[dict[str, Any]] = []
    gated_records: list[dict[str, Any]] = []
    ded_gated_records: list[dict[str, Any]] = []

    for (h, r), raw_text in zip(positive_queries, raw_pos_outputs, strict=True):
        pred_ans, pred_proof, abstain = parsed_map[(h, r)]
        json_parsed = pos_json_parsed[(h, r)]
        if not json_parsed:
            pos_parse_fails += 1
        gold = split.truth.get((h, r), [])
        gated_res = gate_soft.answer(h, r)
        strict_res = gate_strict.answer(h, r)
        is_entailed = strict_res is not None
        is_grounded = bool(gated_res is not None and gated_res.grounded)
        em = json_parsed and set(pred_ans) == set(gold)
        f1 = _f1_sets(pred_ans, gold) if json_parsed else 0.0
        exact_hits += int(em)
        f1_scores.append(f1)

        rec = {
            "arm": arm,
            "seed": seed,
            "split": "unseen_entity",
            "head": h,
            "relation": r,
            "tier": 4,
            "gold": gold,
            "answers": pred_ans,
            "proof": pred_proof if pred_proof else (list(strict_res.proof) if strict_res else []),
            "model_proof_lines": len(pred_proof),
            "json_parsed": json_parsed,
            "abstain": json_parsed and abstain,
            "exact_match": em,
            "f1": f1,
            "entailed": is_entailed,
            "grounded": is_grounded,
            "raw_output": raw_text,
        }
        pred_rows.append(rec)
        raw_for_entail.append({"head": h, "relation": r, "answers": pred_ans})
        gated_records.append(rec)
        if r in DEDUCTIVE_RELATIONS:
            ded_total += 1
            if not json_parsed:
                ded_parse_fails += 1
            ded_exact_hits += int(em)
            ded_raw_for_entail.append({"head": h, "relation": r, "answers": pred_ans})
            ded_gated_records.append(rec)

    true_abstain = 0
    missed_abstain = 0
    false_abstain = 0
    neg_parse_fails = 0
    for (h, r), raw_text in zip(unanswerable_queries, raw_neg_outputs, strict=True):
        pred_ans, pred_proof, abstain, json_parsed = parse_model_output_detailed(raw_text)
        if not json_parsed:
            neg_parse_fails += 1
        did_abstain = json_parsed and (abstain or len(pred_ans) == 0)
        if did_abstain:
            true_abstain += 1
        else:
            missed_abstain += 1
        pred_rows.append(
            {
                "arm": arm,
                "seed": seed,
                "split": "unanswerable",
                "head": h,
                "relation": r,
                "tier": 4,
                "gold": [],
                "answers": pred_ans,
                "proof": pred_proof,
                "model_proof_lines": len(pred_proof),
                "json_parsed": json_parsed,
                "abstain": did_abstain,
                "exact_match": did_abstain,
                "f1": 1.0 if did_abstain else 0.0,
                "entailed": False,
                "grounded": False,
                "raw_output": raw_text,
            }
        )

    for h, r in positive_queries:
        pred_ans, _, abstain = parsed_map[(h, r)]
        if pos_json_parsed[(h, r)] and (abstain or len(pred_ans) == 0):
            false_abstain += 1

    abstain_rec = (
        true_abstain / (true_abstain + missed_abstain) if (true_abstain + missed_abstain) else 0.0
    )
    abstain_prec = (
        true_abstain / (true_abstain + false_abstain) if (true_abstain + false_abstain) else 0.0
    )
    abstain_f1 = (
        2.0 * abstain_prec * abstain_rec / (abstain_prec + abstain_rec)
        if (abstain_prec + abstain_rec)
        else 0.0
    )
    n_neg = len(unanswerable_queries)
    false_answer_rate = missed_abstain / n_neg if n_neg else 0.0

    n_pos = len(positive_queries)
    summary = {
        "arm": arm,
        "seed": seed,
        "n_positive": n_pos,
        "n_deductive": ded_total,
        "n_unanswerable": n_neg,
        "overall_exact_match": exact_hits / n_pos if n_pos else 0.0,
        "overall_answer_f1": sum(f1_scores) / n_pos if n_pos else 0.0,
        "overall_entailment_rate": entailment_rate(raw_for_entail, eval_engine),
        "overall_grounded_fraction": grounded_fraction(gated_records),
        "deductive_exact_match": ded_exact_hits / ded_total if ded_total else 0.0,
        "deductive_entailment_rate": entailment_rate(ded_raw_for_entail, eval_engine),
        "deductive_grounded_fraction": grounded_fraction(ded_gated_records),
        "proof_coverage_tier4": proof_coverage_tier4(gated_records),
        "model_proof_coverage": (
            sum(1 for r in gated_records if r["model_proof_lines"] > 0) / n_pos if n_pos else 0.0
        ),
        "abstain_f1": abstain_f1,
        "false_answer_rate": false_answer_rate,
        "pos_parse_fail_rate": pos_parse_fails / n_pos if n_pos else 0.0,
        "ded_parse_fail_rate": ded_parse_fails / ded_total if ded_total else 0.0,
        "neg_parse_fail_rate": neg_parse_fails / n_neg if n_neg else 0.0,
    }
    return pred_rows, summary


def summarize_wave_b_jsonl(summaries: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    """Aggregate per-(arm, seed) metrics using ``polars``."""
    if not summaries:
        return pl.DataFrame()
    df = pl.DataFrame(list(summaries))
    agg_exprs = [
        pl.len().alias("n_seeds"),
        pl.col("deductive_entailment_rate").mean().alias("mean_ded_entail"),
        pl.col("deductive_grounded_fraction").mean().alias("mean_ded_grounded"),
        pl.col("deductive_exact_match").mean().alias("mean_ded_em"),
        pl.col("overall_exact_match").mean().alias("mean_overall_em"),
        pl.col("overall_answer_f1").mean().alias("mean_overall_f1"),
        pl.col("model_proof_coverage").mean().alias("mean_model_proof_cov"),
        pl.col("abstain_f1").mean().alias("mean_abstain_f1"),
        pl.col("false_answer_rate").mean().alias("mean_false_ans_rate"),
        pl.col("train_wall_s").sum().alias("total_train_s"),
    ]
    if "pos_parse_fail_rate" in df.columns:
        agg_exprs.append(pl.col("pos_parse_fail_rate").mean().alias("mean_pos_parse_fail"))
    if "ded_parse_fail_rate" in df.columns:
        agg_exprs.append(pl.col("ded_parse_fail_rate").mean().alias("mean_ded_parse_fail"))
    if "neg_parse_fail_rate" in df.columns:
        agg_exprs.append(pl.col("neg_parse_fail_rate").mean().alias("mean_neg_parse_fail"))
    return df.group_by("arm").agg(agg_exprs).sort("arm")


def run_wave_b_matrix(
    *,
    arms: Sequence[Arm],
    seeds: Sequence[int],
    output_dir: str | Path,
    cfg: QLoRAConfig | None = None,
    dry_run: bool = False,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> dict[str, Any]:
    """Run Tier-4 Wave B train/eval across ``arms`` x ``seeds`` with atomic resume."""
    conf = cfg or QLoRAConfig()
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    export_dir = out_root / "exports"
    manifest_path = out_root / "manifest.json"
    summary_path = out_root / "summary.jsonl"

    summaries_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    if summary_path.exists():
        for row in _read_jsonl(summary_path):
            summaries_by_key[(str(row["arm"]), int(row["seed"]))] = row

    for seed in seeds:
        split = generate_unseen_benchmark(
            "unseen_entity",
            UnseenBenchmarkConfig(seed=seed, holdout_entity_fraction=0.25),
        )
        train_engine, eval_engine = build_engines_for_split(split, seed)
        train_truth = _build_train_truth(split, train_engine)
        unanswerable = build_unanswerable_eval_queries(split, eval_engine, max_queries=60)
        pos_queries = list(split.workload)
        if max_eval_samples is not None:
            pos_queries = pos_queries[:max_eval_samples]
            unanswerable = unanswerable[: max(1, min(len(unanswerable), max_eval_samples))]

        for arm in arms:
            pred_path = out_root / f"predictions_arm_{arm}_s{seed}.jsonl"
            if pred_path.exists() and (arm, seed) in summaries_by_key:
                log.info("wave_b.skip_completed", arm=arm, seed=seed, path=str(pred_path))
                continue

            exp_manifest = export_tier4_dataset(
                split.train_graph,
                train_engine,
                train_truth,
                split.train_entities,
                split.eval_entities,
                export_dir,
                arm=arm,
                seed=seed,
            )
            train_rows = _read_jsonl(Path(exp_manifest.jsonl_path))
            if max_train_samples is not None:
                train_rows = train_rows[:max_train_samples]

            train_examples: list[tuple[str, str]] = []
            for rec in train_rows:
                ctx = extract_local_context(
                    split.train_graph,
                    rec["head"],
                    rec["relation"],
                    max_hops=conf.max_context_hops,
                    max_facts=conf.max_context_facts,
                )
                prompt = build_sft_prompt(rec["head"], rec["relation"], ctx, arm=arm)
                target = format_sft_target(rec, arm)
                train_examples.append((prompt, target))

            pos_prompts = [
                build_sft_prompt(
                    h,
                    r,
                    extract_local_context(
                        split.eval_graph,
                        h,
                        r,
                        max_hops=conf.max_context_hops,
                        max_facts=conf.max_context_facts,
                    ),
                    arm=arm,
                )
                for h, r in pos_queries
            ]
            neg_prompts = [
                build_sft_prompt(
                    h,
                    r,
                    extract_local_context(
                        split.eval_graph,
                        h,
                        r,
                        max_hops=conf.max_context_hops,
                        max_facts=conf.max_context_facts,
                    ),
                    arm=arm,
                )
                for h, r in unanswerable
            ]

            t0 = time.perf_counter()
            if dry_run:
                predictor: Predictor = MockPredictor(split, arm)
                train_wall_s = 0.0
                raw_pos = predictor.predict(pos_prompts)
                raw_neg = predictor.predict(neg_prompts)
                peak_vram_gib = 0.0
                truncated_train_examples = 0
            else:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                backend = QLoRATrainerBackend(conf, arm, seed)
                adapter_dir = out_root / "adapters" / f"arm_{arm}_s{seed}"
                train_wall_s = backend.fit(train_examples, adapter_dir=adapter_dir)
                raw_pos = backend.predict(pos_prompts)
                raw_neg = backend.predict(neg_prompts)
                peak_vram_gib = (
                    torch.cuda.max_memory_allocated() / 1024**3
                    if torch.cuda.is_available()
                    else 0.0
                )
                truncated_train_examples = backend.truncated_train_examples
                backend.cleanup()
            total_wall_s = time.perf_counter() - t0

            pred_rows, summary = _evaluate_run(
                arm=arm,
                seed=seed,
                split=split,
                eval_engine=eval_engine,
                positive_queries=pos_queries,
                unanswerable_queries=unanswerable,
                raw_pos_outputs=raw_pos,
                raw_neg_outputs=raw_neg,
            )
            summary["train_examples"] = len(train_examples)
            summary["truncated_train_examples"] = truncated_train_examples
            summary["train_wall_s"] = round(train_wall_s, 3)
            summary["total_wall_s"] = round(total_wall_s, 3)
            summary["peak_vram_gib"] = round(peak_vram_gib, 3)
            summary["export_sha256"] = exp_manifest.sha256
            summary["dry_run"] = dry_run

            _atomic_write_jsonl(pred_path, pred_rows)
            summary["predictions_sha256"] = hashlib.sha256(pred_path.read_bytes()).hexdigest()
            summaries_by_key[(arm, seed)] = summary
            ordered_summaries = [summaries_by_key[k] for k in sorted(summaries_by_key)]
            _atomic_write_jsonl(summary_path, ordered_summaries)
            log.info(
                "wave_b.completed_run",
                arm=arm,
                seed=seed,
                ded_entail=round(summary["deductive_entailment_rate"], 4),
                overall_em=round(summary["overall_exact_match"], 4),
                abstain_f1=round(summary["abstain_f1"], 4),
                pos_parse_fail=round(summary["pos_parse_fail_rate"], 4),
                neg_parse_fail=round(summary["neg_parse_fail_rate"], 4),
                wall_s=round(total_wall_s, 2),
            )

    all_summaries = [summaries_by_key[k] for k in sorted(summaries_by_key)]
    agg_df = summarize_wave_b_jsonl(all_summaries)
    manifest = {
        "dry_run": dry_run,
        "config": asdict(conf),
        "arms": list(arms),
        "seeds": [int(s) for s in seeds],
        "runs": all_summaries,
        "aggregate": agg_df.to_dicts() if agg_df.height > 0 else [],
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest
