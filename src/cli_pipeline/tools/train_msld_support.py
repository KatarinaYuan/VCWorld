#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train support-supervised latent mechanism distillation (MSLD) heads.

Core rule:
- adaptation uses support/train examples only
- graph verification is support-time teacher only
- this script outputs phi_star (small heads), not test predictions
"""

from __future__ import annotations

import argparse
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from msld_data import MSLDExample, build_query_prompt_ids, group_examples_by_perturbation, load_msld_examples
from msld_graph import (
    MSLDGraph,
    compute_drug_prior,
    compute_support_evidence,
    load_msld_graph,
    verify_mechanism_posterior,
)
from msld_model import MSLDHeadConfig, MSLDHeads, soft_target_cross_entropy


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def _score_label_candidates(
    model,
    device: torch.device,
    prompt_ids: List[int],
    prefix_ids: List[int],
    candidate_id_lists: List[List[int]],
) -> List[float]:
    scores: List[float] = []
    base_ids = prompt_ids + prefix_ids
    for cand_ids in candidate_id_lists:
        if not cand_ids:
            scores.append(float("-inf"))
            continue
        full_ids = base_ids + cand_ids
        input_t = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=input_t, attention_mask=torch.ones_like(input_t)).logits
            log_probs = F.log_softmax(logits, dim=-1)
        start = len(base_ids)
        s = 0.0
        for j, tok in enumerate(cand_ids):
            pos = start + j - 1
            s += float(log_probs[0, pos, tok].detach().cpu())
        scores.append(s)
    return scores


def _extract_base_scores(label_candidates: List[str], scores: List[float]) -> Dict[str, float]:
    out = {k.strip().lower(): float(v) for k, v in zip(label_candidates, scores)}
    out.setdefault("yes", float("-inf"))
    out.setdefault("no", float("-inf"))
    out.setdefault("insufficient", float("-inf"))
    return out


def _encode_hidden(model, device: torch.device, prompt_ids: List[int]) -> np.ndarray:
    input_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(
            input_ids=input_t,
            attention_mask=torch.ones_like(input_t),
            output_hidden_states=True,
            return_dict=True,
        )
    if out.hidden_states is None or len(out.hidden_states) == 0:
        raise RuntimeError("Model did not return hidden states")
    h = out.hidden_states[-1][0].mean(dim=0).float().detach().cpu().numpy().astype(np.float32)
    return h


def _precompute_support_features(
    *,
    model,
    tokenizer,
    model_device: torch.device,
    examples: List[MSLDExample],
    max_input_tokens: int,
    label_candidates: List[str],
    label_prefix: str,
) -> Dict[str, np.ndarray]:
    prefix_ids = tokenizer(label_prefix, add_special_tokens=False).input_ids
    candidate_id_lists = [
        tokenizer(" " + cand if not cand.startswith(" ") else cand, add_special_tokens=False).input_ids
        for cand in label_candidates
    ]
    hidden_list: List[np.ndarray] = []
    labels: List[float] = []
    gene_ids: List[int] = []
    drug_ids: List[int] = []
    base_margin: List[float] = []
    base_yes_prob: List[float] = []

    for i, ex in enumerate(examples, 1):
        prompt_ids = build_query_prompt_ids(
            tokenizer=tokenizer,
            example=ex,
            max_input_tokens=max_input_tokens,
        )
        h = _encode_hidden(model=model, device=model_device, prompt_ids=prompt_ids)
        scores = _score_label_candidates(
            model=model,
            device=model_device,
            prompt_ids=prompt_ids,
            prefix_ids=prefix_ids,
            candidate_id_lists=candidate_id_lists,
        )
        score_map = _extract_base_scores(label_candidates, scores)
        s_yes = float(score_map["yes"])
        s_no = float(score_map["no"])
        margin = s_yes - s_no
        max_v = max(s_yes, s_no)
        py = math.exp(s_yes - max_v)
        pn = math.exp(s_no - max_v)
        p_yes = py / max(py + pn, 1e-8)

        hidden_list.append(h)
        labels.append(float(ex.label if ex.label is not None else 0))
        gene_ids.append(int(ex.gene_id))
        drug_ids.append(int(ex.drug_id))
        base_margin.append(float(margin))
        base_yes_prob.append(float(p_yes))

        if i % 200 == 0 or i == len(examples):
            print(f"[MSLD] Precompute support features: {i}/{len(examples)}")

    return {
        "hidden": np.asarray(hidden_list, dtype=np.float32),
        "label": np.asarray(labels, dtype=np.float32),
        "gene_id": np.asarray(gene_ids, dtype=np.int64),
        "drug_id": np.asarray(drug_ids, dtype=np.int64),
        "base_margin": np.asarray(base_margin, dtype=np.float32),
        "base_yes_prob": np.asarray(base_yes_prob, dtype=np.float32),
    }


def run(args) -> None:
    _set_seed(args.seed)
    print("[MSLD] Loading graph")
    graph: MSLDGraph = load_msld_graph(
        kg_dir=args.kg_dir,
        alpha=args.graph_alpha,
        diffusion_hops=args.graph_diffusion_hops,
        diffusion_decay=args.graph_diffusion_decay,
        max_modules=args.max_modules,
        nodes_file=args.kg_nodes_file,
        edges_file=args.kg_edges_file,
        graph_file=args.kg_graph_file,
    )
    if graph.num_modules <= 0:
        raise RuntimeError("No modules parsed from KG")

    print("[MSLD] Loading support/train examples")
    support_examples = load_msld_examples(
        prompts_file=args.prompts_file,
        labels_csv=args.labels_csv,
        graph=graph,
        split=args.support_split,
        default_cell_id=args.default_cell_id,
        require_label=True,
    )
    if not support_examples:
        raise RuntimeError("No support examples found")
    print(f"[MSLD] Support examples: {len(support_examples)}")

    model, tokenizer = _build_model_and_tokenizer(args)
    model_device = next(model.parameters()).device
    head_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MSLD] Base model device={model_device} | head device={head_device}")

    label_candidates = [x.strip() for x in args.label_candidates.split(",") if x.strip()]
    if not {"yes", "no", "insufficient"}.issubset({x.lower() for x in label_candidates}):
        raise RuntimeError("--label-candidates must include yes,no,insufficient")

    feat = _precompute_support_features(
        model=model,
        tokenizer=tokenizer,
        model_device=model_device,
        examples=support_examples,
        max_input_tokens=args.max_input_tokens,
        label_candidates=label_candidates,
        label_prefix=args.label_prefix,
    )

    h_np = feat["hidden"]
    y_np = feat["label"]
    gene_id_np = feat["gene_id"]
    base_margin_np = feat["base_margin"]
    base_yes_prob_np = feat["base_yes_prob"]
    n, hidden_dim = h_np.shape
    print(f"[MSLD] Hidden feature shape: {h_np.shape}")

    groups = group_examples_by_perturbation(support_examples)
    group_keys = sorted(groups.keys())
    group_to_id = {k: i for i, k in enumerate(group_keys)}
    sample_group_id = np.asarray(
        [group_to_id[support_examples[i].perturbation_key] for i in range(len(support_examples))],
        dtype=np.int64,
    )
    num_groups = len(group_keys)
    print(f"[MSLD] Support perturbation groups: {num_groups}")

    group_priors: List[np.ndarray] = []
    group_evidence: List[np.ndarray] = []
    group_sample_indices: List[np.ndarray] = []
    group_cell_ids = np.zeros((num_groups,), dtype=np.int64)
    group_drug_ids = np.zeros((num_groups,), dtype=np.int64)
    for gid, gk in enumerate(group_keys):
        idxs = np.asarray(groups[gk], dtype=np.int64)
        group_sample_indices.append(idxs)
        ex0 = support_examples[int(idxs[0])]
        drug_id = int(ex0.drug_id)
        group_cell_ids[gid] = int(ex0.cell_id)
        group_drug_ids[gid] = int(ex0.drug_id)
        prior = compute_drug_prior(graph, drug_id=drug_id)
        evidence = compute_support_evidence(
            graph=graph,
            gene_ids=[int(gene_id_np[i]) for i in idxs.tolist()],
            labels=[int(y_np[i]) for i in idxs.tolist()],
        )
        group_priors.append(prior)
        group_evidence.append(evidence)

    group_priors_t = torch.tensor(np.asarray(group_priors, dtype=np.float32), dtype=torch.float32, device=head_device)
    group_evidence_t = torch.tensor(np.asarray(group_evidence, dtype=np.float32), dtype=torch.float32, device=head_device)

    inferred_max_cell = int(max(1, int(group_cell_ids.max()) + 1))
    inferred_max_drug = int(max(1, int(group_drug_ids.max()) + 1))

    cfg = MSLDHeadConfig(
        hidden_dim=int(hidden_dim),
        num_modules=int(graph.num_modules),
        module_emb_dim=args.module_emb_dim,
        label_hidden_dim=args.label_hidden_dim,
        dropout=args.dropout,
        refine_hidden_dim=args.refine_hidden_dim,
        type_emb_dim=args.type_emb_dim,
        max_cell_types=max(args.max_cell_types, inferred_max_cell),
        max_drug_types=max(args.max_drug_types, inferred_max_drug),
    )
    heads = MSLDHeads(cfg).to(head_device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    h_t = torch.tensor(h_np, dtype=torch.float32, device=head_device)
    y_t = torch.tensor(y_np, dtype=torch.float32, device=head_device)
    base_margin_t = torch.tensor(base_margin_np, dtype=torch.float32, device=head_device)
    base_yes_prob_t = torch.tensor(base_yes_prob_np, dtype=torch.float32, device=head_device)
    sample_group_id_t = torch.tensor(sample_group_id, dtype=torch.long, device=head_device)

    indices = np.arange(n, dtype=np.int64)
    best_loss = float("inf")
    best_state = None

    print("[MSLD] Training heads on support-only adaptation")
    for epoch in range(1, args.epochs + 1):
        heads.train()

        np.random.shuffle(indices)
        total_loss = 0.0
        total_label = 0.0
        total_mech = 0.0
        total_anchor = 0.0
        total_prop = 0.0
        total_ver = 0.0
        total_refine = 0.0
        steps = 0

        for start in range(0, n, args.batch_size):
            batch_idx_np = indices[start:start + args.batch_size]
            batch_idx = torch.tensor(batch_idx_np, dtype=torch.long, device=head_device)
            h_b = h_t[batch_idx]
            y_b = y_t[batch_idx]
            bm_b = base_margin_t[batch_idx]
            by_b = base_yes_prob_t[batch_idx]
            group_b = sample_group_id_t[batch_idx]
            uniq_group = torch.unique(group_b)

            group_to_pos: Dict[int, int] = {}
            z_ref_list: List[torch.Tensor] = []
            z_teacher_list: List[torch.Tensor] = []
            pooled_list: List[torch.Tensor] = []
            for pos, gid_t in enumerate(uniq_group.tolist()):
                gid = int(gid_t)
                group_to_pos[gid] = pos
                idx_t = torch.tensor(group_sample_indices[gid], dtype=torch.long, device=head_device)
                h_g = h_t[idx_t]
                y_g = y_t[idx_t]
                pooled_h = h_g.mean(dim=0, keepdim=True)
                pooled_list.append(pooled_h.squeeze(0))

                u_t = heads.propose_logits(pooled_h).squeeze(0)
                cons_logits = heads.module_consistency_logits(h_g)
                y_expand = y_g.unsqueeze(1).expand_as(cons_logits)
                cons_t = -F.binary_cross_entropy_with_logits(cons_logits, y_expand, reduction="none").mean(dim=0)
                prior_t = group_priors_t[gid]
                evidence_t = group_evidence_t[gid]

                refine_out = heads.refine_posterior(
                    proposer_logits=u_t,
                    graph_prior=prior_t,
                    support_evidence=evidence_t,
                    support_consistency=cons_t,
                    cell_id=int(group_cell_ids[gid]),
                    drug_id=int(group_drug_ids[gid]),
                    alpha=args.alpha_u,
                    beta=args.beta_prior,
                    gamma=args.gamma_evidence,
                    rho=args.rho_consistency,
                    rounds=args.refine_rounds,
                    damping=args.refine_damping,
                    lambda_delta=args.lambda_refine_delta,
                )
                z_ref_list.append(refine_out["z_refined"])

                with torch.no_grad():
                    z_teacher_np = verify_mechanism_posterior(
                        proposer_logits=u_t.detach().cpu().numpy(),
                        graph_prior=prior_t.detach().cpu().numpy(),
                        support_evidence=evidence_t.detach().cpu().numpy(),
                        alpha=args.alpha_u,
                        beta=args.beta_prior,
                        gamma=args.gamma_evidence,
                        consistency=cons_t.detach().cpu().numpy(),
                        rho=args.rho_consistency,
                        top_k=args.teacher_top_k,
                    )
                    z_teacher = torch.tensor(z_teacher_np, dtype=torch.float32, device=head_device)
                z_teacher_list.append(z_teacher)

            z_ref_group_t = torch.stack(z_ref_list, dim=0)
            z_teacher_group_t = torch.stack(z_teacher_list, dim=0)
            pooled_group_t = torch.stack(pooled_list, dim=0)

            batch_group_pos = torch.tensor(
                [group_to_pos[int(g)] for g in group_b.tolist()],
                dtype=torch.long,
                device=head_device,
            )
            z_star_b = z_ref_group_t[batch_group_pos]

            mech_logits = heads.mechanism_logits(h_b)
            pred = heads.predict_margin(h_b, mech_logits, bm_b)
            margin_b = pred["margin"]

            l_label = F.binary_cross_entropy_with_logits(margin_b, y_b)
            # Distill refined perturbation-level mechanism target to example-level mechanism head.
            l_mech = soft_target_cross_entropy(mech_logits, z_star_b.detach())

            if args.lambda_verifier > 0:
                cons_logits_b = pred["consistency_logits"]
                y_expand_b = y_b.unsqueeze(1).expand_as(cons_logits_b)
                bce_mod = F.binary_cross_entropy_with_logits(cons_logits_b, y_expand_b, reduction="none")
                # Only modules that are active in z* should bear larger label-explanation responsibility.
                l_ver = (bce_mod * z_star_b.detach()).sum(dim=-1).mean()
            else:
                l_ver = torch.zeros((), device=head_device)

            if args.lambda_anchor > 0:
                pred_yes = torch.sigmoid(margin_b)
                pred_p = torch.stack([pred_yes, 1.0 - pred_yes], dim=-1).clamp_min(1e-8)
                base_p = torch.stack([by_b, 1.0 - by_b], dim=-1).clamp_min(1e-8)
                l_anchor = F.kl_div(pred_p.log(), base_p, reduction="batchmean")
            else:
                l_anchor = torch.zeros((), device=head_device)

            if args.lambda_proposer > 0:
                prop_logits = heads.propose_logits(pooled_group_t)
                l_prop = soft_target_cross_entropy(prop_logits, z_ref_group_t.detach())
            else:
                l_prop = torch.zeros((), device=head_device)

            if args.lambda_refiner > 0:
                # Refiner learns to approximate graph sparse teacher while staying differentiable.
                l_refine = F.kl_div(
                    z_ref_group_t.clamp_min(1e-8).log(),
                    z_teacher_group_t.detach().clamp_min(1e-8),
                    reduction="batchmean",
                )
            else:
                l_refine = torch.zeros((), device=head_device)

            loss = (
                l_label
                + args.lambda_mech * l_mech
                + args.lambda_verifier * l_ver
                + args.lambda_anchor * l_anchor
                + args.lambda_proposer * l_prop
                + args.lambda_refiner * l_refine
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(heads.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_label += float(l_label.detach().cpu())
            total_mech += float(l_mech.detach().cpu())
            total_ver += float(l_ver.detach().cpu())
            total_anchor += float(l_anchor.detach().cpu())
            total_prop += float(l_prop.detach().cpu())
            total_refine += float(l_refine.detach().cpu())
            steps += 1

        mean_loss = total_loss / max(1, steps)
        print(
            f"[MSLD][Epoch {epoch}/{args.epochs}] "
            f"loss={mean_loss:.6f} label={total_label/max(1,steps):.6f} "
            f"mech={total_mech/max(1,steps):.6f} ver={total_ver/max(1,steps):.6f} "
            f"anchor={total_anchor/max(1,steps):.6f} prop={total_prop/max(1,steps):.6f} "
            f"refine={total_refine/max(1,steps):.6f}"
        )
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {k: v.detach().cpu().clone() for k, v in heads.state_dict().items()}

    if best_state is not None:
        heads.load_state_dict(best_state)
    heads.eval()

    ckpt = {
        "version": "msld_v2_consistency_refine",
        "head_config": {
            "hidden_dim": cfg.hidden_dim,
            "num_modules": cfg.num_modules,
            "module_emb_dim": cfg.module_emb_dim,
            "label_hidden_dim": cfg.label_hidden_dim,
            "dropout": cfg.dropout,
            "refine_hidden_dim": cfg.refine_hidden_dim,
            "type_emb_dim": cfg.type_emb_dim,
            "max_cell_types": cfg.max_cell_types,
            "max_drug_types": cfg.max_drug_types,
        },
        "heads_state_dict": heads.state_dict(),
        "module_names": graph.idx_to_module,
        "module_types": graph.module_types,
        "label_candidates": label_candidates,
        "label_prefix": args.label_prefix,
        "decision": {
            "tau": args.tau,
            "eta1": args.eta1,
            "eta2": args.eta2,
            "eta3": args.eta3,
        },
        "train_summary": {
            "num_support_examples": len(support_examples),
            "num_perturbation_groups": num_groups,
            "num_modules": graph.num_modules,
            "best_loss": best_loss,
            "rho_consistency": args.rho_consistency,
            "refine_rounds": args.refine_rounds,
            "refine_damping": args.refine_damping,
        },
        "args": vars(args),
    }
    torch.save(ckpt, args.out_ckpt)
    print(f"[MSLD] Saved checkpoint: {args.out_ckpt}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train support-supervised latent mechanism distillation heads (MSLD).")
    p.add_argument("--model-name", required=True)
    p.add_argument("--prompts-file", required=True)
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--kg-dir", required=True)
    p.add_argument("--out-ckpt", required=True)

    p.add_argument("--kg-nodes-file", default="nodes.json")
    p.add_argument("--kg-edges-file", default="edges.json")
    p.add_argument("--kg-graph-file", default="graph.json")
    p.add_argument("--support-split", default="train")
    p.add_argument("--default-cell-id", type=int, default=0)

    p.add_argument("--max-modules", type=int, default=2048, help="Fixed global module vocabulary size cap.")
    p.add_argument("--graph-alpha", type=float, default=0.1)
    p.add_argument("--graph-diffusion-hops", type=int, default=1)
    p.add_argument("--graph-diffusion-decay", type=float, default=1.0)
    p.add_argument("--teacher-top-k", type=int, default=256)
    p.add_argument("--alpha-u", type=float, default=1.0)
    p.add_argument("--beta-prior", type=float, default=1.0)
    p.add_argument("--gamma-evidence", type=float, default=1.0)
    p.add_argument("--rho-consistency", type=float, default=1.0)

    p.add_argument("--module-emb-dim", type=int, default=128)
    p.add_argument("--label-hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--refine-hidden-dim", type=int, default=64)
    p.add_argument("--type-emb-dim", type=int, default=16)
    p.add_argument("--max-cell-types", type=int, default=256)
    p.add_argument("--max-drug-types", type=int, default=8192)
    p.add_argument("--refine-rounds", type=int, default=2)
    p.add_argument("--refine-damping", type=float, default=0.5)
    p.add_argument("--lambda-refine-delta", type=float, default=1.0)

    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--lambda-mech", type=float, default=1.0)
    p.add_argument("--lambda-verifier", type=float, default=1.0)
    p.add_argument("--lambda-anchor", type=float, default=0.0)
    p.add_argument("--lambda-proposer", type=float, default=0.0)
    p.add_argument("--lambda-refiner", type=float, default=0.2)

    p.add_argument("--tau", type=float, default=0.0)
    p.add_argument("--eta1", type=float, default=1.0)
    p.add_argument("--eta2", type=float, default=1.0)
    p.add_argument("--eta3", type=float, default=1.0)

    p.add_argument("--max-input-tokens", type=int, default=2048)
    p.add_argument("--label-candidates", default="yes,no,insufficient")
    p.add_argument("--label-prefix", default="\nFinal Deterministic Prediction:\n")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
