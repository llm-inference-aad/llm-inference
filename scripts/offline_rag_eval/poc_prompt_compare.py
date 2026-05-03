#!/usr/bin/env python3
"""POC: compare baseline vs RAG-augmented prompts from a text file.

1. Reads up to N non-empty lines from a UTF-8 file (one mutation prompt per line).
2. Builds RAG context from existing ``rag_data/`` (no snapshot step).
3. Writes ``rendered.jsonl`` then POSTs each variant to the LLM server (same
   ``/generate`` path as the main stack). Records fence-style score plus cheap
   mutation-style static checks (AST + compile on the first ```python``` block;
   optional ``validate_module_source`` via ``--module-validate``) and HTTP retry
   attempt counts in ``results.jsonl``.
4. Writes ``poc_metrics.png``, ``poc_static_qc.png``, and ``report.html`` under ``--output-dir``.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFLINE_DIR = Path(__file__).resolve().parent
for p in (OFFLINE_DIR, REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_offline_eval  # noqa: E402


def _format_safe(template: str, snippet: str) -> str:
    placeholder = "{}"
    idx = template.find(placeholder)
    if idx == -1:
        return template
    head = template[:idx].replace("{", "{{").replace("}", "}}")
    tail = template[idx + len(placeholder) :].replace("{", "{{").replace("}", "}}")
    safe_template = head + placeholder + tail
    return safe_template.format(snippet)


def _build_enhancer(
    rag_dir: Path,
    *,
    code_top_k: int,
    text_top_k: int,
    min_similarity: float,
    min_accuracy: float,
    max_parameters: float | None,
    code_embed_model: str,
    text_embed_model: str,
):
    from rag.backends.faiss_backend import FaissRetrievalBackend  # type: ignore
    from rag.prompt_enhancer import PromptEnhancer, PromptEnhancerConfig  # type: ignore

    backend = FaissRetrievalBackend(
        rag_data_dir=rag_dir,
        code_embed_model=code_embed_model,
        text_embed_model=text_embed_model,
    )
    return PromptEnhancer(
        retrieval_service=backend,
        config=PromptEnhancerConfig(
            top_k=code_top_k,
            text_top_k=text_top_k,
            min_accuracy=min_accuracy,
            max_parameters=max_parameters,
        ),
        memory_store=None,
    )


def _render_pair(
    enhancer,
    *,
    template_text: str,
    snippet: str,
    mutation_label: str,
    use_code_context: bool,
    use_text_context: bool,
) -> tuple[str, str, dict]:
    baseline_prompt = _format_safe(template_text, snippet)
    import cfg.constants as constants  # type: ignore
    import rag.prompt_enhancer as enhancer_module  # type: ignore

    saved_use_code = constants.RAG_USE_CODE_CONTEXT
    saved_use_text = constants.RAG_USE_TEXT_CONTEXT
    try:
        constants.RAG_USE_CODE_CONTEXT = use_code_context
        constants.RAG_USE_TEXT_CONTEXT = use_text_context
        enhancer_module.RAG_USE_CODE_CONTEXT = use_code_context
        enhancer_module.RAG_USE_TEXT_CONTEXT = use_text_context
        augmented_template, mutations = enhancer.enhance_template(
            template=template_text,
            mutation_type=mutation_label,
            query_code=snippet,
            gene_id=None,
        )
    finally:
        constants.RAG_USE_CODE_CONTEXT = saved_use_code
        constants.RAG_USE_TEXT_CONTEXT = saved_use_text
        enhancer_module.RAG_USE_CODE_CONTEXT = saved_use_code
        enhancer_module.RAG_USE_TEXT_CONTEXT = saved_use_text

    rag_prompt = _format_safe(augmented_template, snippet)
    text_contexts = enhancer.build_text_context(
        query_code=snippet, mutation_type=mutation_label
    )
    rag_meta = {
        "retrieved_code_n": len(mutations),
        "retrieved_text_n": len(text_contexts),
        "retrieved_doc_ids_code": [m.gene_id for m in mutations],
        "retrieved_doc_ids_text": [c.document_id for c in text_contexts],
        "context_added_words": max(0, len(rag_prompt.split()) - len(baseline_prompt.split())),
    }
    return baseline_prompt, rag_prompt, rag_meta


def _load_prompts(path: Path, *, max_prompts: int) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
        if len(lines) >= max_prompts:
            break
    if not lines:
        raise SystemExit(f"No prompts found in {path}")
    return lines


def _pin_rag_env(rag_dir: Path, *, use_code: bool, use_text: bool, min_sim: float) -> None:
    os.environ["RAG_DATA_DIR"] = str(rag_dir.resolve())
    os.environ.setdefault("RAG_ENABLED", "true")
    os.environ["RAG_USE_CODE_CONTEXT"] = "true" if use_code else "false"
    os.environ["RAG_USE_TEXT_CONTEXT"] = "true" if use_text else "false"
    os.environ.setdefault("RAG_MIN_SIMILARITY", str(min_sim))
    os.environ.setdefault("RAG_MEMORY_STORE_ENABLED", "false")


def _ev(row: dict, key: str):
    return (row.get("eval") or {}).get(key)


def _score_for_chart(row: dict) -> float | None:
    resp = row.get("response") or {}
    s = resp.get("evaluationScore")
    if s is not None:
        try:
            return float(s)
        except (TypeError, ValueError):
            pass
    ev = row.get("eval") or {}
    if ev.get("eval_score") is not None:
        return float(ev["eval_score"])
    return None


def _fmt_bool(v) -> str:
    if v is True:
        return "Y"
    if v is False:
        return "N"
    return "—"


def _write_report_html(
    path: Path,
    *,
    prompts_path: Path,
    rows_by_pair: dict[str, dict[str, dict]],
    baseline_texts: list[str],
    rag_texts: list[str],
) -> None:
    rows_html = []
    syn_lines: list[str] = []
    for i, pid in enumerate(sorted(rows_by_pair, key=lambda k: int(k.replace("poc", "")))):
        pair = rows_by_pair[pid]
        b = pair.get("baseline", {})
        r = pair.get("rag", {})
        bs = _score_for_chart(b)
        rs = _score_for_chart(r)
        bl = (b.get("request") or {}).get("wallclock_latency_sec")
        rl = (r.get("request") or {}).get("wallclock_latency_sec")
        hb = (b.get("request") or {}).get("http_attempts_used")
        hr = (r.get("request") or {}).get("http_attempts_used")
        seb = _ev(b, "syntax_error")
        ser = _ev(r, "syntax_error")
        syn_lines.append(
            "<li>#"
            f"{i}: baseline: {html.escape(str(seb) if seb else '—')} — RAG: "
            f"{html.escape(str(ser) if ser else '—')}</li>"
        )
        rows_html.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{bs if bs is not None else '—'}</td>"
            f"<td>{rs if rs is not None else '—'}</td>"
            f"<td>{_fmt_bool(_ev(b, 'ast_parse_ok'))}</td>"
            f"<td>{_fmt_bool(_ev(r, 'ast_parse_ok'))}</td>"
            f"<td>{_fmt_bool(_ev(b, 'compile_exec_ok'))}</td>"
            f"<td>{_fmt_bool(_ev(r, 'compile_exec_ok'))}</td>"
            f"<td>{_fmt_bool(_ev(b, 'module_validate_ok'))}</td>"
            f"<td>{_fmt_bool(_ev(r, 'module_validate_ok'))}</td>"
            f"<td>{hb if hb is not None else '—'}</td>"
            f"<td>{hr if hr is not None else '—'}</td>"
            f"<td>{bl if bl is not None else '—'}</td>"
            f"<td>{rl if rl is not None else '—'}</td>"
            "</tr>"
        )
    details = []
    for i, (bt, rt) in enumerate(zip(baseline_texts, rag_texts, strict=True)):
        details.append(
            f"<details><summary>Prompt {i}</summary>"
            f"<h4>Baseline</h4><pre>{html.escape(bt[:8000])}</pre>"
            f"<h4>RAG</h4><pre>{html.escape(rt[:8000])}</pre></details>"
        )
    body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>RAG prompt POC</title></head>
<body>
<h1>RAG prompt POC</h1>
<p>Prompts file: {html.escape(str(prompts_path))}</p>
<p>Charts: <a href="poc_metrics.png">poc_metrics.png</a> (fence score), <a href="poc_static_qc.png">poc_static_qc.png</a> (AST / compile)</p>
<img src="poc_metrics.png" alt="fence score" style="max-width:900px"/>
<img src="poc_static_qc.png" alt="static qc" style="max-width:900px"/>
<h2>Per prompt</h2>
<p>Fence score: server evaluationScore if present, else local eval. Static: first ```python``` block — AST parse and compile(exec); module column is — unless you ran with <code>--module-validate</code>. HTTP attempts: how many POST tries succeeded on (mutation-relevant when the client retries).</p>
<table border="1" cellpadding="4">
<tr><th>#</th><th>score b</th><th>score r</th><th>AST b</th><th>AST r</th><th>compile b</th><th>compile r</th><th>mod b</th><th>mod r</th><th>http # b</th><th>http # r</th><th>lat s b</th><th>lat s r</th></tr>
{"".join(rows_html)}
</table>
<h2>Syntax errors (first block)</h2>
<ul>
{"".join(syn_lines)}
</ul>
<h2>Prompt text</h2>
{"".join(details)}
</body></html>
"""
    path.write_text(body, encoding="utf-8")


def _b01(row: dict, key: str) -> float:
    return 1.0 if _ev(row, key) is True else 0.0


def _plot_static_qc(
    path: Path,
    rows_by_pair: dict[str, dict[str, dict]],
    pair_ids: list[str],
    *,
    show_module_validate: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = list(range(len(pair_ids)))
    yb_ast: list[float] = []
    yr_ast: list[float] = []
    yb_comp: list[float] = []
    yr_comp: list[float] = []
    for pid in pair_ids:
        pair = rows_by_pair[pid]
        b = pair.get("baseline", {})
        r = pair.get("rag", {})
        yb_ast.append(_b01(b, "ast_parse_ok"))
        yr_ast.append(_b01(r, "ast_parse_ok"))
        yb_comp.append(_b01(b, "compile_exec_ok"))
        yr_comp.append(_b01(r, "compile_exec_ok"))

    w = 0.15
    fig, ax = plt.subplots(figsize=(max(6, len(x)), 4))
    for i in x:
        ax.bar(i - 1.5 * w, yb_ast[i], width=w, color="C0", alpha=0.85)
        ax.bar(i - 0.5 * w, yr_ast[i], width=w, color="C0", alpha=0.45)
        ax.bar(i + 0.5 * w, yb_comp[i], width=w, color="C2", alpha=0.85)
        ax.bar(i + 1.5 * w, yr_comp[i], width=w, color="C2", alpha=0.45)
    if show_module_validate:
        yb_mod = [_b01(rows_by_pair[pid].get("baseline", {}), "module_validate_ok") for pid in pair_ids]
        yr_mod = [_b01(rows_by_pair[pid].get("rag", {}), "module_validate_ok") for pid in pair_ids]
        for i in x:
            ax.bar(i + 2.5 * w, yb_mod[i], width=w, color="C3", alpha=0.85)
            ax.bar(i + 3.5 * w, yr_mod[i], width=w, color="C3", alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}" for i in x])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("pass (1) / fail (0)")
    ax.set_title("POC: static QC on first ```python``` block (dark=baseline, light=rag)")
    from matplotlib.patches import Patch

    leg = [
        Patch(facecolor="C0", alpha=0.85, label="AST baseline"),
        Patch(facecolor="C0", alpha=0.45, label="AST rag"),
        Patch(facecolor="C2", alpha=0.85, label="compile baseline"),
        Patch(facecolor="C2", alpha=0.45, label="compile rag"),
    ]
    if show_module_validate:
        leg.extend(
            [
                Patch(facecolor="C3", alpha=0.85, label="module baseline"),
                Patch(facecolor="C3", alpha=0.45, label="module rag"),
            ]
        )
    ax.legend(handles=leg, ncol=3, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_metrics(
    path: Path,
    rows_by_pair: dict[str, dict[str, dict]],
    pair_ids: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = list(range(len(pair_ids)))
    yb: list[float] = []
    yr: list[float] = []
    for pid in pair_ids:
        pair = rows_by_pair[pid]
        bs = _score_for_chart(pair.get("baseline", {}))
        rs = _score_for_chart(pair.get("rag", {}))
        yb.append(bs if bs is not None else 0.0)
        yr.append(rs if rs is not None else 0.0)
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(x)), 4))
    ax.bar([i - w / 2 for i in x], yb, width=w, label="baseline")
    ax.bar([i + w / 2 for i in x], yr, width=w, label="rag")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}" for i in x])
    ax.set_ylabel("score (server or local)")
    ax.legend()
    ax.set_title("POC: per-prompt scores")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts-file",
        type=Path,
        required=True,
        help="UTF-8 text: one mutation prompt per line (# and blank lines skipped).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "scripts" / "offline_rag_eval" / "poc_out",
        help="Directory for rendered.jsonl, results.jsonl, report.html, charts.",
    )
    parser.add_argument(
        "--rag-dir",
        type=Path,
        default=None,
        help="RAG vector store (default: RAG_DATA_DIR or <repo>/rag_data).",
    )
    parser.add_argument("--max-prompts", type=int, default=10)
    parser.add_argument("--server-url", type=str, default=None)
    parser.add_argument("--server-port", type=str, default=os.environ.get("SERVER_PORT", "8000"))
    parser.add_argument("--run-id", type=str, default=os.environ.get("RUN_ID"))
    parser.add_argument("--code-top-k", type=int, default=5)
    parser.add_argument("--text-top-k", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=float(os.environ.get("RAG_MIN_SIMILARITY", 0.3)))
    parser.add_argument("--min-accuracy", type=float, default=0.9)
    parser.add_argument("--no-code-context", action="store_true")
    parser.add_argument("--no-text-context", action="store_true")
    parser.add_argument("--code-embed-model", type=str, default="microsoft/codebert-base")
    parser.add_argument("--text-embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        "--module-validate",
        action="store_true",
        help="Run validate_module_source on each response (heavy; needs torch / full repo imports).",
    )
    args = parser.parse_args()

    rag_dir = (args.rag_dir or Path(os.environ.get("RAG_DATA_DIR", str(REPO_ROOT / "rag_data")))).resolve()
    if not rag_dir.is_dir():
        raise SystemExit(f"RAG directory not found: {rag_dir}")

    prompts = _load_prompts(args.prompts_file, max_prompts=args.max_prompts)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered_path = out_dir / "rendered.jsonl"
    results_path = out_dir / "results.jsonl"

    use_code = not args.no_code_context
    use_text = not args.no_text_context
    _pin_rag_env(rag_dir, use_code=use_code, use_text=use_text, min_sim=args.min_similarity)

    enhancer = _build_enhancer(
        rag_dir,
        code_top_k=args.code_top_k,
        text_top_k=args.text_top_k,
        min_similarity=args.min_similarity,
        min_accuracy=args.min_accuracy,
        max_parameters=None,
        code_embed_model=args.code_embed_model,
        text_embed_model=args.text_embed_model,
    )

    rel = args.prompts_file.resolve().as_posix()
    try:
        rel = args.prompts_file.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        pass

    baseline_texts: list[str] = []
    rag_texts: list[str] = []

    with rendered_path.open("w", encoding="utf-8") as out:
        for i, line in enumerate(prompts):
            snippet = line[:8000]
            pair_id = f"poc{i}"
            bl, rg, meta = _render_pair(
                enhancer,
                template_text=line,
                snippet=snippet,
                mutation_label="poc",
                use_code_context=use_code,
                use_text_context=use_text,
            )
            baseline_texts.append(bl)
            rag_texts.append(rg)
            shared = {
                "pair_id": pair_id,
                "split": "test",
                "mutation_label": "poc",
                "template_path": rel,
                "augment_idx": i,
                "code_snippet": snippet,
            }
            out.write(
                json.dumps(
                    {
                        **shared,
                        "variant": "baseline",
                        "prompt": bl,
                        "prompt_tokens_est": len(bl.split()),
                        "rag": None,
                    }
                )
                + "\n"
            )
            out.write(
                json.dumps(
                    {
                        **shared,
                        "variant": "rag",
                        "prompt": rg,
                        "prompt_tokens_est": len(rg.split()),
                        "rag": meta,
                    }
                )
                + "\n"
            )

    print(f"Wrote {rendered_path}")
    server_url = run_offline_eval.discover_server_url(
        cli_url=args.server_url, run_id=args.run_id, port=args.server_port
    )
    print(f"Server: {server_url}")
    server = run_offline_eval.ServerConfig(
        url=server_url,
        max_new_tokens=1024,
        temperature=0.1,
        top_p=0.15,
        timeout_seconds=300.0,
        max_retries=3,
    )
    summary = run_offline_eval.run(
        rendered_path=rendered_path,
        output_path=results_path,
        server=server,
        job_id="poc_prompt_compare",
        splits=None,
        limit=len(prompts),
        validate_module=args.module_validate,
    )
    print(f"Wrote {results_path}")
    print(json.dumps(summary, indent=2))

    rows_by_pair: dict[str, dict[str, dict]] = {}
    with results_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = row["pair_id"]
            var = row["variant"]
            rows_by_pair.setdefault(pid, {})[var] = row
    pair_ids = [f"poc{i}" for i in range(len(prompts))]

    _plot_metrics(out_dir / "poc_metrics.png", rows_by_pair, pair_ids)
    _plot_static_qc(
        out_dir / "poc_static_qc.png",
        rows_by_pair,
        pair_ids,
        show_module_validate=args.module_validate,
    )
    _write_report_html(
        out_dir / "report.html",
        prompts_path=args.prompts_file,
        rows_by_pair=rows_by_pair,
        baseline_texts=baseline_texts,
        rag_texts=rag_texts,
    )
    print(f"Wrote {out_dir / 'poc_metrics.png'}, {out_dir / 'poc_static_qc.png'}, and {out_dir / 'report.html'}")


if __name__ == "__main__":
    main()
