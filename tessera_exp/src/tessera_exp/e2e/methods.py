from __future__ import annotations

METHOD_KEYS = [
    "dense_concat",
    "naive_rag",
    "carp",
    "tablerag",
    "quasar",
    "unihgkr_dense",
    "tessera",
    "tessera_submod",
    "ablation_no_redundancy_e2e",
    "ablation_no_pathmaxsim_e2e",
]

METHOD_LABELS = {
    "dense_concat": "Dense-Concat",
    "naive_rag": "NaiveRAG",
    "carp": "CARP-Adapter (IJCAI 22)",
    "tablerag": "TableRAG-Adapter (Google)",
    "quasar": "QUASAR-Adapter",
    "unihgkr_dense": "UniHGKR-Base (Dense Baseline)",
    "tessera": "TESSERA (Ours)",
    "ablation_no_redundancy_e2e": "(d) TESSERA w/o Redundancy",
    "ablation_no_pathmaxsim_e2e": "(e) TESSERA w/o PathMaxSim",
    "tessera_submod": "TESSERA-Submod (Ours)",
}

METHOD_MODALITY_COVERAGE = {
    "dense_concat": "T+Tbl+G",
    "naive_rag": "T+Tbl+G+Vec",
    "carp": "T+Tbl+G+Vec",
    "tablerag": "T+Tbl+G+Vec",
    "quasar": "T+Tbl+G+Vec",
    "unihgkr_dense": "T+Tbl+G",
    "tessera": "T+Tbl+G+Vec",
    "ablation_no_redundancy_e2e": "T+Tbl+G+Vec",
    "ablation_no_pathmaxsim_e2e": "T+Tbl+G+Vec",
    "tessera_submod": "T+Tbl+G+Vec",
}

METHOD_PRESETS = {
    "targeted": ["dense_concat", "tessera"],
    "main_only": ["dense_concat", "tessera"],
    "pathmaxsim": ["dense_concat", "tessera", "ablation_no_pathmaxsim_e2e"],
    "redundancy": ["dense_concat", "tessera", "ablation_no_redundancy_e2e"],
    "baselines": ["dense_concat", "naive_rag", "carp", "tablerag", "quasar", "unihgkr_dense"],
    "submod": ["dense_concat", "tessera", "tessera_submod"],
    "full": METHOD_KEYS,
}


def resolve_selected_methods(method_preset: str, methods_raw: str | None) -> list[str]:
    if methods_raw is None or str(methods_raw).strip() == "":
        selected = list(METHOD_PRESETS[str(method_preset)])
    else:
        selected = [m.strip() for m in str(methods_raw).split(",") if m.strip()]

    unknown = [m for m in selected if m not in METHOD_KEYS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Supported: {METHOD_KEYS}")
    if not selected:
        raise ValueError("No methods selected")
    return selected


def resolve_reuse_methods(reuse_methods_raw: str) -> list[str]:
    reuse_methods = [m.strip() for m in str(reuse_methods_raw).split(",") if m.strip()]
    unknown_reuse = [m for m in reuse_methods if m not in METHOD_KEYS]
    if unknown_reuse:
        raise ValueError(f"Unknown reuse methods: {unknown_reuse}. Supported: {METHOD_KEYS}")
    return reuse_methods
