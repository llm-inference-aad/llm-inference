from __future__ import annotations

import copy
from typing import Dict, List


def print_ancestry(data: Dict[str, Dict[str, List[str]]]) -> None:
    """Pretty-print ancestry metadata for debugging."""
    for gene, payload in data.items():
        print(f"gene: {gene}")
        print(f"\t{payload.get('GENES')}")
        print(f"\t{payload.get('MUTATE_TYPE')}")


def update_ancestry(
    gene_id_child: str,
    gene_id_parent: str,
    ancestry: Dict[str, Dict[str, List[str]]],
    mutation_type: str | None = None,
    gene_id_parent2: str | None = None,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Update ancestry information for a newly created child gene.

    Parameters
    ----------
    gene_id_child : str
        Identifier of the new child gene.
    gene_id_parent : str
        Identifier of the first parent gene.
    ancestry : dict
        Global ancestry tracking dictionary.
    mutation_type : str, optional
        Type of mutation applied (for single-parent updates).
    gene_id_parent2 : str, optional
        Identifier of the second parent for crossover events.
    """
    ancestry[gene_id_child] = copy.deepcopy(ancestry[gene_id_parent])

    if gene_id_parent2 is None:
        ancestry[gene_id_child]["GENES"] = copy.deepcopy(
            ancestry[gene_id_parent]["GENES"]
        ) + [gene_id_child]
        ancestry[gene_id_child]["MUTATE_TYPE"] = copy.deepcopy(
            ancestry[gene_id_parent]["MUTATE_TYPE"]
        ) + [mutation_type]
    else:
        cross_id = f"P:{gene_id_parent2}-C:{gene_id_child}"
        ancestry[gene_id_child]["GENES"] = copy.deepcopy(
            ancestry[gene_id_parent]["GENES"]
        ) + [cross_id]
        ancestry[gene_id_child]["MUTATE_TYPE"] = copy.deepcopy(
            ancestry[gene_id_parent]["MUTATE_TYPE"]
        ) + ["CrossOver"]

    return ancestry

