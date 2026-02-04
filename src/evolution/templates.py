from __future__ import annotations

import glob
import os
import random
from typing import List, Tuple

import numpy as np

from src.llm_utils import split_file


def generate_template(
    prob_eot: float,
    generation_count: int,
    top_n_genes: List[Tuple[str, float]],
    sota_root: str,
    seed_network: str,
    root_dir: str,
) -> Tuple[str, str]:
    """
    Generate the mutation template (and metadata describing the mutation type).
    """
    if (prob_eot > np.random.uniform()) and (generation_count > 0):
        print("\t‣ EoT")
        top_gene = np.random.choice([x[0] for x in top_n_genes])
        parts_x = split_file(f"{sota_root}/models/network_{top_gene}.py")
        parts_y = split_file(seed_network)
        parts = [
            (x.strip(), y.strip(), idx)
            for idx, (x, y) in enumerate(zip(parts_x[1:], parts_y[1:]))
        ]
        random.shuffle(parts)
        for x, y, augment_idx in parts:
            if x.strip() != y.strip():
                break

        eot_template_path = os.path.join(root_dir, "templates/EoT/EoT.txt")
        with open(eot_template_path, "r") as file:
            eot_template_txt = file.read()

        template_txt = eot_template_txt.format(x, y, "{}")
        mute_type = "EoT"
    else:
        print("\t‣ FixedPrompts")
        prompt_templates = glob.glob(f"{root_dir}/templates/FixedPrompts/*/*.txt")
        template_path = np.random.choice(prompt_templates)
        mute_type = os.path.basename(template_path).split(".")[0]
        with open(template_path, "r") as file:
            template_txt = file.read()
        with open(f"{root_dir}/templates/ConstantRules.txt", "r") as file:
            rules_txt = file.read()
        template_txt = f"{template_txt}\n{rules_txt}"

    return template_txt, mute_type

