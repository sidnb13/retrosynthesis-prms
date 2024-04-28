import functools
import os
from collections.abc import Mapping
from hashlib import md5

import numpy as np
import torch
import torch.distributed as dist
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def process(reagents: dict, smarts):
    """Process a SMARTS reaction string
    into source and target tokens"""
    rxn = AllChem.ReactionFromSmarts(smarts)
    prods = rxn.GetProducts()
    if len(prods) > 1:
        return None

    rxn.Initialize()
    try:
        reactants = list(zip(rxn.GetReactants(), rxn.GetReactingAtoms()))
    except ValueError:
        # Likely that initialization failed
        # print('Failed to initialize')
        return None

    prod_smis = []
    for mol in prods:
        # Clear atom mappings
        [x.ClearProp("molAtomMapNumber") for x in mol.GetAtoms()]
        smi = Chem.MolToSmiles(mol)
        prod_smis.append(smi)

    react_smis = []
    reagent_syms = []
    for mol, atoms in reactants:
        # Clear atom mappings
        [x.ClearProp("molAtomMapNumber") for x in mol.GetAtoms()]

        # Remove molecules with no reacting atoms (reagents)
        # But represent as a symbol
        if not atoms:
            smi = Chem.MolToSmiles(mol)
            if smi not in reagents:
                reagents[smi] = len(reagents)
            reagent_syms.append("[A{}]".format(reagents[smi]))

        else:
            smi = Chem.MolToSmiles(mol)
            react_smis.append(smi)

    source = react_smis
    if reagent_syms:
        source.extend(reagent_syms)
    target = prod_smis

    return source, target


def clean(line):
    return line.strip().split()[0]


def torch_default_data_collator(features):
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = (
            first["label"].item()
            if isinstance(first["label"], torch.Tensor)
            else first["label"]
        )
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = (
                torch.long if isinstance(first["label_ids"][0], int) else torch.float
            )
            batch["labels"] = torch.tensor(
                [f["label_ids"] for f in features], dtype=dtype
            )

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


class ProcessSupervisedDataset(Dataset):
    """Finetuning model to generate next step in retrosynthesis."""

    reagent_template = "<reagent>{}<reagent>"
    product_template = "<product>{}<product>"

    def __init__(
        self,
        model_name_or_path: str,
        data_file: str,
        limit: int = -1,
        max_length: int = 128,
    ) -> None:
        super().__init__()

        self.data = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_length = max_length

        data_cache_path = f"{data_file}_{limit}.pt"
        # process
        if not os.path.exists(data_cache_path) and (
            not dist.is_initialized() or dist.get_rank() == 0
        ):
            seen = set()
            reagents = {}

            with open(data_file, "r") as f:
                lines = f.readlines()

                if limit > 0:
                    print(f"Limiting to {limit} reactions")
                    lines = lines[:limit]

                it = tqdm(map(functools.partial(process, reagents), map(clean, lines)))
                for toks in it:
                    if toks is None:
                        continue

                    # Hash the processed reaction to check for duplicates
                    h = md5(
                        "_".join("".join(ts) for ts in toks).encode("utf8")
                    ).hexdigest()
                    if h in seen:
                        continue
                    else:
                        seen.add(h)

                    # find all possible terminating reagents
                    reagents, prod = toks
                    reagents = reagents[::-1]

                    for i in range(len(reagents) + 1):
                        self.data.append((reagents[:i], prod))

                    it.set_postfix(reactions=len(self.data), reagents=len(reagents))
            # save data
            torch.save(self.data, data_cache_path)
        else:
            if dist.is_initialized():
                dist.barrier()
            self.data = torch.load(data_cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        reagents, product = self.data[idx]

        # add eos token after every reagent
        prompt = self.tokenizer.bos_token + "\n".join(
            [
                self.reagent_template.format(r) + self.tokenizer.eos_token
                for r in reagents
            ]
        )
        label = self.product_template.format(product[0])

        prompt_tokens = self.tokenizer.encode(
            prompt,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        label_tokens = self.tokenizer.encode(
            label,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        combined = prompt_tokens + label_tokens

        # left padding
        padded_input = [self.tokenizer.pad_token_id] * (
            self.max_length - len(prompt_tokens)
        ) + prompt_tokens
        padded_labels = [-100] * (self.max_length - len(combined)) + combined
        padded_labels[-len(combined) : -len(combined) + len(prompt_tokens)] = [
            -100
        ] * len(prompt_tokens)
        attn_mask = (
            (np.array(padded_input) != self.tokenizer.pad_token_id).astype(int).tolist()
        )

        return {
            "input_ids": padded_input,
            "attention_mask": attn_mask,
            "labels": padded_labels,
        }


class ProcessRewardDataset(Dataset):
    """Ground truth rewards for each reagent to train a PRM."""

    pass
