cell_lines = [
    ("C32 cells", "C32 is a human amelanotic melanoma cell line derived from skin... (完整描述保留)... It harbors hallmark alterations including BRAF V600E mutation, PTEN deletion..."),
    ("PANC-1 cells", "PANC-1 is a human pancreatic ductal adenocarcinoma cell line... (完整描述保留)... It is KRAS-mutant (G12D), TP53-mutant (R273H)..."),
    ("HepG2C3A cells", "HepG2/C3A is a clonal derivative of the human hepatocellular carcinoma line HepG2... (完整描述保留)..."),
    ("HOP62 cells", "HOP-62 is a human non-small cell lung carcinoma (NSCLC) cell line... (完整描述保留)... carries KRAS (G12C) and STK11 mutations..."),
    ("Hs766T cells", "Hs 766T is a human pancreatic ductal adenocarcinoma (PDAC) cell line... (完整描述保留)... carries hallmark pancreatic cancer alterations including KRAS G12D mutation, TP53 mutation..."),
]


prompt_direct_sft_de_train = f"""[Start of Prompt]
You are given a biological perturbation classification task.

Task:
Determine whether perturbing {{pert}} in the {{cell_short}} cell line causes differential expression of {{gene}}.

Instructions:
- Use the query information only.
- Do not generate analysis or explanation.
- Output exactly one final label.

Allowed labels:
- yes
- no
[End of Prompt]

[Start of Input]
Drug: {{pert}}
Gene: {{gene}}
Cell line: {{cell_short}}
Drug description: {{pert_desc}}
Gene description: {{gene_desc}}
Cell context: {{cell_desc}}
[End of Input]

[Start of Output]
[End of Output]"""


prompt_direct_sft_de_infer = f"""[Start of Prompt]
You are given a biological perturbation classification task.

Task:
Determine whether perturbing {{pert}} in the {{cell_short}} cell line causes differential expression of {{gene}}.

Instructions:
- Use the query information only.
- Do not generate analysis or explanation.
- Output exactly one final label.

Allowed labels:
- yes
- no
- insufficient
[End of Prompt]

[Start of Input]
Drug: {{pert}}
Gene: {{gene}}
Cell line: {{cell_short}}
Drug description: {{pert_desc}}
Gene description: {{gene_desc}}
Cell context: {{cell_desc}}
[End of Input]

[Start of Output]
[End of Output]"""


# Compatibility with existing stage loader (expects prompt_vcworld_DE or prompt_test_de).
prompt_vcworld_DE = prompt_direct_sft_de_infer


# For retrieved observation formatting when enabled.
choices_de = [
    "yes",
    "no",
]
