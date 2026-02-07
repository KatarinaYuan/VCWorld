
cell_lines = [
    ("C32 cells", "C32 is a human amelanotic melanoma cell line derived from skin and widely used in cancer biology and drug discovery. It harbors hallmark alterations including BRAF V600E mutation, PTEN deletion, CDKN2A mutation, and a TERT promoter mutation, reflecting key oncogenic pathways in melanoma. With a doubling time of ~53 hours, microsatellite stability (MSS), and well-defined HLA typing, C32 serves as a valuable model for immunological studies and dependency analyses in large-scale omics resources such as CCLE and DepMap. Its predominantly European genetic ancestry further contextualizes its relevance for investigating melanoma pathogenesis, signaling dependencies, and targeted therapeutic strategies."),
    ("PANC-1 cells", "PANC-1 is a human pancreatic ductal adenocarcinoma cell line, established from a 56-year-old male patient with pancreatic carcinoma. It is KRAS-mutant (G12D), TP53-mutant (R273H), and carries other alterations typical of pancreatic cancer, making it a canonical in vitro model for studying tumorigenesis, epithelial–mesenchymal transition, chemoresistance, and signaling pathways in pancreatic cancer. PANC-1 cells grow adherently with an epithelial morphology, have a doubling time of ~52 hours, and exhibit a relatively aggressive phenotype with stem-like properties. They are extensively used in drug discovery, metabolic research, and investigations of pancreatic cancer biology due to their robustness and reproducibility."),
    ("HepG2C3A cells", "HepG2/C3A is a clonal derivative of the human hepatocellular carcinoma line HepG2, originally isolated at ATCC. Compared to the parental HepG2, C3A cells exhibit enhanced contact inhibition, more uniform morphology, and improved capacity for producing plasma proteins, including albumin, α-fetoprotein, and various clotting factors. They retain a diploid karyotype, wild-type TP53, and microsatellite stability (MSS). C3A cells are widely used as a model for drug metabolism, hepatotoxicity testing, and liver-specific function assays, given their stable growth characteristics and preserved hepatic functions. With a doubling time of ~40–50 hours, they provide a reproducible platform for both basic liver biology and applied pharmaceutical research."),
    ("HOP62 cells", "HOP-62 is a human non–small cell lung carcinoma (NSCLC) cell line, established from the pleural effusion of a female patient with lung adenocarcinoma. It is part of the NCI-60 cancer cell line panel and has been extensively characterized in pharmacogenomic and drug screening studies. HOP-62 carries KRAS (G12C) and STK11 mutations, alterations that define aggressive tumor biology and influence therapeutic responses. The line shows microsatellite stability (MSS), a doubling time of ~40–50 hours, and adherent epithelial-like growth. Because of its well-documented omics profiles (genomic, transcriptomic, and proteomic) and integration into NCI’s large-scale pharmacology databases, HOP-62 is a widely used model for exploring lung cancer signaling pathways, oncogene-driven vulnerabilities, and candidate drug responses."),
    ("Hs766T cells", "Hs 766T is a human pancreatic ductal adenocarcinoma (PDAC) cell line, established from the lymph node metastasis of a 73-year-old female patient. It carries hallmark pancreatic cancer alterations including KRAS G12D mutation, TP53 mutation, and CDKN2A inactivation, reflecting the canonical molecular landscape of PDAC. The cells grow adherently with an epithelial morphology, display a doubling time of ~60–70 hours, and are microsatellite stable (MSS). Hs 766T is widely used in studies of pancreatic tumor progression, metastasis, stromal interactions, and therapeutic resistance, and is included in large-scale resources such as the NCI-60 panel for pharmacogenomic profiling."),
]


desc_pert = "description of drug that is to perturb the cell"
desc_gene = "description of gene, the impact on which you wish to infer"
desc_context = "description of cell line in which the genes are expressed"
desc_obs = "set of experimental observations that describe the impact of small molecule perturbations on related genes, to contextualize your answer"


prompt_vcworld_DIR = f"""[Start of Prompt]
You are VCWorld, a sophisticated Biological World Model and Causal Reasoning Engine. Your task is to simulate and predict the **direction** of cellular response to drug perturbations.

Goal: Determine if a perturbation of {{pert}} in the {{cell_short}} cell line results in a **Decrease** or **Increase** in the expression of {{gene}}.

Input Data:
- Drug ({{pert}}): {desc_pert}
- Gene ({{gene}}): {desc_gene}
- **Biological Context**: {desc_context}
  *(Instruction: You MUST explicitly utilize the description above AND enhance it with your internal biological knowledge regarding {{cell_short}}'s tissue of origin, hallmark mutations (e.g., KRAS, TP53 status), and signaling idiosyncrasies.)*
- Evidence Set: {desc_obs}

Reasoning Guidelines:
Do not rely on superficial text matching. Perform a stepwise biological simulation to deduce the net directionality (up/down) of the effect.

Output: Provide a structured analysis answering the following steps.

1) **Mechanism & Analogue Identification:**
   Identify drugs in the evidence set that share the same *Mechanism of Action (MoA)* or target specific pathway nodes as {{pert}}.

2) **Specificity & Relevance Analysis (Drug-Gene-Cell Triad):**
   Analyze the potential associations between the Drug, Gene, and Cell Line:
   - **Specificity:** Is the drug's effect broad or specific?
   - **Relevance:** Given the enhanced cell line context (e.g., specific mutations), is the drug's target relevant? (e.g., Is the targeted pathway constitutively active due to a mutation like KRAS G12D, making the cell highly sensitive to inhibition?)

3) **Directional Signaling Cascade Simulation:**
   Trace the signaling cascade initiated by {{pert}} to determine the downstream impact.
   - **Action:** Does {{pert}} inhibit or activate its target?
   - **Propagation:** How does this signal propagate? (e.g., Inhibition of Kinase X -> Reduced phosphorylation of Transcription Factor Y -> Deactivation of TF Y).
   - *Constraint:* Ensure this simulation aligns with the enhanced context of {{cell_short}}.

4) **Regulatory Logic & Evidence Synthesis:**
   Connect the drug's downstream effect to the gene's regulatory requirements to deduce the **Direction of Change**.
   - **Logic Construction:** - If the drug suppresses an *Activator* of {{gene}}, predict **Decrease**. 
     - If the drug suppresses a *Repressor* of {{gene}}, predict **Increase**.
     - (And vice versa for drug activation).
   - **Evidence Support:** Cite specific "Analogue Cases" that show a consistent direction.
   - **Soft Reference:** Briefly refer to "Contrast Cases" or cases with opposite outcomes as supplementary context to refine the directional hypothesis (e.g., to check if the directionality is context-dependent).

5) **Final Deterministic Prediction:**
   Based on the causal logic constructed above, determine the net direction of the perturbation.

   End your response with exactly one of the following options:
   - Decrease. Perturbation of {{pert}} results in a decrease in expression of {{gene}}.
   - Increase. Perturbation of {{pert}} results in an increase in expression of {{gene}}.
   - There is insufficient evidence to determine how perturbation of {{pert}} affects {{gene}}.
[End of Prompt]

[Start of Input]
- Description of molecule drug ({{pert}}): {{pert_desc}}
- Description of gene of interest ({{gene}}): {{gene_desc}}
- Context: {{cell_desc}}
- Examples: {{obs}}
[End of Input]

[Start of Output]
1)
2)
3)
4)
5)
[End of Output]"""

choices_dir = [
    "A) Perturbation of this drug results in a decrease in expression of the gene of interest.",
    "B) Perturbation of this drug results in an increase in expression of the gene of interest."
]