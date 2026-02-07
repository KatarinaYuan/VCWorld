cell_lines = [
    ("C32 cells", "C32 is a human amelanotic melanoma cell line derived from skin... (完整描述保留)... It harbors hallmark alterations including BRAF V600E mutation, PTEN deletion..."),
    ("PANC-1 cells", "PANC-1 is a human pancreatic ductal adenocarcinoma cell line... (完整描述保留)... It is KRAS-mutant (G12D), TP53-mutant (R273H)..."),
    ("HepG2C3A cells", "HepG2/C3A is a clonal derivative of the human hepatocellular carcinoma line HepG2... (完整描述保留)..."),
    ("HOP62 cells", "HOP-62 is a human non–small cell lung carcinoma (NSCLC) cell line... (完整描述保留)... carries KRAS (G12C) and STK11 mutations..."),
    ("Hs766T cells", "Hs 766T is a human pancreatic ductal adenocarcinoma (PDAC) cell line... (完整描述保留)... carries hallmark pancreatic cancer alterations including KRAS G12D mutation, TP53 mutation..."),
]


prompt_vcworld_DE = f"""[Start of Prompt]
You are VCWorld, a sophisticated Biological World Model and Causal Reasoning Engine. Your task is to simulate and predict the cellular response to drug perturbations.

Goal: Determine if a perturbation of {{pert}} in the {{cell_short}} cell line results in the differential expression (DE) of {{gene}}.

Input Data:
- Drug ({{pert}}): {desc_pert}
- Gene ({{gene}}): {desc_gene}
- **Biological Context**: {desc_context}
  *(Instruction: You MUST explicitly utilize the description above AND enhance it with your internal biological knowledge regarding {{cell_short}}'s tissue of origin, hallmark mutations (e.g., KRAS, TP53 status), and signaling idiosyncrasies.)*
- Evidence Set: {desc_obs}

Reasoning Guidelines:
Do not rely on superficial text matching. Perform a stepwise biological simulation as follows.

Output: Provide a structured analysis answering the following steps.

1) **Mechanism & Analogue Identification:**
   Identify drugs in the evidence set that share the same *Mechanism of Action (MoA)* or target specific pathway nodes as {{pert}}.

2) **Specificity & Relevance Analysis (Drug-Gene-Cell Triad):**
   Analyze the potential associations between the Drug, Gene, and Cell Line:
   - **Specificity:** Is the drug's effect broad (e.g., general stress) or specific (e.g., targeted kinase inhibition)? Is the gene's expression tissue-specific?
   - **Relevance:** Given the enhanced cell line context (e.g., its mutations), is the drug's target relevant in this specific cellular environment? (e.g., Does the cell rely on the targeted pathway?)

3) **Downstream Signaling Cascade Simulation:**
   Trace the signaling cascade initiated by {{pert}}. When {{pert}} inhibits/activates its target, which specific downstream kinases, transcription factors, or stress responses are modulated?
   *Constraint:* Ensure this simulation aligns with the enhanced context of {{cell_short}}.

4) **Causal Bridge & Evidence Synthesis:**
   Connect the drug's downstream effect to the gene's regulatory requirements.
   - Construct a logical bridge: Drug -> Target -> Pathway -> TF -> Gene.
   - Cite specific "Analogue Cases" that support this link.
   - *Soft Reference:* Briefly refer to "Contrast Cases" (if available) as supplementary context to see if they offer a different perspective or boundary condition, but focus primarily on constructing the positive mechanism.

5) **Final Deterministic Prediction:**
   Based on the analysis above, determine if the drug effectively perturbs the gene in this specific cell line.
   
   End your response with exactly one of the following options:
   - No. Perturbation of {{pert}} does not impact {{gene}}.
   - Yes. Perturbation of {{pert}} results in differential expression of {{gene}}.
   - There is insufficient evidence to determine how Perturbation of {{pert}} affects {{gene}}.
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


choices_de = [
    "A) Perturbation of this drug does not impact the gene of interest.",
    "B) Perturbation of this drug results in differential expression of the gene of interest.",
]