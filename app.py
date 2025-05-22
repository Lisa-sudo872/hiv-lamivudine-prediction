import gradio as gr
import joblib
import pandas as pd
from collections import Counter

import gradio as gr
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier

import gradio as gr
import joblib, numpy as np, pandas as pd, xgboost as xgb
from scipy import sparse   # TF-IDF is a sparse CSR matrix

# ─── Load the TF-IDF vectorizer ────────────────────────────────────────────────
tfidf_vectorizer = joblib.load("tfidf_vectorizer_fixed.pkl")

# ─── Load the six native XGBoost boosters (one per drug) ──────────────────────
boosters = []
for i in range(6):                       # 0-5  (3TC … TDF)
    booster = xgb.Booster()
    booster.load_model(f"xgb_booster_label_{i}.json")
    boosters.append(booster)

# helper that does raw booster prediction → class (0-4)
def boosters_predict(X_csr):
    """
    X_csr : scipy CSR matrix with TF-IDF features (n_samples × n_features)
    Returns np.ndarray shape (n_samples, 6) with integer class 0-4 per drug.
    """
    dmat = xgb.DMatrix(X_csr)            # XGBoost’s native data container
    preds = []
    for bst in boosters:
        # bst.predict gives probability for each class (shape n_samples × 5)
        proba = bst.predict(dmat)
        preds.append(np.argmax(proba, axis=1))   # pick highest-prob class
    return np.vstack(preds).T                   # shape (n_samples, 6)


# ─────────────────────────────
# Constants - Labels and Dictionaries for drugs and resistance levels

drug_labels = ["3TC", "ABC", "D4T", "AZT", "DDI", "TDF"]

full_names = {
    "3TC": "Lamivudine",
    "ABC": "Abacavir",
    "AZT": "Zidovudine",
    "D4T": "Stavudine",
    "DDI": "Didanosine",
    "TDF": "Tenofovir"
}

res_labels = {
    0: "Susceptible",
    1: "Potential Low Resistance",
    2: "Low Resistance",
    3: "Intermediate Resistance",
    4: "High Resistance"
}

# ─────────────────────────────
# Known drug resistance mutations (DRMs) relevant to NRTIs
# These are from clinical databases like Stanford HIVdb
KNOWN_NRTI_DRMS = {
    "M184V","M184I","K65R","K65N","D67N","D67G","K70R","K70E","K70G","K70N",
    "L74V","L74I","A62V","V75T","F77L","Y115F","Q151M","T69D","T69N","T69I",
    "M41L","L210W","T215Y","T215F","K219Q","K219E"
}

# ─────────────────────────────
# Detailed mutation effects and clinical notes —
# Provide insight for clinicians or researchers about what each mutation means

CROSS_EFFECTS_DETAILED = {
    "M184V": {
        "effect": "High-level resistance to 3TC and FTC; increases AZT and TDF susceptibility",
        "clinical": "May keep AZT and TDF effective despite 3TC resistance."
    },
    "M184I": {
        "effect": "Similar to M184V but less common",
        "clinical": "Also causes 3TC resistance and AZT hypersusceptibility."
    },
    "K65R": {
        "effect": "Reduces susceptibility to TDF, ABC, DDI, and 3TC",
        "clinical": "Avoid TDF and ABC-containing regimens if K65R present."
    },
    "Q151M": {
        "effect": "Multi-NRTI resistance affecting 3TC, ABC, DDI, TDF, AZT, D4T",
        "clinical": "Indicates broad NRTI resistance, alternative drug classes needed."
    },
    "M41L": {
        "effect": "Thymidine Analog Mutation (TAM) causing resistance to AZT and D4T",
        "clinical": "Compromises thymidine analogs, often in combination with other TAMs."
    },
    "D67N": {
        "effect": "TAM contributing to AZT and D4T resistance",
        "clinical": "Part of TAM cluster; increases resistance when combined with others."
    },
    "K70R": {
        "effect": "TAM that reduces susceptibility to AZT and D4T",
        "clinical": "Common TAM, associated with cross-resistance."
    },
    "L210W": {
        "effect": "TAM increasing resistance to AZT and D4T",
        "clinical": "Enhances TAM effects."
    },
    "T215Y": {
        "effect": "Major TAM causing high-level AZT resistance",
        "clinical": "Strongly compromises AZT efficacy."
    },
    "T215F": {
        "effect": "TAM similar to T215Y",
        "clinical": "Contributes to high AZT resistance."
    },
    "K219Q": {
        "effect": "Minor TAM contributing to AZT resistance",
        "clinical": "Often seen with other TAMs."
    },
    "K219E": {
        "effect": "Minor TAM similar to K219Q",
        "clinical": "Adds to resistance profile."
    },
    "A62V":{
        "effect": "generally has minimal impact on NRTI susceptibility.",
        "clinical": "part of the Q151M complex, it helps cause broad NRTI resistance"
    },
    "T69D":{
        "effect": "alone usually causes little or no resistance.",
        "clinical": "presence often indicates multi-NRTI resistance."
    },
    "L74V":{
        "effect": "Confers reduced susceptibility (resistance) to ABC and DDI.",
        "clinical": "compromise the effectiveness of ABC or ddI"
    }
    # Add more mutations as needed...
}

# ─────────────────────────────
# Wild-type (WT) reference RT sequence (240 amino acids)
# We use this to identify mutations by position
# This sequence is standard from HIV-1 RT (Reverse Transcriptase) region

WT = (
    "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIKKKDSTKWRKL"
    "VDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDKDFRKYTAFTIPSINNETPGIRYQYNVL"
    "PQGWKGSPAIFQSSMTKILEPFRKQNPDIVIYQYMDDLYVGSDLEIGQHRTKIEELRQHLLRWGFTTPDKKHQKE"
    "PPFLWMGYELHPDKWTVQPIVLPEKDSWTVNDIQKLVGKLNWASQIYAGIKVKQLCKLLRGTKALTEVIPLTEEAE"
    "LELAENREILKEPVHGVYYDPSKDLIAEIQKQGQGQWTYQIYQEPFKNLKTGKYARMRGAHTNDVKQLTEAVQKI"
    "ATESIVIWGKTPKFKLPIQKETWEAWWTEYWQATWIPEWEFVNTPPLVKLWYQLEKEPIVGAETFYVDGAANRET"
    "KLGKAGYVTDRGRQKVVSLTDTTNQKTELQAIHLALQDSGLEVNIVTDSQYALGIIQAQPDKSESELVSQIIEQL"
    "IKKEKVYLAWVPAHKGIGGNEQVDKLVSAGIRKVL"
)[:240]

# ─────────────────────────────
# Function: Identify mutations compared to WT sequence
def list_mutations(seq: str):
    """
    Compares input RT sequence to WT.
    Returns list of mutation strings, e.g. "M184V" meaning WT=M at pos184 mutated to V.
    Pads sequence if shorter than 240 aa with WT amino acids.
    """
    seq = seq.upper().replace(" ", "").replace("\n", "")
    if len(seq) < 240:
        seq += WT[len(seq):]
    else:
        seq = seq[:240]

    mutations = []
    for i, (wt_aa, seq_aa) in enumerate(zip(WT, seq), start=1):
        if seq_aa != wt_aa:
            mutations.append(f"{wt_aa}{i}{seq_aa}")
    return mutations

# ─────────────────────────────
# Function: Generate k-mers from sequence for vectorization
def generate_kmers(sequence: str, k=5):
    """
    Breaks sequence into overlapping k-mers separated by spaces.
    Example: sequence='ABCDE', k=3 -> 'ABC BCD CDE'
    """
    return " ".join(sequence[i:i+k] for i in range(len(sequence) - k + 1))

# ─────────────────────────────
# Function: Suggest treatment action based on resistance level for a drug
def suggest_alt(drug, resistance_label):
    """
    Provides recommendation based on resistance category.
    """
    if resistance_label in {"High Resistance", "Intermediate Resistance"}:
        return "Consider switching"
    else:
        return "Continue use"

# ─────────────────────────────
# Function: Get clinical details text for mutations detected
def mutation_details(mutations):
    """
    Returns detailed clinical effect descriptions for known mutations.
    Unknown mutations get a placeholder text.
    """
    details = []
    for m in mutations:
        if m in CROSS_EFFECTS_DETAILED:
            det = CROSS_EFFECTS_DETAILED[m]
            details.append(f"{m}: {det['effect']} ({det['clinical']})")
        else:
            details.append(f"{m}: Unknown clinical impact")
    return "\n".join(details) if details else "None"

# ─────────────────────────────
# Main prediction + annotation function
def predict_and_display(seq, drug_filter):
    """
    Takes input RT sequence and drug filter,
    returns resistance predictions, mutation clinical details,
    and unknown mutations found.
    """

    seq = seq.strip()

    # Check length constraint
    if len(seq) < 100 or len(seq) > 240:
        raise gr.Error("Sequence length must be between 100 and 240 amino acids.")

    # Vectorize input sequence using k-mer TF-IDF
    kmer_seq = generate_kmers(seq, k=5)
    X = tfidf_vectorizer.transform([kmer_seq])

    # Predict resistance levels for all drugs (multioutput)
    preds = boosters_predict(X)[0]   # first (and only) row  # Assuming shape (1,6), one per drug

    # Identify mutations in sequence
    mutations = list_mutations(seq)
    known_mutations = [m for m in mutations if m in KNOWN_NRTI_DRMS]
    unknown_mutations = [m for m in mutations if m not in KNOWN_NRTI_DRMS]

    # Build results table (DataFrame)
    rows = []
    for i, drug in enumerate(drug_labels):
        label = res_labels[int(preds[i])]
        hint = ""

        # For known mutations, check if there's a clinical effect relevant to this drug
        for m in known_mutations:
            if m in CROSS_EFFECTS_DETAILED and drug in {"AZT", "D4T", "3TC"}:
                hint = CROSS_EFFECTS_DETAILED[m]["effect"]
                break

        rows.append({
            "Drug": drug,
            "Full Name": full_names[drug],
            "Resistance": label + (f"  ({hint})" if hint else ""),
            "Suggested Alternative": suggest_alt(drug, label)
        })

    df = pd.DataFrame(rows)

    # Filter results by selected drug if not "All"
    if drug_filter != "All":
        df = df[df["Drug"] == drug_filter]

    # Mutation clinical details for UI output
    known_details_text = mutation_details(known_mutations)
    unknown_details_text = ", ".join(unknown_mutations) if unknown_mutations else "None"

    return df, known_details_text, unknown_details_text

# ─────────────────────────────
# Examples for the Gradio interface quick testing

examples = [
    [
        "MLWQTKVTVLDVGDAYFSVPLDLEGKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDLEGKWRKLVDFRELNKRTQDFWEVQLGVKHPAGLKKKKSVTVLDVGDAYFSVPLDKDFRKYTAFTIPSINNETPGIRYQYNVL",
        "All"
    ],
    [
        "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIKKKDSTKWRKLVDDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDKDFRKYTAFTIPSINNETPGIRYQYNVL",
        "3TC"
    ],
]

# ─────────────────────────────
# Gradio UI setup

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # HIV-1 NRTI Drug Resistance Predictor (Lamivudine-focused)
        Enter a Reverse Transcriptase (RT) amino acid sequence (length 100-240).
        The model predicts resistance levels to key NRTIs, identifies known mutations,
        and provides clinical insights.
        **Select Drug Filter:** View all drugs or focus on one.
        ---
        """
    )

    with gr.Row():
        seq_input = gr.Textbox(
            label="RT Sequence (AA)", lines=6, placeholder="Paste HIV-1 RT amino acid sequence here..."
        )
        drug_filter = gr.Dropdown(
            label="Filter by Drug",
            choices=["All"] + drug_labels,
            value="All"
        )

    predict_btn = gr.Button("Predict Resistance")

    output_table = gr.DataFrame(
        headers=["Drug", "Full Name", "Resistance", "Suggested Alternative"],
        interactive=False,
        label="Resistance Predictions"
    )

    known_mut_output = gr.Textbox(
        label="Known Mutation Clinical Effects",
        interactive=False,
        lines=6,
        max_lines=12
    )

    unknown_mut_output = gr.Textbox(
        label="Unknown Mutations Found",
        interactive=False,
        lines=3
    )

    # Link button click to prediction function
    predict_btn.click(
        fn=predict_and_display,
        inputs=[seq_input, drug_filter],
        outputs=[output_table, known_mut_output, unknown_mut_output]
    )

    # Add example buttons
    gr.Examples(
        examples=examples,
        inputs=[seq_input, drug_filter],
        cache_examples=False
    )

demo.launch()
