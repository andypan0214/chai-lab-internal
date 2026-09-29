"""Lab-internal Streamlit frontend for Chai-1 on the HPC.

    streamlit run streamlit_app.py

UI only. The architecture boundary is:

    this file (form + result views)
      -> form_state / presets      (user-facing state, CSV example presets)
      -> input_builder / restraint_builder   (FASTA + restraint CSV, validation)
      -> job_manager               (job directory + sbatch run_chai.slurm)
      -> run_chai.py -> chai_lab   (inference on a compute node; GPU or CPU
                                    per the server's execution profile)
      -> result_adapter            (reads Chai's CIF / scores / PAE files)

Nothing here serializes FASTA or restraint CSV, runs inference or computes
scores. See README_HPC.md, "Web app".
"""

from __future__ import annotations

import html
import io
import json
import logging
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from form_state import (
    FormValidationError,
    config_to_form,
    empty_form,
    form_to_config,
    input_restraint_rows,
    input_sequence_rows,
    new_molecule_row,
    new_restraint_row,
    set_restraint_type,
    total_known_tokens,
    validate_config,
)
from input_builder import VALID_MOLECULE_TYPES, build_chai_input
from job_manager import FAILED, QUEUED, RUNNING, SUCCESS, TERMINAL_STATUSES, JobError, JobStore
from offline import missing_model_assets, offline_mode_enabled
from presets import PresetError, list_presets, load_preset
from restraint_builder import VALID_RESTRAINT_TYPES
from result_adapter import Candidate, ResultError, load_candidates, output_files

st.set_page_config(page_title="Chai-1 structure prediction", layout="wide")

# Vendored 3Dmol.js (see vendor/3Dmol/README.md). It is inlined into the
# viewer iframe, so the browser never fetches JavaScript from anywhere else.
THREE_DMOL_JS_PATH = Path(__file__).resolve().parent / "vendor" / "3Dmol" / "3Dmol-min.js"

OFFLINE = offline_mode_enabled()


@st.cache_resource
def _three_dmol_js() -> str:
    # "</script" inside the library would end the inline <script> element early.
    return THREE_DMOL_JS_PATH.read_text(encoding="utf-8").replace("</script", "<\\/script")

# pLDDT bins and colours (AlphaFold/Chai convention), highest first.
PLDDT_BINS = [
    (90, "#0053D6", "[100, 90]"),
    (70, "#65CBF3", "[90, 70]"),
    (50, "#FFDB13", "[70, 50]"),
    (0, "#FF7D45", "[50, 0]"),
]
PAE_COLORSCALE = [[0.0, "#FFFFFF"], [1.0, "#6B5FD3"]]

STATUS_STYLE = {
    SUCCESS: ("#DCFCE7", "#15803D"),
    RUNNING: ("#DBEAFE", "#1D4ED8"),
    QUEUED: ("#F3F4F6", "#4B5563"),
    FAILED: ("#FEE2E2", "#B91C1C"),
}

st.markdown(
    """
<style>
.cz-table {width:100%; border-collapse:collapse; font-size:0.92rem;}
.cz-table th {text-align:left; font-weight:600; color:#374151; padding:10px 12px;
              border-bottom:1px solid #818CF8;}
.cz-table td {padding:12px; color:#4B5563; vertical-align:top; border:none;}
.cz-seq {font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
         word-break:break-all; font-size:0.85rem; line-height:1.6;}
.cz-badge {display:inline-block; padding:4px 12px; border-radius:6px; font-size:0.8rem;
           font-weight:600; letter-spacing:0.02em;}
.cz-chip {display:inline-block; padding:6px 18px; margin-right:12px; border-radius:8px;
          background:#F9FAFB; border:1px solid #E5E7EB; font-size:0.95rem;}
.cz-legend {display:flex; justify-content:space-around; margin-top:6px; font-size:0.85rem;}
.cz-dot {display:inline-block; width:14px; height:14px; border-radius:50%;
         margin-right:8px; vertical-align:-2px;}
.cz-caption {text-align:center; font-size:0.85rem; color:#374151; margin-top:4px;}
</style>
""",
    unsafe_allow_html=True,
)


def _store() -> JobStore:
    return JobStore()


# ===========================================================================
# Predict page: form state helpers
# ===========================================================================
ss = st.session_state


def _form() -> dict:
    if "form" not in ss:
        ss.form = empty_form()
    return ss.form


def _reset_widgets() -> None:
    for key in [k for k in ss.keys() if isinstance(k, str) and k.startswith("w_")]:
        del ss[key]


def _seed(key: str, value) -> str:
    """Give a widget its value from the form unless Streamlit already holds one
    (widget state is dropped while a widget is hidden, the form dict is not)."""
    if key not in ss:
        ss[key] = value
    return key


def _flash(kind: str, message: str) -> None:
    ss.flash = (kind, message)


def _load_preset_cb(path) -> None:
    try:
        config = load_preset(path)
    except PresetError as e:
        _flash("error", f"This example could not be loaded: {e}")
        return
    ss.form = config_to_form(config)
    ss.pop("submit_errors", None)
    _reset_widgets()
    _flash("success", f"Loaded example '{config['name']}'.")


def _reset_cb() -> None:
    ss.form = empty_form()
    ss.pop("submit_errors", None)
    _reset_widgets()


def _templates_cb() -> None:
    form = _form()
    form["use_templates"] = ss.w_use_templates
    if ss.w_use_templates and not ss.get("w_use_msa"):
        ss.w_use_msa = form["use_msa"] = True
        _flash("info", "Use MSAs was turned on: templates require MSAs.")


def _msa_cb() -> None:
    form = _form()
    form["use_msa"] = ss.w_use_msa
    if not ss.w_use_msa and ss.get("w_use_templates"):
        ss.w_use_templates = form["use_templates"] = False
        _flash("info", "Use Templates was turned off: templates require MSAs.")


def _add_molecule_cb() -> None:
    _form()["molecules"].append(new_molecule_row())


def _remove_molecule_cb(uid: str) -> None:
    form = _form()
    form["molecules"] = [m for m in form["molecules"] if m["uid"] != uid]


def _move_molecule_cb(uid: str, delta: int) -> None:
    mols = _form()["molecules"]
    i = next(i for i, m in enumerate(mols) if m["uid"] == uid)
    j = i + delta
    if 0 <= j < len(mols):
        mols[i], mols[j] = mols[j], mols[i]


def _add_restraint_cb() -> None:
    form = _form()
    form["restraints"].append(new_restraint_row())
    form["specify_restraints"] = True
    ss.w_specify_restraints = True


def _remove_restraint_cb(uid: str) -> None:
    form = _form()
    form["restraints"] = [r for r in form["restraints"] if r["uid"] != uid]


def _restraint_type_cb(uid: str) -> None:
    row = next(r for r in _form()["restraints"] if r["uid"] == uid)
    set_restraint_type(row, ss[f"w_res_{uid}_type"])
    ss[f"w_res_{uid}_r1"] = row["residue1"]


# ===========================================================================
# Predict page
# ===========================================================================
def predict_page() -> None:
    form = _form()
    st.title("Predict new structure with Chai-1")

    # --- examples -------------------------------------------------------
    show_candidates = os.environ.get("CHAI_WEB_SHOW_CANDIDATE_PRESETS") == "1"
    presets = list_presets(include_candidates=show_candidates)
    with st.container(border=True):
        st.markdown("Try an example")
        if presets:
            cols = st.columns(len(presets) + 1)
            for col, (label, path) in zip(cols, presets):
                col.button(label, key=f"preset_{path.name}", on_click=_load_preset_cb, args=(path,))
        else:
            st.caption("No examples are installed.")

    if "flash" in ss:
        kind, message = ss.pop("flash")
        getattr(st, kind)(message)

    # --- options --------------------------------------------------------
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        form["use_msa"] = c1.checkbox(
            "Use MSAs (slower)", key=_seed("w_use_msa", form["use_msa"]), on_change=_msa_cb,
            help="Search the ColabFold MMseqs2 server for multiple sequence alignments.",
        )
        c1.caption(
            "When MSAs are disabled, Chai-1 uses ESM language-model embeddings "
            "instead, which is much faster."
        )
        form["use_templates"] = c2.checkbox(
            "Use Templates", key=_seed("w_use_templates", form["use_templates"]),
            on_change=_templates_cb,
        )
        c2.caption(
            "Search for known structures with similar sequences and give them to "
            "the model as templates. Requires MSAs."
        )
        if OFFLINE:
            for col in (c1, c2):
                col.markdown(
                    ":orange[**Unavailable: this server runs in offline mode** "
                    "(MSA/template search uses an external server).]"
                )
        form["specify_restraints"] = c3.checkbox(
            "Specify restraints", key=_seed("w_specify_restraints", form["specify_restraints"])
        )
        c3.caption(
            "Restraints tell the model which residues or chains should be close "
            "together in the predicted complex."
        )

    # --- name -----------------------------------------------------------
    n1, n2 = st.columns([6, 1], vertical_alignment="bottom")
    form["name"] = n1.text_input(
        "Prediction name", key=_seed("w_name", form["name"]),
        placeholder="Name: e.g. protein-1 (left blank, a name is generated)",
        label_visibility="collapsed", max_chars=100,
    )
    n2.button("Reset", on_click=_reset_cb, width="stretch", icon=":material/cancel:")

    # --- molecules ------------------------------------------------------
    with st.container(border=True):
        widths = [0.35, 1.3, 0.8, 1.1, 6, 0.35]
        head = st.columns(widths)
        for col, label in zip(head[1:5], ["Molecule Type", "Copies", "Chain IDs", "Sequence Text"]):
            col.markdown(f"**{label}**")
        for i, row in enumerate(form["molecules"]):
            uid = row["uid"]
            cols = st.columns(widths)
            with cols[0]:
                st.button("", key=f"w_mol_{uid}_up", icon=":material/keyboard_arrow_up:",
                          on_click=_move_molecule_cb, args=(uid, -1), disabled=i == 0,
                          type="tertiary")
                st.button("", key=f"w_mol_{uid}_down", icon=":material/keyboard_arrow_down:",
                          on_click=_move_molecule_cb, args=(uid, 1),
                          disabled=i == len(form["molecules"]) - 1, type="tertiary")
            row["molecule_type"] = cols[1].selectbox(
                "Molecule Type", VALID_MOLECULE_TYPES, label_visibility="collapsed",
                key=_seed(f"w_mol_{uid}_type", row["molecule_type"]),
            )
            row["copies"] = cols[2].number_input(
                "Copies", min_value=1, max_value=50, step=1, label_visibility="collapsed",
                key=_seed(f"w_mol_{uid}_copies", int(row["copies"])),
            )
            row["chain_ids"] = cols[3].text_input(
                "Chain IDs", label_visibility="collapsed", placeholder="auto",
                key=_seed(f"w_mol_{uid}_chains", row["chain_ids"]),
                help="One ID per copy, comma-separated (e.g. A or A,B). Leave blank to assign automatically.",
            )
            placeholder = (
                "Enter a SMILES string, e.g. CCO"
                if row["molecule_type"] == "ligand"
                else "Enter sequence here with modifications as CCD code in parentheses; "
                "for example: (ACE)GQLEEIAKQLEEIAWQLEEIAQG(NH2)"
            )
            row["sequence"] = cols[4].text_area(
                "Sequence Text", label_visibility="collapsed", placeholder=placeholder,
                height=110, key=_seed(f"w_mol_{uid}_seq", row["sequence"]),
            )
            cols[5].button("", key=f"w_mol_{uid}_rm", icon=":material/close:",
                           on_click=_remove_molecule_cb, args=(uid,), type="tertiary",
                           disabled=len(form["molecules"]) == 1, help="Remove molecule")

        try:
            chains = build_chai_input(form_to_config({**form, "specify_restraints": False})["molecules"]).chains
            st.caption("Chains: " + ", ".join(f"**{c.chain_id}** ({c.molecule_type})" for c in chains))
        except Exception:
            pass

    # --- restraints -----------------------------------------------------
    if form["specify_restraints"]:
        with st.container(border=True):
            widths = [1.2, 1.2, 1.2, 1.2, 1.2, 0.9, 0.35]
            head = st.columns(widths)
            for col, label in zip(head, ["Restraint Type", "Chain 1", "Residue index 1",
                                         "Chain 2", "Residue index 2", "Distance (Å)"]):
                col.markdown(f"**{label}**")
            if not form["restraints"]:
                st.caption("No restraints yet. Use **Add restraint** below.")
            for row in form["restraints"]:
                uid = row["uid"]
                cols = st.columns(widths)
                row["type"] = cols[0].selectbox(
                    "Restraint Type", VALID_RESTRAINT_TYPES, label_visibility="collapsed",
                    key=_seed(f"w_res_{uid}_type", row["type"]),
                    on_change=_restraint_type_cb, args=(uid,),
                )
                pocket = row["type"] == "pocket"
                row["chain1"] = cols[1].text_input(
                    "Chain 1", label_visibility="collapsed",
                    placeholder="Whole chain" if pocket else "Chain",
                    key=_seed(f"w_res_{uid}_c1", row["chain1"]),
                    help="Pocket: the whole chain that should be near the target residue." if pocket else None,
                )
                r1 = cols[2].text_input(
                    "Residue index 1", label_visibility="collapsed",
                    placeholder="— (whole chain)" if pocket else "e.g. C387",
                    key=_seed(f"w_res_{uid}_r1", row["residue1"]), disabled=pocket,
                )
                row["residue1"] = "" if pocket else r1
                row["chain2"] = cols[3].text_input(
                    "Chain 2", label_visibility="collapsed",
                    placeholder="Target chain" if pocket else "Chain",
                    key=_seed(f"w_res_{uid}_c2", row["chain2"]),
                )
                row["residue2"] = cols[4].text_input(
                    "Residue index 2", label_visibility="collapsed",
                    placeholder="Target residue, e.g. Y101" if pocket else "e.g. Y101",
                    key=_seed(f"w_res_{uid}_r2", row["residue2"]),
                )
                row["distance"] = cols[5].number_input(
                    "Distance (Å)", min_value=0.0, step=0.5, format="%.1f",
                    label_visibility="collapsed",
                    key=_seed(f"w_res_{uid}_dist", float(row["distance"])),
                )
                cols[6].button("", key=f"w_res_{uid}_rm", icon=":material/close:",
                               on_click=_remove_restraint_cb, args=(uid,), type="tertiary",
                               help="Remove restraint")
            st.caption(
                "**Contact**: a residue in chain 1 and a residue in chain 2 should be within "
                "the distance. **Pocket**: any atom of the whole chain 1 should be within the "
                "distance of the target residue in chain 2 (no residue on the chain 1 side). "
                "Residues are a one-letter code followed by the 1-based position, e.g. "
                "\"R42\". Restraints can only reference protein chains."
            )

    # --- actions --------------------------------------------------------
    a1, a2, _, a4 = st.columns([1.2, 1.2, 3.6, 2])
    a1.button("Add molecule", icon=":material/add:", on_click=_add_molecule_cb, width="stretch")
    a2.button("Add restraint", icon=":material/add:", on_click=_add_restraint_cb, width="stretch")
    submitted = a4.button("Predict 3D structure", type="primary", icon=":material/arrow_forward:",
                          width="stretch")

    if submitted:
        _submit(form)
    for message in ss.get("submit_errors", []):
        st.error(message, icon=":material/error:")


def _submit(form: dict) -> None:
    ss.pop("submit_errors", None)
    try:
        validated = validate_config(form_to_config(form), offline=OFFLINE)
    except FormValidationError as e:
        ss.submit_errors = e.messages
        return
    if OFFLINE:
        problems = missing_model_assets()
        if problems:
            # Details (which may contain server paths) go to the server log only.
            logging.getLogger("streamlit_app").error("Offline asset preflight failed: %s", problems)
            ss.submit_errors = [
                "This server runs in offline mode, but the Chai-1 model files are not "
                "installed yet, so predictions cannot run. Please contact the administrator "
                "(details are in the web server log)."
            ]
            return
    try:
        with st.spinner("Submitting prediction ..."):
            job = _store().create_and_submit(validated)
    except (JobError, OSError) as e:
        ss.submit_errors = [f"The prediction could not be created: {e}"]
        return
    if job["status"] == FAILED:
        ss.submit_errors = [f"The prediction could not be submitted: {job['error']}"]
        return
    ss.open_job = job["job_id"]
    st.switch_page(PAGES["predictions"])


# ===========================================================================
# Predictions page
# ===========================================================================
def _msa_label(job: dict) -> str:
    return "MMseqs2" if job["use_msa"] else "None"


def _fmt_time(iso: str) -> str:
    return iso.replace("T", " ")


def _badge(status: str) -> str:
    bg, fg = STATUS_STYLE.get(status, ("#F3F4F6", "#374151"))
    return f'<span class="cz-badge" style="background:{bg};color:{fg}">{html.escape(status)}</span>'


def _html_table(columns: list[str], rows: list[list[str]], raw_cols: set[int] = frozenset()) -> str:
    """Rows are escaped unless their column index is in raw_cols (pre-built HTML)."""
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(
            f"<td>{cell if i in raw_cols else html.escape(str(cell))}</td>" for i, cell in enumerate(row)
        ) + "</tr>"
        for row in rows
    )
    return f'<table class="cz-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def predictions_page() -> None:
    job_id = st.query_params.get("job") or ss.pop("open_job", None)
    if job_id:
        st.query_params["job"] = job_id
        result_view(job_id)
        return

    st.title("Predictions")
    store = _store()
    jobs = store.list_jobs()
    if not jobs:
        st.info("No predictions yet. Start one from **Predict new structure**.")
        return
    jobs = [store.refresh(j) for j in jobs]

    df = pd.DataFrame(
        {
            "Name": [j["name"] for j in jobs],
            "MSAs": [_msa_label(j) for j in jobs],
            "Templates": ["Yes" if j["use_templates"] else "No" for j in jobs],
            "Restraints": ["Yes" if j["specify_restraints"] else "No" for j in jobs],
            "Created at": [_fmt_time(j["created_at"]) for j in jobs],
            "Last updated": [_fmt_time(j["updated_at"]) for j in jobs],
            "Status": [j["status"] for j in jobs],
        }
    )

    def _status_css(value: str) -> str:
        bg, fg = STATUS_STYLE.get(value, ("#F3F4F6", "#374151"))
        return f"background-color:{bg};color:{fg};font-weight:600"

    st.caption("Select a row to open the prediction.")
    event = st.dataframe(
        df.style.map(_status_css, subset=["Status"]),
        hide_index=True, width="stretch",
        on_select="rerun", selection_mode="single-row", key="jobs_table",
    )
    if event.selection.rows:
        st.query_params["job"] = jobs[event.selection.rows[0]]["job_id"]
        st.rerun()

    if any(j["status"] not in TERMINAL_STATUSES for j in jobs):
        _auto_refresh_list(tuple((j["job_id"], j["status"]) for j in jobs))


@st.fragment(run_every="30s")
def _auto_refresh_list(snapshot: tuple) -> None:
    store = _store()
    for job_id, status in snapshot:
        if status not in TERMINAL_STATUSES and store.refresh(store.get_job(job_id))["status"] != status:
            st.rerun()


@st.fragment(run_every="20s")
def _auto_refresh_job(job_id: str, status: str) -> None:
    store = _store()
    if store.refresh(store.get_job(job_id))["status"] != status:
        st.rerun()


# ===========================================================================
# Result view
# ===========================================================================
def result_view(job_id: str) -> None:
    store = _store()
    if st.button("All predictions", icon=":material/arrow_back:", type="tertiary"):
        del st.query_params["job"]
        ss.pop("jobs_table", None)  # drop the row selection that opened this job
        st.rerun()
    try:
        job = store.refresh(store.get_job(job_id))
        config = store.get_config(job_id)
    except (JobError, OSError, ValueError):
        st.error("This prediction does not exist.")
        return

    st.title("Structure prediction results")
    st.markdown(
        _html_table(
            ["Name", "MSAs", "Templates", "Restraints", "Created at", "Last updated", "Status"],
            [[job["name"], _msa_label(job), "Yes" if job["use_templates"] else "No",
              "Yes" if job["specify_restraints"] else "No", _fmt_time(job["created_at"]),
              _fmt_time(job["updated_at"]), _badge(job["status"])]],
            raw_cols={6},
        ),
        unsafe_allow_html=True,
    )

    seq_rows = input_sequence_rows(config, config["resolved_chains"])

    if job["status"] in (QUEUED, RUNNING):
        st.info(
            "The prediction is waiting for compute resources." if job["status"] == QUEUED
            else "The prediction is running. This page updates automatically.",
            icon=":material/hourglass_top:",
        )
        _auto_refresh_job(job_id, job["status"])
    elif job["status"] == FAILED:
        st.error(job.get("error") or "The prediction failed.")
        log = store.log_tail(job_id)
        if log:
            with st.expander("Technical details"):
                st.code(log, language=None)
    selected = _render_results(store, job, seq_rows) if job["status"] == SUCCESS else None

    _render_inputs(config, seq_rows)
    if selected is not None:
        _render_downloads(store, job, *selected)


def _render_results(store: JobStore, job: dict, seq_rows: list[dict]) -> tuple[Candidate, bool] | None:
    """Scores + structure/PAE card. Returns (selected candidate, PAE saved?)."""
    try:
        candidates = load_candidates(store.output_dir(job["job_id"]))
    except (ResultError, OSError, ValueError) as e:
        st.error(f"The result files could not be read: {e}")
        return None
    if not candidates:
        st.error("No structure candidates were found for this prediction.")
        return None

    if len(candidates) > 1:
        cand = st.selectbox(
            "Candidate",
            candidates,
            index=0,
            format_func=lambda c: (
                f"Rank {c.rank} of {len(candidates)} · model_idx {c.candidate_index} · "
                f"aggregate score {c.aggregate_score:.4f}"
            ),
            key=f"cand_{job['job_id']}",
        )
    else:
        cand = candidates[0]

    st.markdown(
        f'<span class="cz-chip"><b>ipTM</b> = {cand.iptm:.4f}</span>'
        f'<span class="cz-chip"><b>pTM</b> = {cand.ptm:.4f}</span>',
        unsafe_allow_html=True,
    )

    confidence = cand.load_confidence()
    with st.container(border=True):
        left, right = st.columns([1.15, 1])
        with left:
            _embed_html(_structure_html(cand.cif_path.read_text()), height=520)
            legend = "".join(
                f'<span><span class="cz-dot" style="background:{color}"></span>{label}</span>'
                for _, color, label in PLDDT_BINS
            )
            st.markdown(
                f'<div class="cz-legend">{legend}</div>'
                '<div class="cz-caption">Colored by predicted LDDT (pLDDT)</div>',
                unsafe_allow_html=True,
            )
        with right:
            if confidence is not None and "pae" in confidence:
                pae = confidence["pae"]
                st.plotly_chart(_pae_figure(pae), width="stretch",
                                config={"displaylogo": False})
                expected = total_known_tokens(seq_rows)
                if expected is not None and expected != pae.shape[0]:
                    st.warning(
                        f"The PAE matrix has {pae.shape[0]} tokens but the inputs suggest "
                        f"{expected}; token offsets below may not line up with the matrix."
                    )
            else:
                st.info("The PAE matrix was not saved for this prediction.")

    return cand, confidence is not None


def _embed_html(content: str, height: int) -> None:
    if hasattr(st, "iframe"):
        st.iframe(content, height=height)
    else:  # Streamlit releases before st.iframe
        import streamlit.components.v1 as components

        components.html(content, height=height)


def _structure_html(cif_text: str) -> str:
    bins_js = json.dumps([[lo, color] for lo, color, _ in PLDDT_BINS])
    cif_js = json.dumps(cif_text).replace("</", "<\\/")  # keep "</script>" inert
    return f"""
<div id="viewer" style="width:100%;height:500px;position:relative;"></div>
<script>{_three_dmol_js()}</script>
<script>
  const bins = {bins_js};
  // B_iso_or_equiv holds Chai's per-atom pLDDT on a 0-100 scale.
  function plddtColor(atom) {{
    for (const [lo, color] of bins) {{ if (atom.b >= lo) return color; }}
    return bins[bins.length - 1][1];
  }}
  const viewer = $3Dmol.createViewer(document.getElementById("viewer"), {{backgroundColor: "white"}});
  viewer.addModel({cif_js}, "cif");
  viewer.setStyle({{}}, {{cartoon: {{colorfunc: plddtColor}}}});
  viewer.setStyle({{hetflag: true}}, {{stick: {{colorfunc: plddtColor}}}});
  viewer.zoomTo();
  viewer.render();
</script>
"""


def _pae_figure(pae: np.ndarray) -> go.Figure:
    n = pae.shape[0]
    # Lower bound 0 = lower edge of Chai's PAE bins (0-32 A); upper bound is
    # this matrix's actual maximum.
    zmax = float(np.max(pae))
    # Token t occupies [t-0.5, t+0.5]; label cell edges 0, n/4, ..., n.
    ticks = sorted({round(n * q / 4) for q in range(5)})
    axis = dict(tickvals=[t - 0.5 for t in ticks], ticktext=[str(t) for t in ticks],
                range=[-0.5, n - 0.5], constrain="domain", showgrid=False,
                ticks="outside", zeroline=False)
    fig = go.Figure(
        go.Heatmap(
            z=pae,
            zmin=0.0,
            zmax=zmax,
            colorscale=PAE_COLORSCALE,
            hovertemplate="Aligned token %{y}<br>Scored token %{x}<br>PAE %{z:.2f} Å<extra></extra>",
            colorbar=dict(
                orientation="h", x=0.5, xanchor="center", y=-0.3, yanchor="top",
                len=0.8, thickness=16, outlinecolor="#374151", outlinewidth=1,
                tickvals=[0, zmax], ticktext=["0", f"{zmax:.2f}"],
                title=dict(text="Expected Position Error (Å)", side="bottom"),
            ),
        )
    )
    fig.update_xaxes(title="Scored token", **axis)
    # Reversed y range puts token 0 at the top-left, as on Chai's result page.
    fig.update_yaxes(title="Aligned token", scaleanchor="x",
                     **{**axis, "range": [n - 0.5, -0.5]})
    fig.update_layout(
        title=dict(text="<b>Predicted Aligned Error</b>", x=0.5, xanchor="center"),
        height=560, margin=dict(l=60, r=20, t=60, b=130),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def _render_downloads(store: JobStore, job: dict, cand: Candidate, has_confidence: bool) -> None:
    st.subheader("Downloads")
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Download selected CIF", cand.cif_path.read_bytes(),
                       file_name=cand.cif_path.name, mime="chemical/x-cif",
                       icon=":material/download:", width="stretch")
    d2.download_button("Download score file", cand.scores_path.read_bytes(),
                       file_name=cand.scores_path.name, mime="application/octet-stream",
                       icon=":material/download:", width="stretch")
    if has_confidence:
        d3.download_button("Download PAE / pLDDT data", cand.confidence_path.read_bytes(),
                           file_name=cand.confidence_path.name, mime="application/octet-stream",
                           icon=":material/download:", width="stretch")
    else:
        d3.button("PAE data not saved", disabled=True, width="stretch")

    zip_key = f"zip_{job['job_id']}"
    if zip_key in ss:
        d4.download_button("Download complete results ZIP", ss[zip_key],
                           file_name=f"{_safe_name(job['name'])}_{job['job_id']}.zip",
                           mime="application/zip", icon=":material/folder_zip:",
                           type="primary", width="stretch")
    elif d4.button("Prepare complete results ZIP", icon=":material/folder_zip:",
                   width="stretch"):
        with st.spinner("Building ZIP ..."):
            ss[zip_key] = _results_zip(store, job["job_id"])
        st.rerun()


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:60] or "prediction"


def _results_zip(store: JobStore, job_id: str) -> bytes:
    job_dir = store.job_dir(job_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ("config.json", "input.fasta", "restraints.csv"):
            if (job_dir / name).is_file():
                zf.write(job_dir / name, f"inputs/{name}")
        out_dir = store.output_dir(job_id)
        for path in output_files(out_dir):
            zf.write(path, f"outputs/{path.relative_to(out_dir).as_posix()}")
    return buf.getvalue()


def _render_inputs(config: dict, seq_rows: list[dict]) -> None:
    st.subheader("Input sequences")
    st.markdown(
        _html_table(
            ["Molecule type", "Copies", "Chain IDs", "Length", "Token offset", "Sequence"],
            [
                [r["Molecule type"], r["Copies"], r["Chain IDs"],
                 "—" if r["Length"] is None else r["Length"],
                 "—" if r["Token offset"] is None
                 else f'<span style="white-space:nowrap">{r["Token offset"]}</span>',
                 f'<span class="cz-seq">{html.escape(r["Sequence"])}</span>']
                for r in seq_rows
            ],
            raw_cols={4, 5},
        ),
        unsafe_allow_html=True,
    )
    if any(r["Token offset"] is None for r in seq_rows):
        st.caption(
            "Token offsets are shown only up to the first ligand or modified residue: "
            "Chai tokenizes those per atom, which cannot be counted from the sequence alone."
        )

    restraint_rows = input_restraint_rows(config)
    if restraint_rows:
        st.subheader("Input restraints")
        cols = list(restraint_rows[0])
        st.markdown(
            _html_table(cols, [[f"{r[c]:g}" if c == "Distance (Å)" else r[c] for c in cols]
                               for r in restraint_rows]),
            unsafe_allow_html=True,
        )


# ===========================================================================
# Navigation
# ===========================================================================
# Emoji page icons: Streamlit fetches ":material/...:" *navigation* icons as
# SVGs from fonts.gstatic.com, which an offline browser cannot reach (button
# and markdown material icons use the bundled font and are fine).
PAGES = {
    "predict": st.Page(predict_page, title="Predict new structure", icon="➕", default=True),
    "predictions": st.Page(predictions_page, title="Predictions", icon="📋",
                           url_path="predictions"),
}

if OFFLINE:
    st.sidebar.caption(":material/cloud_off: Offline mode: no external servers are used.")

st.navigation(list(PAGES.values())).run()
