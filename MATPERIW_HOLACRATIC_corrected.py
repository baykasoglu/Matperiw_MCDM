# ============================================================
# HOLACRATIC MATPERIW-MCDM
# Matrix Permanent-Based Integrated Weighting Method
# for Holacratic / Hierarchical MCDM Problems
#
# Reads:
#   - holacratic structure from an Excel sheet
#   - pairwise comparison matrices from separate Excel tabs
#   - decision matrices and criterion types from Excel tabs
#
# Important rule:
#   Objective entropy weights and normalization are applied only to raw/direct
#   criteria read from DM_<Holon> sheets. Propagated permanent scores from child
#   holons are carried upward directly; they are not normalized or entropy-weighted again.
#
# Author idea: Prof. Dr. Adil Baykasoğlu
# Extended version: holacratic MATPERIW-MCDM implementation
# 15.05.2026
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import Matrix

from crispyn import weighting_methods as mcda_weights
from pyDecision.algorithm import ahp_method


# ============================================================
# BASIC FLAT MATPERIW FUNCTIONS
# ============================================================

def compute_ahp_weights(
    pairwise_matrix: np.ndarray,
    weight_derivation: str = "max_eigen",
    cr_threshold: float = 0.10,
) -> Tuple[np.ndarray, float, bool]:
    """Compute subjective AHP weights and consistency ratio."""
    pairwise_matrix = np.asarray(pairwise_matrix, dtype=float)
    weights, cr = ahp_method(pairwise_matrix, wd=weight_derivation)
    weights = np.asarray(weights, dtype=float)
    cr = float(cr)
    return weights, cr, cr <= cr_threshold


def compute_entropy_weights(decision_matrix: np.ndarray) -> np.ndarray:
    """Compute objective weights using the entropy weighting method."""
    decision_matrix = np.asarray(decision_matrix, dtype=float)
    objective_weights = mcda_weights.entropy_weighting(decision_matrix)
    return np.asarray(objective_weights, dtype=float)


def normalize_decision_matrix(decision_matrix: np.ndarray, criterion_types: np.ndarray) -> np.ndarray:
    """Normalize decision matrix according to benefit/cost criterion types."""
    decision_matrix = np.asarray(decision_matrix, dtype=float)
    criterion_types = np.asarray(criterion_types, dtype=int)
    normalized_matrix = mcda_weights.sum_normalization(decision_matrix, criterion_types)
    return np.asarray(normalized_matrix, dtype=float)


def build_interaction_matrix(subjective_weights: np.ndarray) -> np.ndarray:
    """
    Construct AHP-based interaction matrix beta.

    beta_jk = w_j / (w_j + w_k)
    beta_kj = 1 - beta_jk
    beta_jj = 1
    """
    subjective_weights = np.asarray(subjective_weights, dtype=float)
    n_criteria = len(subjective_weights)
    interaction_matrix = np.zeros((n_criteria, n_criteria), dtype=float)
    np.fill_diagonal(interaction_matrix, 1.0)

    for j in range(n_criteria):
        for k in range(j + 1, n_criteria):
            denom = subjective_weights[j] + subjective_weights[k]
            if denom <= 0:
                raise ValueError("AHP weights must be positive to build interaction matrix.")
            interaction_matrix[j, k] = subjective_weights[j] / denom
            interaction_matrix[k, j] = 1.0 - interaction_matrix[j, k]

    return interaction_matrix


def compute_permanent_scores(
    diagonal_score_matrix: np.ndarray,
    interaction_matrix: np.ndarray,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Compute permanent scores for all alternatives.

    IMPORTANT FOR HOLACRATIC MATPERIW-MCDM
    --------------------------------------
    The values placed on the diagonal must already be prepared before this
    function is called.

    For a raw/direct criterion:
        diagonal value = entropy_weight_j * normalized_value_ij

    For a propagated child-holon criterion:
        diagonal value = carried permanent score, without re-normalization
                         and without entropy re-weighting.

    Off-diagonal values are the AHP-based interaction matrix.
    """
    diagonal_score_matrix = np.asarray(diagonal_score_matrix, dtype=float)
    interaction_matrix = np.asarray(interaction_matrix, dtype=float)

    n_alternatives, n_criteria = diagonal_score_matrix.shape

    if interaction_matrix.shape != (n_criteria, n_criteria):
        raise ValueError("interaction_matrix must be n_criteria x n_criteria.")

    permanent_scores = np.zeros(n_alternatives, dtype=float)
    alternative_matrices: List[np.ndarray] = []

    for i in range(n_alternatives):
        alternative_matrix = interaction_matrix.copy()
        np.fill_diagonal(alternative_matrix, diagonal_score_matrix[i, :])
        alternative_matrices.append(alternative_matrix.copy())
        permanent_scores[i] = float(Matrix(alternative_matrix).per())

    return permanent_scores, alternative_matrices


def create_ranking_table(permanent_scores: np.ndarray, alternative_names: List[str]) -> pd.DataFrame:
    """Create ranking table sorted descending by permanent score."""
    ranking_table = pd.DataFrame(
        {
            "Alternative": alternative_names,
            "Permanent Score": np.asarray(permanent_scores, dtype=float),
        }
    )
    ranking_table = ranking_table.sort_values("Permanent Score", ascending=False).reset_index(drop=True)
    ranking_table["Rank"] = np.arange(1, len(ranking_table) + 1)
    return ranking_table[["Rank", "Alternative", "Permanent Score"]]


def evaluate_single_holon(
    decision_matrix: pd.DataFrame,
    pairwise_matrix: np.ndarray,
    criterion_types: List[int],
    criterion_is_raw: Optional[List[bool]] = None,
    alternative_names: Optional[List[str]] = None,
    criterion_names: Optional[List[str]] = None,
    ahp_weight_derivation: str = "max_eigen",
    cr_threshold: float = 0.10,
) -> Dict[str, Any]:
    """
    Evaluate one holon using MATPERIW-MCDM.

    This version distinguishes two criterion sources:

    1. Raw/direct criteria:
       - read from DM_<Holon> sheets,
       - normalized according to criterion type,
       - objective entropy weights are computed only for these raw columns,
       - diagonal value = entropy_weight * normalized_value.

    2. Propagated child-holon criteria:
       - obtained as permanent scores from lower-level holons,
       - used directly as carried scores,
       - not normalized again,
       - not entropy-weighted again,
       - diagonal value = carried permanent score.

    Therefore, objective weights are applied only where corresponding raw data
    are entered. Once a lower holon's permanent score is carried upward, it is
    not re-weighted objectively.
    """
    if isinstance(decision_matrix, pd.DataFrame):
        local_df = decision_matrix.copy()
        if criterion_names is None:
            criterion_names = list(local_df.columns)
        if alternative_names is None:
            alternative_names = list(local_df.index.astype(str))
        X = local_df.to_numpy(dtype=float)
    else:
        X = np.asarray(decision_matrix, dtype=float)
        n_alt, n_crit = X.shape
        if alternative_names is None:
            alternative_names = [f"A{i + 1}" for i in range(n_alt)]
        if criterion_names is None:
            criterion_names = [f"C{j + 1}" for j in range(n_crit)]
        local_df = pd.DataFrame(X, index=alternative_names, columns=criterion_names)

    criterion_types = np.asarray(criterion_types, dtype=int)
    pairwise_matrix = np.asarray(pairwise_matrix, dtype=float)

    n_alt, n_crit = X.shape
    if pairwise_matrix.shape != (n_crit, n_crit):
        raise ValueError(
            f"Pairwise matrix must be {n_crit}x{n_crit}; got {pairwise_matrix.shape}."
        )
    if len(criterion_types) != n_crit:
        raise ValueError(
            f"criterion_types must contain {n_crit} values; got {len(criterion_types)}."
        )

    if criterion_is_raw is None:
        # Backward-compatible flat behavior: all criteria are raw/direct criteria.
        criterion_is_raw = [True] * n_crit
    if len(criterion_is_raw) != n_crit:
        raise ValueError(
            f"criterion_is_raw must contain {n_crit} values; got {len(criterion_is_raw)}."
        )

    criterion_is_raw_arr = np.asarray(criterion_is_raw, dtype=bool)
    raw_idx = np.where(criterion_is_raw_arr)[0]
    propagated_idx = np.where(~criterion_is_raw_arr)[0]

    subjective_weights, cr, is_consistent = compute_ahp_weights(
        pairwise_matrix, weight_derivation=ahp_weight_derivation, cr_threshold=cr_threshold
    )
    interaction_matrix = build_interaction_matrix(subjective_weights)

    # Matrix used for reporting: raw columns are normalized; propagated columns are carried as-is.
    normalized_or_carried_matrix = X.copy().astype(float)

    # Matrix whose rows are placed directly on the diagonal of each alternative matrix.
    diagonal_score_matrix = np.zeros_like(X, dtype=float)

    # Entropy weights are meaningful only for raw/direct criteria.
    objective_weights_entropy = np.full(n_crit, np.nan, dtype=float)

    # Effective objective multiplier used on the diagonal.
    # Raw criteria use entropy weights; propagated criteria use 1.0.
    effective_objective_weights = np.ones(n_crit, dtype=float)

    if len(raw_idx) > 0:
        raw_X = X[:, raw_idx]
        raw_types = criterion_types[raw_idx]
        raw_normalized = normalize_decision_matrix(raw_X, raw_types)
        raw_entropy_weights = compute_entropy_weights(raw_X)

        normalized_or_carried_matrix[:, raw_idx] = raw_normalized
        objective_weights_entropy[raw_idx] = raw_entropy_weights
        effective_objective_weights[raw_idx] = raw_entropy_weights
        diagonal_score_matrix[:, raw_idx] = raw_normalized * raw_entropy_weights

    if len(propagated_idx) > 0:
        # Carried lower-level permanent scores are already integrated scores.
        # They are not normalized and not entropy-weighted again.
        normalized_or_carried_matrix[:, propagated_idx] = X[:, propagated_idx]
        diagonal_score_matrix[:, propagated_idx] = X[:, propagated_idx]
        effective_objective_weights[propagated_idx] = 1.0

    permanent_scores, alternative_matrices = compute_permanent_scores(
        diagonal_score_matrix, interaction_matrix
    )
    ranking_table = create_ranking_table(permanent_scores, alternative_names)

    criterion_source = ["raw/direct" if is_raw else "propagated/permanent" for is_raw in criterion_is_raw]

    return {
        "decision_matrix": local_df,
        "criterion_types": pd.Series(criterion_types, index=criterion_names, name="Type"),
        "criterion_source": pd.Series(criterion_source, index=criterion_names, name="Source"),
        "criterion_is_raw": pd.Series(criterion_is_raw_arr, index=criterion_names, name="Is Raw"),
        "subjective_weights_ahp": pd.Series(subjective_weights, index=criterion_names, name="AHP Weight"),
        "ahp_consistency_ratio": cr,
        "ahp_is_consistent": is_consistent,
        "objective_weights_entropy": pd.Series(objective_weights_entropy, index=criterion_names, name="Entropy Weight"),
        "effective_objective_multiplier": pd.Series(effective_objective_weights, index=criterion_names, name="Effective Objective Multiplier"),
        "normalized_matrix": pd.DataFrame(normalized_or_carried_matrix, index=alternative_names, columns=criterion_names),
        "diagonal_score_matrix": pd.DataFrame(diagonal_score_matrix, index=alternative_names, columns=criterion_names),
        "interaction_matrix": pd.DataFrame(interaction_matrix, index=criterion_names, columns=criterion_names),
        "alternative_matrices": alternative_matrices,
        "permanent_scores": pd.Series(permanent_scores, index=alternative_names, name="Permanent Score"),
        "ranking_table": ranking_table,
    }


# ============================================================
# EXCEL INPUT HELPERS
# ============================================================

@dataclass
class HolonNode:
    """Node of the holacratic MCDM tree."""
    name: str
    parent: Optional[str]
    criteria: List[str]
    children: List[str] = field(default_factory=list)


class MatperiwExcelReader:
    """
    Reads a holacratic MATPERIW-MCDM model from Excel.

    Expected workbook structure
    ---------------------------
    Sheet: Structure
        Columns: Holon, Parent, Criteria
        Example:
            Holon   Parent   Criteria
            Main             C1,C2,C3
            C1      Main     C11,C12
            C2      Main     C21,C22
            C3      Main     C31,C32,C33

    Sheet: Alternatives  [optional]
        Column: Alternative
        Example: A1, A2, A3, ...

    Sheet: CriterionTypes  [optional]
        Columns: Holon, Criterion, Type
        Type: 1 for benefit, -1 for cost
        If missing, all criteria are assumed benefit type.

    Sheets: AHP_<HolonName>
        AHP_Main, AHP_C1, AHP_C2, ...
        Can be either:
            - square matrix with no labels, or
            - square matrix with row/column labels.

    Sheets: DM_<HolonName>
        Decision matrix for direct/local criteria of a holon.
        First column should be Alternative. Other columns are criterion names.
        Leaf holons normally need DM sheets.
        Parent holons do not need DM sheets if all their criteria are child holons.
    """

    def __init__(self, excel_path: str | Path):
        self.excel_path = Path(excel_path)
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {self.excel_path}")
        self.xlsx = pd.ExcelFile(self.excel_path)

    def sheet_exists(self, sheet_name: str) -> bool:
        return sheet_name in self.xlsx.sheet_names

    @staticmethod
    def _clean_name(x: Any) -> str:
        return str(x).strip()

    @staticmethod
    def _split_criteria(value: Any) -> List[str]:
        if pd.isna(value):
            return []
        if isinstance(value, str):
            value = value.replace(";", ",")
            return [v.strip() for v in value.split(",") if v.strip()]
        return [str(value).strip()]

    def read_structure(self, sheet_name: str = "Structure") -> Tuple[Dict[str, HolonNode], str]:
        if not self.sheet_exists(sheet_name):
            raise ValueError(f"Required sheet '{sheet_name}' was not found.")

        df = pd.read_excel(self.xlsx, sheet_name=sheet_name)
        required = {"Holon", "Parent", "Criteria"}
        if not required.issubset(df.columns):
            raise ValueError(f"Sheet '{sheet_name}' must contain columns: {sorted(required)}")

        nodes: Dict[str, HolonNode] = {}
        for _, row in df.iterrows():
            holon = self._clean_name(row["Holon"])
            parent_raw = row["Parent"]
            parent = None if pd.isna(parent_raw) or str(parent_raw).strip() == "" else self._clean_name(parent_raw)
            criteria = self._split_criteria(row["Criteria"])
            nodes[holon] = HolonNode(name=holon, parent=parent, criteria=criteria)

        root_candidates = [name for name, node in nodes.items() if node.parent is None]
        if len(root_candidates) != 1:
            raise ValueError(
                "Exactly one root holon must have empty Parent in the Structure sheet. "
                f"Found: {root_candidates}"
            )
        root = root_candidates[0]

        for name, node in nodes.items():
            if node.parent is not None:
                if node.parent not in nodes:
                    raise ValueError(f"Holon '{name}' has parent '{node.parent}', but parent is not defined.")
                nodes[node.parent].children.append(name)

        return nodes, root

    def read_alternatives(self, sheet_name: str = "Alternatives") -> Optional[List[str]]:
        if not self.sheet_exists(sheet_name):
            return None
        df = pd.read_excel(self.xlsx, sheet_name=sheet_name)
        if "Alternative" not in df.columns:
            raise ValueError(f"Sheet '{sheet_name}' must contain an 'Alternative' column.")
        return [self._clean_name(x) for x in df["Alternative"].dropna().tolist()]

    def read_criterion_types(self, sheet_name: str = "CriterionTypes") -> Dict[Tuple[str, str], int]:
        if not self.sheet_exists(sheet_name):
            return {}
        df = pd.read_excel(self.xlsx, sheet_name=sheet_name)
        required = {"Holon", "Criterion", "Type"}
        if not required.issubset(df.columns):
            raise ValueError(f"Sheet '{sheet_name}' must contain columns: {sorted(required)}")

        types: Dict[Tuple[str, str], int] = {}
        for _, row in df.iterrows():
            holon = self._clean_name(row["Holon"])
            criterion = self._clean_name(row["Criterion"])
            ctype = int(row["Type"])
            if ctype not in (-1, 1):
                raise ValueError(f"Criterion type must be 1 or -1. Got {ctype} for {holon}/{criterion}.")
            types[(holon, criterion)] = ctype
        return types

    def read_decision_matrix(self, holon_name: str) -> Optional[pd.DataFrame]:
        sheet_name = f"DM_{holon_name}"
        if not self.sheet_exists(sheet_name):
            return None
        df = pd.read_excel(self.xlsx, sheet_name=sheet_name)
        if df.empty:
            return None

        first_col = df.columns[0]
        if str(first_col).strip().lower() != "alternative":
            # still allow the first column to be used as alternative names
            pass
        df = df.copy()
        df[first_col] = df[first_col].astype(str).str.strip()
        df = df.set_index(first_col)
        df.index.name = "Alternative"
        df.columns = [str(c).strip() for c in df.columns]
        return df.apply(pd.to_numeric, errors="raise")

    def read_pairwise_matrix(self, holon_name: str, criteria_order: List[str]) -> np.ndarray:
        sheet_name = f"AHP_{holon_name}"
        if not self.sheet_exists(sheet_name):
            raise ValueError(f"Required pairwise comparison sheet '{sheet_name}' was not found.")

        raw = pd.read_excel(self.xlsx, sheet_name=sheet_name, header=None)
        raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
        n = len(criteria_order)

        # Case 1: raw n x n numeric matrix without labels
        numeric_raw = raw.apply(pd.to_numeric, errors="coerce")
        if raw.shape[0] >= n and raw.shape[1] >= n:
            candidate = numeric_raw.iloc[:n, :n]
            if candidate.notna().all().all():
                return candidate.to_numpy(dtype=float)

        # Case 2: labels in first row and first column
        df = pd.read_excel(self.xlsx, sheet_name=sheet_name, index_col=0)
        df.index = [str(x).strip() for x in df.index]
        df.columns = [str(x).strip() for x in df.columns]
        missing_rows = [c for c in criteria_order if c not in df.index]
        missing_cols = [c for c in criteria_order if c not in df.columns]
        if missing_rows or missing_cols:
            raise ValueError(
                f"AHP sheet '{sheet_name}' must contain row/column labels matching criteria order. "
                f"Missing rows: {missing_rows}; missing columns: {missing_cols}"
            )
        matrix = df.loc[criteria_order, criteria_order].to_numpy(dtype=float)
        return matrix


# ============================================================
# HOLACRATIC MATPERIW-MCDM MODEL
# ============================================================

class HolacraticMatperiwMCDM:
    """
    Recursive holacratic MATPERIW-MCDM model.

    The procedure evaluates lower-level holons first. The permanent score of a lower-level
    holon becomes the criterion value of the corresponding upper-level criterion.
    """

    def __init__(
        self,
        excel_path: str | Path,
        structure_sheet: str = "Structure",
        alternatives_sheet: str = "Alternatives",
        criterion_types_sheet: str = "CriterionTypes",
        ahp_weight_derivation: str = "max_eigen",
        cr_threshold: float = 0.10,
        default_criterion_type: int = 1,
    ):
        self.reader = MatperiwExcelReader(excel_path)
        self.nodes, self.root_holon = self.reader.read_structure(structure_sheet)
        self.alternatives = self.reader.read_alternatives(alternatives_sheet)
        self.criterion_types_map = self.reader.read_criterion_types(criterion_types_sheet)
        self.ahp_weight_derivation = ahp_weight_derivation
        self.cr_threshold = cr_threshold
        self.default_criterion_type = default_criterion_type
        self.results: Dict[str, Dict[str, Any]] = {}

    def _is_child_holon_of(self, criterion_name: str, parent_holon: str) -> bool:
        return criterion_name in self.nodes and self.nodes[criterion_name].parent == parent_holon

    def _align_series_to_alternatives(self, s: pd.Series, holon_name: str) -> pd.Series:
        s = s.copy()
        s.index = s.index.astype(str)
        if self.alternatives is None:
            return s
        missing = [a for a in self.alternatives if a not in s.index]
        if missing:
            raise ValueError(f"Holon '{holon_name}' is missing alternatives: {missing}")
        return s.loc[self.alternatives]

    def _get_types_for_holon(self, holon_name: str, criteria: List[str]) -> List[int]:
        types: List[int] = []
        for c in criteria:
            types.append(int(self.criterion_types_map.get((holon_name, c), self.default_criterion_type)))
        return types

    def _build_local_decision_matrix(self, holon_name: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Build the local decision matrix of a holon.

        Returns
        -------
        local_df : pd.DataFrame
            Matrix with all local criteria in the order defined in Structure.
            Direct/raw criteria are read from DM_<Holon>. Child-holon criteria
            are filled with permanent scores computed recursively.
        column_sources : dict
            criterion -> "raw/direct" or "propagated/permanent".
        """
        node = self.nodes[holon_name]
        criteria = node.criteria

        direct_dm = self.reader.read_decision_matrix(holon_name)
        if direct_dm is not None:
            direct_dm.index = direct_dm.index.astype(str)
            if self.alternatives is not None:
                missing_alt = [a for a in self.alternatives if a not in direct_dm.index]
                if missing_alt:
                    raise ValueError(f"DM_{holon_name} is missing alternatives: {missing_alt}")
                direct_dm = direct_dm.loc[self.alternatives]

        local_columns: Dict[str, pd.Series] = {}
        column_sources: Dict[str, str] = {}
        inferred_alternatives: Optional[List[str]] = self.alternatives

        for criterion in criteria:
            if self._is_child_holon_of(criterion, holon_name):
                child_result = self.evaluate_holon(criterion)
                local_columns[criterion] = self._align_series_to_alternatives(
                    child_result["permanent_scores"], criterion
                )
                column_sources[criterion] = "propagated/permanent"
                if inferred_alternatives is None:
                    inferred_alternatives = list(local_columns[criterion].index.astype(str))
            else:
                if direct_dm is None or criterion not in direct_dm.columns:
                    raise ValueError(
                        f"Criterion '{criterion}' in holon '{holon_name}' is not a child holon and "
                        f"was not found in sheet 'DM_{holon_name}'."
                    )
                col = direct_dm[criterion].astype(float)
                if inferred_alternatives is None:
                    inferred_alternatives = list(col.index.astype(str))
                local_columns[criterion] = col
                column_sources[criterion] = "raw/direct"

        if inferred_alternatives is None:
            raise ValueError(f"Cannot infer alternatives for holon '{holon_name}'.")

        local_df = pd.DataFrame(local_columns)
        local_df.index = local_df.index.astype(str)
        local_df = local_df.loc[inferred_alternatives, criteria]
        local_df.index.name = "Alternative"
        return local_df, column_sources

    def evaluate_holon(self, holon_name: str) -> Dict[str, Any]:
        """Evaluate a holon recursively. Lower-level holons are evaluated first."""
        if holon_name in self.results:
            return self.results[holon_name]

        if holon_name not in self.nodes:
            raise ValueError(f"Holon '{holon_name}' is not defined in the Structure sheet.")

        node = self.nodes[holon_name]
        if not node.criteria:
            raise ValueError(f"Holon '{holon_name}' has no criteria in the Structure sheet.")

        local_df, column_sources = self._build_local_decision_matrix(holon_name)
        criteria_order = list(local_df.columns)
        alternatives = list(local_df.index.astype(str))
        criterion_types = self._get_types_for_holon(holon_name, criteria_order)
        pairwise_matrix = self.reader.read_pairwise_matrix(holon_name, criteria_order)
        criterion_is_raw = [column_sources[c] == "raw/direct" for c in criteria_order]

        result = evaluate_single_holon(
            decision_matrix=local_df,
            pairwise_matrix=pairwise_matrix,
            criterion_types=criterion_types,
            criterion_is_raw=criterion_is_raw,
            alternative_names=alternatives,
            criterion_names=criteria_order,
            ahp_weight_derivation=self.ahp_weight_derivation,
            cr_threshold=self.cr_threshold,
        )
        result["holon_name"] = holon_name
        result["parent"] = node.parent
        result["criteria_order"] = criteria_order
        result["children"] = node.children
        result["column_sources"] = column_sources
        result["raw_criteria"] = [c for c in criteria_order if column_sources[c] == "raw/direct"]
        result["propagated_criteria"] = [c for c in criteria_order if column_sources[c] == "propagated/permanent"]

        self.results[holon_name] = result
        return result

    def run(self) -> Dict[str, Any]:
        """Run the full recursive model and return all results."""
        final_result = self.evaluate_holon(self.root_holon)
        return {
            "root_holon": self.root_holon,
            "final_result": final_result,
            "all_holon_results": self.results,
        }

    def export_results(self, output_path: str | Path) -> None:
        """Export all holon results to an Excel workbook."""
        output_path = Path(output_path)
        if not self.results:
            self.run()

        def safe_sheet_name(name: str) -> str:
            invalid = ["\\", "/", "?", "*", "[", "]", ":"]
            clean = name
            for ch in invalid:
                clean = clean.replace(ch, "_")
            return clean[:31]

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Summary sheet
            summary_rows = []
            for holon_name, res in self.results.items():
                summary_rows.append(
                    {
                        "Holon": holon_name,
                        "Parent": res.get("parent"),
                        "Criteria": ", ".join(res["criteria_order"]),
                        "Raw/Direct Criteria": ", ".join(res.get("raw_criteria", [])),
                        "Propagated Criteria": ", ".join(res.get("propagated_criteria", [])),
                        "AHP CR": res["ahp_consistency_ratio"],
                        "AHP Consistent": res["ahp_is_consistent"],
                    }
                )
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

            # Final ranking
            self.results[self.root_holon]["ranking_table"].to_excel(
                writer, sheet_name="Final_Ranking", index=False
            )

            # Per-holon details
            for holon_name, res in self.results.items():
                prefix = safe_sheet_name(holon_name)
                res["ranking_table"].to_excel(writer, sheet_name=safe_sheet_name(f"R_{prefix}"), index=False)
                res["decision_matrix"].to_excel(writer, sheet_name=safe_sheet_name(f"DM_{prefix}"))
                res["normalized_matrix"].to_excel(writer, sheet_name=safe_sheet_name(f"N_{prefix}"))
                res["diagonal_score_matrix"].to_excel(writer, sheet_name=safe_sheet_name(f"Diag_{prefix}"))
                res["interaction_matrix"].to_excel(writer, sheet_name=safe_sheet_name(f"Beta_{prefix}"))

                weights_df = pd.DataFrame(
                    {
                        "Criterion": res["criteria_order"],
                        "Source": res["criterion_source"].values,
                        "Type": res["criterion_types"].values,
                        "AHP Subjective Weight": res["subjective_weights_ahp"].values,
                        "Entropy Objective Weight": res["objective_weights_entropy"].values,
                        "Effective Objective Multiplier": res["effective_objective_multiplier"].values,
                    }
                )
                weights_df.to_excel(writer, sheet_name=safe_sheet_name(f"W_{prefix}"), index=False)

    def print_final_results(self) -> None:
        """Print concise final ranking and CR summary."""
        if not self.results:
            self.run()
        final = self.results[self.root_holon]
        print("\n" + "=" * 80)
        print(f"HOLACRATIC MATPERIW-MCDM FINAL RESULT | ROOT HOLON: {self.root_holon}")
        print("=" * 80)
        print(final["ranking_table"].to_string(index=False))
        print("\nAHP consistency summary:")
        for holon_name, res in self.results.items():
            status = "OK" if res["ahp_is_consistent"] else "REVIEW"
            print(f"  {holon_name}: CR={res['ahp_consistency_ratio']:.6f} | {status}")


def plot_final_ranking(final_ranking: pd.DataFrame, title: str = "Holacratic MATPERIW-MCDM Final Ranking") -> None:
    """Plot final ranking bar chart."""
    plt.figure(figsize=(9, 5))
    alternatives = final_ranking["Alternative"]
    scores = final_ranking["Permanent Score"]
    bars = plt.bar(alternatives, scores)
    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.title(title)
    plt.xlabel("Alternatives")
    plt.ylabel("Permanent Score")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Create an Excel workbook with the structure explained below, then run:
    #
    # model = HolacraticMatperiwMCDM("matperiw_input.xlsx")
    # output = model.run()
    # model.print_final_results()
    # model.export_results("matperiw_holacratic_results.xlsx")
    # plot_final_ranking(output["final_result"]["ranking_table"])
    #
    # Expected Excel sheets:
    #   Structure
    #   Alternatives                       optional
    #   CriterionTypes                     optional
    #   AHP_Main, AHP_C1, AHP_C2, ...
    #   DM_C1, DM_C2, DM_C3, ...           for leaf holons
    #
    # Example Structure sheet:
    #   Holon | Parent | Criteria
    #   Main  |        | C1,C2,C3
    #   C1    | Main   | C11,C12
    #   C2    | Main   | C21,C22
    #   C3    | Main   | C31,C32,C33  
    pass