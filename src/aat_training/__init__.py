"""Reproducible AAT lane-classification training utilities."""

from .folds import FoldBuildResult, FoldFeasibilityError, build_fold_artifacts, build_nested_folds, validate_group_disjointness
from .classical import ClassicalRunResult, fit_profile_pipeline, make_classifier, profile_feature_vector, run_classical_nested_cv
from .experiments import ExperimentRun, complete_experiment, create_experiment
from .labels import LabelDecision, LabelPolicy, decide_label, load_label_policy
from .metrics import cluster_bootstrap_ci, evaluate_alleles, evaluate_common, evaluate_referral, evaluate_retrieval
from .predictions import read_prediction_rows, validate_prediction_rows, write_prediction_rows
from .preprocessing import InputBuildResult, LetterboxMetadata, build_training_inputs, crop_lane, extract_intensity_profile, letterbox_rgb
from .snapshot import SnapshotBuildResult, SnapshotValidationError, build_snapshot, load_snapshot_lanes

__all__ = [
    "LabelDecision",
    "LabelPolicy",
    "InputBuildResult",
    "LetterboxMetadata",
    "FoldBuildResult",
    "FoldFeasibilityError",
    "ClassicalRunResult",
    "ExperimentRun",
    "SnapshotBuildResult",
    "SnapshotValidationError",
    "build_snapshot",
    "build_fold_artifacts",
    "build_nested_folds",
    "build_training_inputs",
    "cluster_bootstrap_ci",
    "complete_experiment",
    "create_experiment",
    "crop_lane",
    "decide_label",
    "evaluate_alleles",
    "evaluate_common",
    "evaluate_referral",
    "evaluate_retrieval",
    "fit_profile_pipeline",
    "load_label_policy",
    "load_snapshot_lanes",
    "read_prediction_rows",
    "extract_intensity_profile",
    "letterbox_rgb",
    "make_classifier",
    "profile_feature_vector",
    "validate_group_disjointness",
    "validate_prediction_rows",
    "write_prediction_rows",
    "run_classical_nested_cv",
]
