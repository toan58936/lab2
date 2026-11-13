"""
going_modular — CIFAR-10 Transfer Learning helpers.

Import theo module (cách dùng trong notebook):
    from going_modular import data_setup, engine, evaluate, model_builder, utils

Hoặc flat import (13 symbols theo hợp đồng 00_lab2_description.md mục 2.8.6):
    from going_modular import (
        ModelConfig, build_model, count_params,
        set_train_mode, get_param_groups,
        create_train_val_test_dataloaders, train, evaluate_step,
        set_seed, get_device, run_inference, evaluate_predictions,
    )
"""

from .data_setup import (
    create_train_val_test_dataloaders,
    get_pretrained_transform,
    stratified_split_indices,
)

from .model_builder import (
    build_model,
    count_params,
    get_param_groups,
    set_train_mode,
    ModelConfig,
    ARCH_SPECS,
)

from .engine import (
    train,
    train_step,
    evaluate_step,
)

from .evaluate import (
    predict_probs,
    predict_dataloader,
    compute_confusion_matrix,
    per_class_accuracy,
    top_confusions,
    plot_history,
    plot_confusion_matrix,
    plot_sample_predictions,
    run_inference,
    classification_metrics,
    wilson_interval,
    misclassified_indices,
    get_mean_std,
    denormalize,
    measure_inference_time,
    file_size_mb,
    build_cost_record,
    evaluate_predictions,
    format_comparison_table,
    finalize_figure,
    InferenceResult,
)

from .utils import (
    set_seed,
    get_device,
    get_env_info,
    create_writer,
    save_model,
    load_model,
    save_config,
    save_history,
    load_experiment,
    save_test_metrics,
    load_test_metrics,
    artifacts_exist,
    save_checkpoint,
    load_checkpoint,
    make_session_id,
    make_eval_id,
    make_eval_output_dir,
    make_session_log_dir,
    file_sha256,
    ensure_compatible_state_dict,
)

__all__ = [
    # 13 symbols theo hợp đồng 00_lab2_description.md mục 2.8.6
    "ModelConfig",
    "build_model",
    "count_params",
    "set_train_mode",
    "get_param_groups",
    "create_train_val_test_dataloaders",
    "train",
    "evaluate_step",
    "set_seed",
    "get_device",
    "run_inference",
    "evaluate_predictions",
]
