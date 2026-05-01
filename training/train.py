"""LambdaMART model training using LightGBM."""

import lightgbm as lgb
import mlflow
import mlflow.lightgbm

from training.dataset import FEATURE_COLS, load_training_data, get_groups, train_test_split_by_request


DEFAULT_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [3, 5],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "verbose": -1,
}


def train(
    data_path: str = "data/training_set.parquet",
    params: dict | None = None,
    num_boost_round: int = 300,
    mlflow_tracking_uri: str = "http://localhost:5000",
    run_name: str = "lambdamart_v1",
) -> lgb.Booster:
    """Train a LambdaMART ranking model.

    Args:
        data_path: Path to parquet training data.
        params: LightGBM params (uses defaults if None).
        num_boost_round: Number of boosting rounds.
        mlflow_tracking_uri: MLflow server URI.
        run_name: Name for the MLflow run.

    Returns:
        Trained LightGBM Booster.
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    # Load & split
    df = load_training_data(data_path)
    train_df, val_df = train_test_split_by_request(df, test_fraction=0.2)

    train_groups = get_groups(train_df)
    val_groups = get_groups(val_df)

    train_dataset = lgb.Dataset(
        train_df[FEATURE_COLS], label=train_df["label"], group=train_groups
    )
    val_dataset = lgb.Dataset(
        val_df[FEATURE_COLS], label=val_df["label"], group=val_groups, reference=train_dataset
    )

    # Train with MLflow tracking
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    with mlflow.start_run(run_name=run_name):
        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=num_boost_round,
            valid_sets=[val_dataset],
            valid_names=["validation"],
        )

        # Log
        mlflow.log_params(params)
        mlflow.log_metric("num_boost_round", num_boost_round)
        mlflow.lightgbm.log_model(model, "model")

        # Feature importance
        importance = dict(zip(FEATURE_COLS, model.feature_importance().tolist()))
        mlflow.log_dict(importance, "feature_importance.json")

    return model


def train_local(data_path: str = "data/training_set.parquet", num_boost_round: int = 100) -> lgb.Booster:
    """Train without MLflow (for quick local testing)."""
    df = load_training_data(data_path)
    train_df, val_df = train_test_split_by_request(df, test_fraction=0.2)

    train_groups = get_groups(train_df)
    val_groups = get_groups(val_df)

    train_dataset = lgb.Dataset(
        train_df[FEATURE_COLS], label=train_df["label"], group=train_groups
    )
    val_dataset = lgb.Dataset(
        val_df[FEATURE_COLS], label=val_df["label"], group=val_groups, reference=train_dataset
    )

    model = lgb.train(
        DEFAULT_PARAMS,
        train_dataset,
        num_boost_round=num_boost_round,
        valid_sets=[val_dataset],
        valid_names=["validation"],
    )
    return model
