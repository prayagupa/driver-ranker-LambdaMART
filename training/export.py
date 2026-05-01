"""Export trained LightGBM model to ONNX format for serving."""

from pathlib import Path


def export_to_onnx(model, n_features: int, output_path: str = "model/ranker.onnx"):
    """Export LightGBM model to ONNX format.

    Args:
        model: Trained LightGBM Booster.
        n_features: Number of input features.
        output_path: Where to save the ONNX file.
    """
    import onnxmltools
    from onnxconverter_common import FloatTensorType

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    initial_type = [("features", FloatTensorType([None, n_features]))]
    onnx_model = onnxmltools.convert_lightgbm(
        model, initial_types=initial_type, target_opset=12
    )
    onnxmltools.utils.save_model(onnx_model, output_path)
    print(f"Exported ONNX model to {output_path}")
    return output_path


def export_to_lightgbm_native(model, output_path: str = "model/ranker.txt"):
    """Export as LightGBM native text format (simpler, for fallback)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output_path)
    print(f"Exported LightGBM model to {output_path}")
    return output_path
