import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

import zonos2_tts_comfyui_test.loader as loader_module
from zonos2_tts_comfyui_test.loader import (
    ATTENTION_OPTIONS,
    BF16_REPO_ID,
    DTYPE_OPTIONS,
    FP8_REPO_ID,
    FP8_E4M3_FORMAT,
    FP8_E4M3_POLICY,
    LEGACY_FP8_E4M3_FORMAT,
    inspect_checkpoint_quantization,
    resolve_dtype,
    resolve_model_path,
    resolve_model_source,
    validate_quantized_checkpoint,
)
from zonos2_tts_comfyui_test.nodes import (
    QUALITY_BUCKET_LABELS,
    Zonos2VoiceClone,
    Zonos2VoiceGeneration,
)
from zonos2_tts_comfyui_test.runtime import (
    MAX_REFERENCE_SECONDS,
    Zonos2SpeakerEncoder,
    _require_or_download,
    logger,
)


def test_loader_option_order():
    assert DTYPE_OPTIONS == ["auto", "bf16", "fp16"]
    assert ATTENTION_OPTIONS == ["auto", "SDPA", "flash_attention"]


def test_model_catalog_routes_each_preset_to_its_own_repo():
    bf16 = resolve_model_source("ZONOS2 BF16 - drbaph/ZONOS2-BF16")
    fp8 = resolve_model_source("ZONOS2 FP8 Mixed - drbaph/ZONOS2-FP8")

    assert bf16.filename == "zonos2-bf16.safetensors"
    assert bf16.repo_id == BF16_REPO_ID
    assert fp8.filename == "zonos2-fp8-mixed.safetensors"
    assert fp8.repo_id == FP8_REPO_ID


def test_missing_selected_model_downloads_even_when_assets_exist(
    tmp_path,
    monkeypatch,
):
    for directory, filenames in {
        "dac_44khz": ("config.json", "model.safetensors"),
        "speaker_encoder": (
            "config.json",
            "model.safetensors",
            "configuration_ecapa_tdnn.py",
            "modeling_ecapa_tdnn.py",
        ),
    }.items():
        local_dir = tmp_path / directory
        local_dir.mkdir()
        for filename in filenames:
            (local_dir / filename).touch()

    calls = []

    def fake_download(filename, repo_id):
        calls.append((filename, repo_id))
        output = tmp_path / filename
        output.touch()
        return output

    monkeypatch.setattr(loader_module, "model_dir", lambda: tmp_path)
    monkeypatch.setattr(loader_module, "_download_model", fake_download)

    result = resolve_model_path(
        "ZONOS2 FP8 Mixed - drbaph/ZONOS2-FP8",
        download_if_missing=True,
    )

    assert result == tmp_path / "zonos2-fp8-mixed.safetensors"
    assert calls == [("zonos2-fp8-mixed.safetensors", FP8_REPO_ID)]


def test_complete_asset_folder_never_downloads(tmp_path):
    local_dir = tmp_path / "dac_44khz"
    local_dir.mkdir()
    (local_dir / "config.json").touch()
    (local_dir / "model.safetensors").touch()

    result = _require_or_download(
        "drbaph/does-not-need-to-exist",
        local_dir,
        allow_download=True,
        repo_subdir="dac_44khz",
        required_files=("config.json", "model.safetensors"),
    )

    assert result == local_dir


def test_mixed_fp8_metadata_and_dtype_policy(tmp_path):
    checkpoint = tmp_path / "zonos2-fp8-mixed.safetensors"
    save_file(
        {"weight": torch.ones(1, dtype=torch.bfloat16)},
        str(checkpoint),
        metadata={"zonos2_quantization": FP8_E4M3_FORMAT},
    )

    assert inspect_checkpoint_quantization(checkpoint) == "fp8_e4m3"
    assert resolve_dtype("auto", checkpoint, torch.device("cuda")) == torch.bfloat16
    assert resolve_dtype("bf16", checkpoint, torch.device("cuda")) == torch.bfloat16

    with pytest.raises(ValueError, match="requires dtype auto or bf16"):
        resolve_dtype("fp16", checkpoint, torch.device("cuda"))


def test_legacy_mixed_fp8_checkpoint_is_rejected(tmp_path):
    checkpoint = tmp_path / "legacy-fp8.safetensors"
    save_file(
        {"weight": torch.ones(1, dtype=torch.bfloat16)},
        str(checkpoint),
        metadata={"zonos2_quantization": LEGACY_FP8_E4M3_FORMAT},
    )

    with pytest.raises(ValueError, match="retired all-layer mixed FP8"):
        inspect_checkpoint_quantization(checkpoint)


def test_quantized_checkpoint_guard_accepts_expert_gate_up_layout(tmp_path):
    checkpoint = tmp_path / "valid-fp8.safetensors"
    prefix = "layers.3.feed_forward.experts.experts.0.w13"
    save_file(
        {
            f"{prefix}.weight": torch.ones(
                (4, 2),
                dtype=torch.float8_e4m3fn,
            ),
            f"{prefix}.weight_scale": torch.ones((), dtype=torch.float32),
            f"{prefix}.comfy_quant": torch.ones(1, dtype=torch.uint8),
            "layers.3.feed_forward.experts.experts.0.w2.weight": torch.ones(
                (2, 2),
                dtype=torch.bfloat16,
            ),
        },
        str(checkpoint),
        metadata={
            "zonos2_quantization": FP8_E4M3_FORMAT,
            "quantization_policy": FP8_E4M3_POLICY,
            "compute_dtype": "bfloat16",
            "quantized_modules": "1",
        },
    )

    validate_quantized_checkpoint(checkpoint)


def test_quantized_checkpoint_guard_rejects_3d_expert_weight(tmp_path):
    checkpoint = tmp_path / "invalid-fp8.safetensors"
    prefix = "layers.3.feed_forward.experts.experts.0.w13"
    save_file(
        {
            f"{prefix}.weight": torch.ones(
                (1, 4, 2),
                dtype=torch.float8_e4m3fn,
            ),
            f"{prefix}.weight_scale": torch.ones((), dtype=torch.float32),
            f"{prefix}.comfy_quant": torch.ones(1, dtype=torch.uint8),
        },
        str(checkpoint),
        metadata={
            "zonos2_quantization": FP8_E4M3_FORMAT,
            "quantization_policy": FP8_E4M3_POLICY,
            "compute_dtype": "bfloat16",
            "quantized_modules": "1",
        },
    )

    with pytest.raises(ValueError, match="must be a 2D FP8 E4M3 tensor"):
        validate_quantized_checkpoint(checkpoint)


def test_clone_has_native_audio_input():
    required = Zonos2VoiceClone.INPUT_TYPES()["required"]
    assert required["reference_audio"][0] == "AUDIO"
    assert "reference_text" not in required


def test_clone_defaults_match_upstream_conditioning():
    required = Zonos2VoiceClone.INPUT_TYPES()["required"]
    assert required["clean_speaker_background"][1]["default"] is False
    assert required["accurate_mode"][1]["default"] is True


def test_both_generation_nodes_expose_all_quality_features():
    for node in (Zonos2VoiceGeneration, Zonos2VoiceClone):
        required = node.INPUT_TYPES()["required"]
        for control in QUALITY_BUCKET_LABELS:
            assert control in required


def test_reference_audio_over_limit_is_clipped_with_cli_warning(monkeypatch):
    encoder = Zonos2SpeakerEncoder(nn.Linear(1, 1))
    sample_rate = 24_000
    waveform = torch.zeros(1, 1, sample_rate * 61)
    warnings = []
    monkeypatch.setattr(
        logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )

    prepared = encoder._prepare_audio(waveform, sample_rate)

    assert prepared.shape[-1] == int(sample_rate * MAX_REFERENCE_SECONDS)
    assert "Clipping to the first 60.0 seconds" in warnings[0]
