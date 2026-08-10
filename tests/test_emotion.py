from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import zonos2_tts_comfyui_test.runtime as runtime_module
from zonos2_tts_comfyui_test.emotion import (
    EMOTION_NONE,
    EmotionCalibration,
    EmotionDirections,
    emotion_choices,
    emotion_hidden_delta,
    load_calibration,
    load_directions,
)
from zonos2_tts_comfyui_test.native import (
    SamplingOptions,
    Zonos2Model,
    _apply_emotion_cfg,
    generate_audio_codes,
    read_config,
)
from zonos2_tts_comfyui_test.nodes import Zonos2VoiceClone
from zonos2_tts_comfyui_test.runtime import generate_zonos2_audio


ROOT = Path(__file__).resolve().parents[1]


def _small_model() -> Zonos2Model:
    config = replace(
        read_config(ROOT / "assets" / "params.json"),
        n_layers=2,
        dim=8,
        head_dim=4,
        n_heads=2,
        n_kv_heads=1,
        intermediate_size=12,
        speaker_embedding_dim=8,
        speaker_lda_dim=4,
        moe_start_from_layer=99,
    )
    model = Zonos2Model(config)
    model.materialize_runtime_buffers(torch.device("cpu"))
    return model


def test_shipped_directions_are_post_projection():
    directions = load_directions()

    assert directions is not None
    assert directions.space == "proj"
    assert directions.dim == 2048
    assert set(directions.named) == {"happy", "sad", "angry", "surprised"}
    assert set(directions.axes) == {"valence", "arousal"}
    assert all(
        vector.shape == (2048,) and vector.dtype == torch.float32
        for vector in {**directions.named, **directions.axes}.values()
    )


def test_shipped_calibration_matches_upstream_defaults():
    calibration = load_calibration()

    assert calibration is not None
    assert calibration.global_default == 3.0
    assert calibration.strength("surprised") == 4.0
    assert calibration.strength("happy") == 3.0
    assert calibration.strength("valence") == 3.0


def test_delta_folds_the_calibrated_strength_into_each_weight():
    directions = load_directions()
    calibration = load_calibration()

    delta = emotion_hidden_delta(sliders={"surprised": 1.0})

    assert delta is not None
    assert torch.allclose(delta, 4.0 * directions.named["surprised"])
    assert calibration.strength("surprised") == 4.0


def test_delta_scales_linearly_with_strength():
    single = emotion_hidden_delta(sliders={"happy": 1.0}, strength=1.0)
    double = emotion_hidden_delta(sliders={"happy": 1.0}, strength=2.0)

    assert torch.allclose(double, 2.0 * single)


def test_axes_are_combined_with_the_named_direction():
    directions = load_directions()

    delta = emotion_hidden_delta(
        sliders={"sad": 1.0},
        valence=-1.0,
        arousal=0.5,
    )

    expected = (
        3.0 * directions.named["sad"]
        - 3.0 * directions.axes["valence"]
        + 1.5 * directions.axes["arousal"]
    )
    assert torch.allclose(delta, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"sliders": {"happy": 0.0}},
        {"sliders": {"happy": 1.0}, "strength": 0.0},
    ],
)
def test_nothing_requested_yields_no_delta(kwargs):
    assert emotion_hidden_delta(**kwargs) is None


def test_unknown_direction_is_skipped_with_a_warning(caplog):
    with caplog.at_level("WARNING"):
        delta = emotion_hidden_delta(sliders={"smug": 1.0})

    assert delta is None
    assert "smug" in caplog.text


def test_raw_space_directions_are_rejected():
    directions = EmotionDirections(
        dim=4,
        space="raw",
        named={"happy": torch.ones(4)},
    )

    with pytest.raises(ValueError, match="only supports post-projection"):
        emotion_hidden_delta(sliders={"happy": 1.0}, directions=directions)


def test_missing_direction_files_disable_emotion(tmp_path):
    assert load_directions(tmp_path) is None
    assert load_calibration(tmp_path) is None


def test_uncalibrated_directions_use_the_raw_weights():
    directions = EmotionDirections(
        dim=4,
        space="proj",
        named={"happy": torch.ones(4)},
    )

    delta = emotion_hidden_delta(
        sliders={"happy": 2.0},
        directions=directions,
        calibration=EmotionCalibration(),
    )

    assert torch.allclose(delta, 2.0 * torch.ones(4))


def test_model_adds_the_delta_after_the_speaker_projection():
    model = _small_model()
    speaker = torch.randn(1, model.config.speaker_embedding_dim)
    delta = torch.randn(model.config.dim)
    prompt = torch.zeros(
        1,
        4,
        model.config.n_codebooks + 1,
        dtype=torch.int32,
    )

    def run(emotion_delta):
        embedded = []
        handle = model.multi_embedder.register_forward_hook(
            lambda module, inputs, output: embedded.append(output)
        )
        try:
            model(
                prompt,
                model.create_kv_cache(1, 8, torch.device("cpu"), torch.float32),
                "SDPA",
                speaker_embedding=speaker,
                speaker_position=0,
                emotion_delta=emotion_delta,
            )
        finally:
            handle.remove()
        return embedded[0]

    projected = model._speaker_projection(speaker)

    assert torch.allclose(run(None)[:, 0], projected)
    assert torch.allclose(run(delta)[:, 0], projected + delta)


def test_model_rejects_a_delta_of_the_wrong_width():
    model = _small_model()
    prompt = torch.zeros(
        1,
        4,
        model.config.n_codebooks + 1,
        dtype=torch.int32,
    )
    caches = model.create_kv_cache(1, 8, torch.device("cpu"), torch.float32)

    with pytest.raises(ValueError, match="does not match the projected"):
        model(
            prompt,
            caches,
            "SDPA",
            speaker_embedding=torch.randn(1, model.config.speaker_embedding_dim),
            speaker_position=0,
            emotion_delta=torch.randn(model.config.dim + 1),
        )


def _codes_with_cfg(model, scale, delta):
    prompt = torch.zeros(1, 4, model.config.n_codebooks + 1, dtype=torch.int32)
    return generate_audio_codes(
        model,
        prompt,
        attention_backend="SDPA",
        options=SamplingOptions(max_new_tokens=3, temperature=0.0),
        speaker_embedding=torch.randn(1, model.config.speaker_embedding_dim),
        speaker_position=0,
        emotion_delta=delta,
        emotion_cfg_scale=scale,
    )


def test_guidance_runs_an_unguided_twin_in_the_same_batch():
    model = _small_model()
    widths = []
    handle = model.multi_embedder.register_forward_hook(
        lambda module, inputs, output: widths.append(output.shape[0])
    )
    try:
        _codes_with_cfg(model, 1.5, torch.randn(model.config.dim))
    finally:
        handle.remove()

    assert widths and set(widths) == {2}


@pytest.mark.parametrize("scale", [1.0, 1.5])
def test_guidance_leaves_the_batch_alone_without_a_delta(scale):
    model = _small_model()
    widths = []
    handle = model.multi_embedder.register_forward_hook(
        lambda module, inputs, output: widths.append(output.shape[0])
    )
    try:
        _codes_with_cfg(model, scale, None)
    finally:
        handle.remove()

    assert widths and set(widths) == {1}


def test_cfg_combination_follows_the_upstream_formula():
    logits = torch.stack(
        [
            torch.full((2, 4), 3.0),
            torch.full((2, 4), 1.0),
        ]
    )

    combined = _apply_emotion_cfg(logits, 1.5)

    assert combined.shape == (1, 2, 4)
    assert torch.allclose(combined, torch.full((1, 2, 4), 1.0 + 1.5 * 2.0))


def test_clone_exposes_guidance_as_opt_in():
    required = Zonos2VoiceClone.INPUT_TYPES()["required"]

    assert required["emotion_cfg_scale"][1]["default"] == 1.0
    assert required["emotion_cfg_scale"][1]["min"] == 1.0


def _generate_with_emotion(monkeypatch, reference_audio, **emotion):
    captured = {}

    def fake_generate_audio_codes(model, prompt, **kwargs):
        captured.update(kwargs)
        return torch.zeros(1, 1, 9), None

    monkeypatch.setattr(runtime_module, "ensure_codec", lambda bundle: codec)
    monkeypatch.setattr(runtime_module, "ensure_speaker_encoder", lambda bundle: None)
    monkeypatch.setattr(runtime_module, "resume_bundle_to_device", lambda bundle: None)
    monkeypatch.setattr(
        runtime_module,
        "extract_speaker_embedding",
        lambda bundle, audio: torch.zeros(1, 8),
    )
    monkeypatch.setattr(
        runtime_module,
        "build_prompt",
        lambda config, **kwargs: (torch.zeros(1, 1, 9), 0),
    )
    monkeypatch.setattr(
        runtime_module,
        "generate_audio_codes",
        fake_generate_audio_codes,
    )

    class Codec:
        sample_rate = 44_100

        def decode(self, codes, pad_id, eos_frame):
            return torch.zeros(1, 16)

    codec = Codec()
    bundle = SimpleNamespace(
        model=object(),
        config=SimpleNamespace(audio_pad_id=1025),
        attention="SDPA",
    )
    generate_zonos2_audio(
        bundle,
        text="hello",
        options=SamplingOptions(),
        speaking_rate_bucket=-1,
        quality_buckets=[],
        reference_audio=reference_audio,
        **emotion,
    )
    return captured["emotion_delta"]


def test_runtime_passes_the_calibrated_delta_to_the_generator(monkeypatch):
    delta = _generate_with_emotion(
        monkeypatch,
        reference_audio={"waveform": torch.zeros(1, 1, 8), "sample_rate": 24_000},
        emotion_sliders={"angry": 1.0},
    )

    assert torch.allclose(delta, 3.0 * load_directions().named["angry"])


def test_runtime_sends_no_delta_without_a_reference_voice(monkeypatch):
    assert (
        _generate_with_emotion(
            monkeypatch,
            reference_audio=None,
            emotion_sliders={"angry": 1.0},
        )
        is None
    )


def test_clone_exposes_the_shipped_emotions():
    required = Zonos2VoiceClone.INPUT_TYPES()["required"]

    assert required["emotion"][0] == emotion_choices()
    assert required["emotion"][0][0] == EMOTION_NONE
    assert required["emotion"][1]["default"] == EMOTION_NONE
    assert required["emotion_strength"][1]["default"] == 1.0
    assert required["emotion_valence"][1]["default"] == 0.0
    assert required["emotion_arousal"][1]["default"] == 0.0


def test_emotion_widgets_come_after_the_existing_ones():
    required = list(Zonos2VoiceClone.INPUT_TYPES()["required"])

    assert required[-5:] == [
        "emotion",
        "emotion_strength",
        "emotion_valence",
        "emotion_arousal",
        "emotion_cfg_scale",
    ]
    assert required.index("seed") < required.index("emotion")
