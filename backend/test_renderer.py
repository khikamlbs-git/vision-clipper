import math
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from backend.renderer import _validate_rendered_video, extract_and_format_clip
import numpy as np

from backend.renderer import (
    SceneCutDetector,
    _build_shot_trajectory,
    _natural_padding,
    _parse_loudnorm_json,
    _write_crop_commands,
    build_audio_filter,
    build_audio_pre_chain,
    build_split_screen_graph,
    _calc_margin_v,
    generate_ass_subtitles,
    detect_tracking_engines,
    select_tracking_engine,
    compute_smart_crop_trajectory,
    _SpeakerTargetSelector,
    _MouthActivityTracker,
    _detect_faces_yunet,
    _detect_persons_yolo,
    _classify_shot_modes,
    build_hybrid_smart_crop_graph,
)


class RendererIntegrityTest(unittest.TestCase):
    def test_validate_rendered_video_rejects_nonexistent_or_empty_file(self):
        self.assertFalse(_validate_rendered_video("/nonexistent/path/video.mp4"))
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tf:
            tf.write(b"")
            tf.flush()
            self.assertFalse(_validate_rendered_video(tf.name))

    @patch("backend.renderer.get_ffprobe_binary", return_value="ffprobe")
    @patch("backend.renderer.subprocess.run")
    def test_validate_rendered_video_accepts_valid_probe(self, mock_run, _mock_ffprobe):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tf:
            tf.write(b"x" * 2048)
            tf.flush()

            # Mock stream probe & duration probe
            res_stream = MagicMock()
            res_stream.returncode = 0
            res_stream.stdout = '{"streams": [{"codec_name": "h264", "width": 1080, "height": 1920}]}'

            res_dur = MagicMock()
            res_dur.returncode = 0
            res_dur.stdout = "15.5\n"

            mock_run.side_effect = [res_stream, res_dur]

            self.assertTrue(_validate_rendered_video(tf.name, min_duration=5.0))

    @patch("backend.renderer.compute_smart_crop_trajectory")
    @patch("backend.renderer.generate_ass_subtitles")
    @patch("backend.renderer.run_command")
    def test_extract_and_format_clip_raises_if_all_codecs_fail(self, mock_run_command, mock_ass, mock_smart_crop):
        mock_run_command.side_effect = Exception("Encoder hardware error")
        with tempfile.TemporaryDirectory() as out_dir:
            clip_info = {
                "title": "Test Viral Clip",
                "start": 0.0,
                "end": 10.0,
                "caption": "Test Caption"
            }
            with self.assertRaises(RuntimeError) as ctx:
                extract_and_format_clip(
                    video_path="dummy.mp4",
                    clip_info=clip_info,
                    index=1,
                    segments=[],
                    out_dir=out_dir
                )
            self.assertIn("All codecs failed to render clip", str(ctx.exception))
            # Pastikan file mp4 final tidak dibuat
            files = os.listdir(out_dir)
            mp4_files = [f for f in files if f.endswith(".mp4")]
            self.assertEqual(len(mp4_files), 0)


class ShotTrajectoryTest(unittest.TestCase):
    def _samples(self, total, stride, fn):
        idx = list(range(0, total, stride))
        return idx, [fn(i) for i in idx]

    def test_hard_cut_snaps_without_glide(self):
        total, cut = 200, 100
        idx, vals = self._samples(total, 5, lambda i: 500.0 if i < cut else 1500.0)
        out = _build_shot_trajectory(total, idx, vals, [cut], deadzone_px=50, sigma=25, default_x=960)
        self.assertEqual(len(out), total)
        np.testing.assert_allclose(out[:cut], 500.0)
        np.testing.assert_allclose(out[cut:], 1500.0)

    def test_without_cut_glides_like_before(self):
        total = 200
        idx, vals = self._samples(total, 5, lambda i: 500.0 if i < 100 else 1500.0)
        out = _build_shot_trajectory(total, idx, vals, [], deadzone_px=50, sigma=25, default_x=960)
        # Tanpa cut ada transisi halus (nilai di antara 500 dan 1500 di sekitar frame 100)
        self.assertTrue(np.any((out > 600) & (out < 1400)))

    def test_smooth_within_shot(self):
        total = 300
        idx, vals = self._samples(total, 5, lambda i: 500.0 + i * 3.0)
        out = _build_shot_trajectory(total, idx, vals, [150], deadzone_px=10, sigma=25, default_x=960)
        # Di dalam shot tidak boleh ada lompatan besar per frame (gimbal halus)
        self.assertLess(np.max(np.abs(np.diff(out[:150]))), 20)
        self.assertLess(np.max(np.abs(np.diff(out[150:]))), 20)

    def test_shot_without_detection_uses_previous_shot(self):
        total = 120
        idx = [0, 5, 10, 60, 65, 70]
        vals = [400.0, 400.0, 400.0, None, None, None]
        out = _build_shot_trajectory(total, idx, vals, [60], deadzone_px=50, sigma=10, default_x=960)
        np.testing.assert_allclose(out[60:], 400.0)

    def test_first_shot_without_detection_uses_default(self):
        out = _build_shot_trajectory(30, [0, 5], [None, None], [], deadzone_px=50, sigma=10, default_x=960)
        np.testing.assert_allclose(out, 960.0)

    def test_failed_detection_is_held_within_shot_only(self):
        total = 100
        idx = [0, 5, 10, 50, 55]
        vals = [300.0, None, None, 1200.0, None]
        out = _build_shot_trajectory(total, idx, vals, [50], deadzone_px=50, sigma=5, default_x=960)
        np.testing.assert_allclose(out[:50], 300.0)
        np.testing.assert_allclose(out[50:], 1200.0)

    def test_ignores_out_of_range_cuts(self):
        out = _build_shot_trajectory(10, [0], [700.0], [0, 10, 99, -3], deadzone_px=50, sigma=5, default_x=960)
        np.testing.assert_allclose(out, 700.0)


class SceneCutDetectorTest(unittest.TestCase):
    @staticmethod
    def _solid(bgr, noise_seed=None):
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        frame[:] = bgr
        if noise_seed is not None:
            rng = np.random.default_rng(noise_seed)
            frame = np.clip(frame.astype(np.int16) + rng.integers(-4, 5, frame.shape), 0, 255).astype(np.uint8)
        return frame

    def test_detects_hard_cut(self):
        det = SceneCutDetector(fps=30)
        cuts = [i for i in range(60) if det.update(i, self._solid((30, 30, 200) if i < 30 else (200, 120, 20), noise_seed=i))]
        self.assertEqual(cuts, [30])

    def test_small_noise_is_not_a_cut(self):
        det = SceneCutDetector(fps=30)
        cuts = [i for i in range(90) if det.update(i, self._solid((90, 140, 180), noise_seed=i))]
        self.assertEqual(cuts, [])

    def test_flash_within_min_gap_is_ignored(self):
        det = SceneCutDetector(fps=30)
        frames = [(30, 30, 200)] * 20 + [(255, 255, 255)] + [(30, 30, 200)] * 20
        cuts = [i for i, c in enumerate(frames) if det.update(i, self._solid(c))]
        self.assertEqual(cuts, [20])  # kembalinya dari flash (frame 21) diabaikan

    def test_similar_color_angle_change_detected_by_adaptive_rule(self):
        # Dua "angle" dengan warna rata-rata mirip tapi layout berbeda (setengah kiri/kanan ditukar)
        a = np.zeros((90, 160, 3), dtype=np.uint8); a[:, :80] = (40, 60, 160); a[:, 80:] = (60, 50, 90)
        b = a[:, ::-1].copy()
        det = SceneCutDetector(fps=30)
        cuts = [i for i in range(40) if det.update(i, a if i < 20 else b)]
        self.assertEqual(cuts, [20])


class CropCommandTimingTest(unittest.TestCase):
    def test_commands_are_half_frame_early_and_deduplicated(self):
        traj = {"fps": 30.0, "positions": np.array([100.0, 100.0, 101.0, 600.0, 600.0])}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cmd.txt")
            x0 = _write_crop_commands(traj, path)
            with open(path, encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
        self.assertEqual(x0, 100)
        # 101 dibulatkan ke genap -> 100 (duplikat dibuang); cut di frame 3 -> (3-0.5)/30
        self.assertEqual(lines, ["0.0000 crop x 100;", f"{2.5 / 30:.4f} crop x 600;"])


class AudioFilterTest(unittest.TestCase):
    MEASURED = {"input_i": -27.3, "input_tp": -8.1, "input_lra": 6.2, "input_thresh": -37.9, "target_offset": 0.4}

    def test_parse_loudnorm_json_from_stderr(self):
        stderr = (
            "[Parsed_loudnorm_3 @ 0x1] \n{\n\t\"input_i\" : \"-27.30\",\n\t\"input_tp\" : \"-8.10\",\n"
            "\t\"input_lra\" : \"6.20\",\n\t\"input_thresh\" : \"-37.90\",\n\t\"output_i\" : \"-14.0\",\n"
            "\t\"normalization_type\" : \"dynamic\",\n\t\"target_offset\" : \"0.40\"\n}\n"
        )
        self.assertEqual(_parse_loudnorm_json(stderr), self.MEASURED)

    def test_parse_loudnorm_handles_inf_and_garbage(self):
        stderr = '{"input_i" : "-inf", "input_tp" : "-inf", "input_lra" : "0.00", "input_thresh" : "-inf", "target_offset" : "inf"}'
        parsed = _parse_loudnorm_json(stderr)
        self.assertTrue(math.isinf(parsed["input_i"]))
        self.assertIsNone(_parse_loudnorm_json("no json here"))
        self.assertIsNone(_parse_loudnorm_json('{"input_i": "abc"}'))
        self.assertIsNone(_parse_loudnorm_json(""))

    def test_two_pass_filter_uses_measurement(self):
        af, mode = build_audio_filter(9.5, build_audio_pre_chain(), -14.0, self.MEASURED)
        self.assertEqual(mode, "two_pass")
        self.assertIn("loudnorm=I=-14:TP=-1.5:LRA=11:measured_I=-27.30", af)
        self.assertIn("linear=true", af)
        self.assertTrue(af.endswith("afade=t=out:st=9.500:d=0.5"))
        self.assertLess(af.index("acompressor"), af.index("loudnorm"))
        self.assertLess(af.index("loudnorm"), af.index("afade"))

    def test_single_pass_fallback_when_measurement_missing(self):
        af, mode = build_audio_filter(5.0, build_audio_pre_chain(), -14.0, None)
        self.assertEqual(mode, "single_pass")
        self.assertIn("loudnorm=I=-14:TP=-1.5:LRA=11,", af)
        self.assertNotIn("measured_I", af)

    def test_silent_clip_skips_loudnorm(self):
        for measured in (
            dict(self.MEASURED, input_i=float("-inf"), input_thresh=float("-inf")),
            dict(self.MEASURED, input_i=-62.0),
        ):
            af, mode = build_audio_filter(5.0, build_audio_pre_chain(), -14.0, measured)
            self.assertEqual(mode, "skip")
            self.assertNotIn("loudnorm", af)

    def test_pre_chain_denoise_toggle(self):
        self.assertNotIn("afftdn", build_audio_pre_chain(denoise=False))
        self.assertIn("afftdn", build_audio_pre_chain(denoise=True))
        self.assertIn("highpass=f=80", build_audio_pre_chain())


class SplitScreenAndSubtitleEnhancementTest(unittest.TestCase):
    @patch("backend.renderer.detect_speaker_anchors")
    @patch("backend.renderer._probe_video_stream")
    def test_build_split_screen_graph_structure(self, mock_probe, mock_anchors):
        mock_probe.return_value = (1920, 1080, 30.0)
        mock_anchors.return_value = {"left_anchor": 480.0, "right_anchor": 1440.0}

        finish_chain = "eq=contrast=1.1,fade=t=out:st=9.5:d=0.5"
        graph = build_split_screen_graph("dummy.mp4", 0.0, 10.0, finish_chain)

        self.assertIn("split=2[vtop_in][vbot_in]", graph)
        self.assertIn("scale=1080:960", graph)
        self.assertIn("vstack=inputs=2[stacked]", graph)
        self.assertIn("drawbox=x=0:y=958:w=1080:h=4", graph)
        self.assertIn(finish_chain, graph)

    def test_calc_margin_v(self):
        # 68% from top -> 32% from bottom: 1920 * 0.32 = 614
        self.assertEqual(_calc_margin_v(68), 614)
        # 50% from top -> 50% from bottom: 1920 * 0.50 = 960
        self.assertEqual(_calc_margin_v(50), 960)
        # 80% from top -> 20% from bottom: 1920 * 0.20 = 384
        self.assertEqual(_calc_margin_v(80), 384)

    @patch("backend.renderer.build_split_screen_graph")
    @patch("backend.renderer.generate_ass_subtitles")
    @patch("backend.renderer.run_command")
    @patch("backend.renderer._validate_rendered_video")
    def test_extract_and_format_clip_split_screen_mode(
        self, mock_validate, mock_run, mock_ass, mock_split
    ):
        mock_split.return_value = "[0:v]split=2[top][bot]vstack[v]"
        def fake_run(cmd, *args, **kwargs):
            part_path = cmd[-1]
            with open(part_path, "wb") as f:
                f.write(b"x" * 2048)
            return "Done"
        mock_run.side_effect = fake_run
        mock_validate.return_value = True

        with tempfile.TemporaryDirectory() as out_dir:
            clip_info = {
                "title": "Split Screen Test",
                "start": 0.0,
                "end": 8.0,
                "caption": "Test Split"
            }
            out = extract_and_format_clip(
                video_path="dummy.mp4",
                clip_info=clip_info,
                index=1,
                segments=[],
                out_dir=out_dir,
                framing_mode="split_screen",
                sub_position=65,
            )
            self.assertTrue(mock_split.called)
            self.assertTrue(mock_ass.called)
            self.assertEqual(mock_ass.call_args[1]["sub_position"], 65)
            self.assertTrue(os.path.exists(out))

    def test_generate_ass_subtitles_with_watermark_glass_effect(self):
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as tf:
            ass_path = tf.name

        try:
            segments = [
                {
                    "start": 1.0,
                    "end": 3.0,
                    "words": [
                        {"word": "Halo", "start": 1.0, "end": 1.5},
                        {"word": "dunia", "start": 1.6, "end": 2.2}
                    ]
                }
            ]
            generate_ass_subtitles(
                segments,
                ass_path,
                clip_start=0.0,
                clip_end=5.0,
                sub_style="tiktok_viral",
                watermark_enabled=True,
                source_watermark="Deddy Corbuzier",
                account_watermark="@visionclipper"
            )

            self.assertTrue(os.path.exists(ass_path))
            with open(ass_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("Style: GlassSource,Plus Jakarta Sans,34", content)
            self.assertIn("Style: GlassAccount,Plus Jakarta Sans,36", content)
            # Liquid glass: BorderStyle=1 (tanpa background box BorderStyle=3 / &H70181412)
            self.assertNotIn("&H70181412", content)
            self.assertNotIn(",3,14,3,", content)
            self.assertNotIn("▶", content)
            self.assertIn("Source: Deddy Corbuzier", content)
            self.assertIn("@visionclipper", content)
            self.assertIn("ViralStyle", content)
        finally:
            if os.path.exists(ass_path):
                os.remove(ass_path)

    def test_generate_ass_subtitles_no_subtitle_with_watermark(self):
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as tf:
            ass_path = tf.name

        try:
            generate_ass_subtitles(
                segments=[],
                out_ass_path=ass_path,
                clip_start=0.0,
                clip_end=4.0,
                sub_style="no_subtitle",
                watermark_enabled=True,
                source_watermark="Close The Door",
                account_watermark="myhandle"
            )

            self.assertTrue(os.path.exists(ass_path))
            with open(ass_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("Style: GlassSource", content)
            self.assertIn("Style: GlassAccount", content)
            self.assertNotIn("▶", content)
            self.assertIn("Source: Close The Door", content)
            self.assertIn("@myhandle", content)
            self.assertNotIn("ViralStyle", content)
        finally:
            if os.path.exists(ass_path):
                os.remove(ass_path)

    def test_generate_ass_subtitles_no_subtitle_and_no_watermark(self):
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as tf:
            ass_path = tf.name
        # Remove file so we can check if it gets created
        os.remove(ass_path)

        generate_ass_subtitles(
            segments=[],
            out_ass_path=ass_path,
            clip_start=0.0,
            clip_end=4.0,
            sub_style="no_subtitle",
            watermark_enabled=False
        )
        self.assertFalse(os.path.exists(ass_path))

    @patch("backend.renderer.generate_ass_subtitles")
    @patch("backend.renderer.run_command")
    @patch("backend.renderer._validate_rendered_video")
    def test_extract_and_format_clip_passes_watermark_params(
        self, mock_validate, mock_run, mock_ass
    ):
        def fake_run(cmd, *args, **kwargs):
            part_path = cmd[-1]
            with open(part_path, "wb") as f:
                f.write(b"x" * 2048)
            return "Done"
        mock_run.side_effect = fake_run
        mock_validate.return_value = True

        with tempfile.TemporaryDirectory() as out_dir:
            clip_info = {
                "title": "Watermark Param Test",
                "start": 0.0,
                "end": 5.0,
                "caption": "Test"
            }
            extract_and_format_clip(
                video_path="dummy.mp4",
                clip_info=clip_info,
                index=1,
                segments=[],
                out_dir=out_dir,
                watermark_enabled=True,
                source_watermark="YouTube Channel",
                account_watermark="@clipper"
            )
            self.assertTrue(mock_ass.called)
            kwargs = mock_ass.call_args[1]
            self.assertTrue(kwargs["watermark_enabled"])
            self.assertEqual(kwargs["source_watermark"], "YouTube Channel")
            self.assertEqual(kwargs["account_watermark"], "@clipper")

    def test_native_mode_keeps_word_highlight(self):
        """Regresi: konten native (id->id) tidak boleh berubah gayanya."""
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as tf:
            ass_path = tf.name

        try:
            segments = [{
                "start": 1.0, "end": 3.0,
                "words": [
                    {"word": "Konsisten", "start": 1.0, "end": 1.5},
                    {"word": "itu", "start": 1.6, "end": 2.0},
                ],
            }]
            generate_ass_subtitles(
                segments, ass_path, clip_start=0.0, clip_end=4.0,
                sub_style="tiktok_viral",
            )
            with open(ass_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("ViralStyle", content)
            self.assertIn("&H0099D334&", content)
        finally:
            if os.path.exists(ass_path):
                os.remove(ass_path)


class TrackingEngineSelectionTest(unittest.TestCase):
    def test_detect_tracking_engines_prefers_yunet_with_yolo_fallback(self):
        with patch("backend.renderer._is_yunet_available", return_value=True), \
             patch("backend.renderer._is_yolo_available", return_value=True), \
             patch("backend.renderer._is_haar_available", return_value=True), \
             patch("backend.renderer._get_torch_device", return_value="cpu"):
            info = detect_tracking_engines()
            self.assertEqual(info["recommended"], "yunet")
            self.assertEqual(info["fallback"], "yolo")
            self.assertEqual(info["available"], ["yunet", "yolo", "haar"])

            plan = select_tracking_engine("auto")
            self.assertEqual(plan["primary"], "yunet")
            self.assertEqual(plan["fallback"], "yolo")

    def test_select_tracking_engine_respects_explicit_preference_and_fallbacks(self):
        with patch("backend.renderer._is_yunet_available", return_value=True), \
             patch("backend.renderer._is_yolo_available", return_value=True), \
             patch("backend.renderer._is_haar_available", return_value=True), \
             patch("backend.renderer._get_torch_device", return_value="cuda"):
            plan_yolo = select_tracking_engine("yolo")
            self.assertEqual(plan_yolo["primary"], "yolo")
            self.assertEqual(plan_yolo["fallback"], "yunet")

            plan_yunet = select_tracking_engine("yunet")
            self.assertEqual(plan_yunet["primary"], "yunet")
            self.assertEqual(plan_yunet["fallback"], "haar")

        # Jika YuNet tidak tersedia, otomatis pakai YOLO
        with patch("backend.renderer._is_yunet_available", return_value=False), \
             patch("backend.renderer._is_yolo_available", return_value=True), \
             patch("backend.renderer._is_haar_available", return_value=True), \
             patch("backend.renderer._get_torch_device", return_value="cpu"):
            plan = select_tracking_engine("auto")
            self.assertEqual(plan["primary"], "yolo")
            self.assertEqual(plan["fallback"], "haar")

    @patch("backend.renderer._probe_video_stream", return_value=(1920, 1080, 30.0))
    @patch("backend.renderer._iter_ffmpeg_frames")
    @patch("backend.renderer._get_yolo_model")
    @patch("backend.renderer._get_yunet_detector")
    @patch("backend.renderer._detect_faces_yunet")
    def test_compute_trajectory_skips_yolo_load_when_yunet_succeeds(
        self, mock_detect_yunet, mock_get_yunet, mock_get_yolo, mock_iter_frames, _mock_probe
    ):
        frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(15)]
        mock_iter_frames.return_value = iter(frames)
        mock_get_yunet.return_value = MagicMock()
        mock_detect_yunet.return_value = [{"cx": 200.0, "cy": 150.0, "w": 60.0, "h": 80.0, "area": 4800.0, "engine": "yunet"}]

        with patch("backend.renderer.select_tracking_engine", return_value={
            "requested": "auto", "primary": "yunet", "fallback": "yolo", "device": "cpu", "available": ["yunet", "yolo"]
        }):
            traj = compute_smart_crop_trajectory("dummy.mp4", 0.0, 0.5)

        self.assertEqual(traj["engine"], "yunet")
        self.assertGreater(traj["engine_hits"]["yunet"], 0)
        self.assertEqual(traj["engine_hits"]["yolo"], 0)
        # Efisiensi utama: YOLOv8 tidak pernah di-load saat YuNet sudah mendeteksi wajah
        mock_get_yolo.assert_not_called()

    @patch("backend.renderer._probe_video_stream", return_value=(1920, 1080, 30.0))
    @patch("backend.renderer._iter_ffmpeg_frames")
    @patch("backend.renderer._get_yolo_model")
    @patch("backend.renderer._get_yunet_detector")
    @patch("backend.renderer._detect_faces_yunet", return_value=[])
    @patch("backend.renderer._detect_persons_yolo")
    def test_compute_trajectory_lazy_loads_yolo_when_yunet_misses(
        self, mock_detect_yolo, _mock_detect_yunet, mock_get_yunet, mock_get_yolo, mock_iter_frames, _mock_probe
    ):
        frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(15)]
        mock_iter_frames.return_value = iter(frames)
        mock_get_yunet.return_value = MagicMock()
        mock_get_yolo.return_value = MagicMock()
        mock_detect_yolo.return_value = [{"cx": 480.0, "cy": 200.0, "w": 120.0, "h": 240.0, "area": 28800.0, "engine": "yolo"}]

        with patch("backend.renderer.select_tracking_engine", return_value={
            "requested": "auto", "primary": "yunet", "fallback": "yolo", "device": "cpu", "available": ["yunet", "yolo"]
        }):
            traj = compute_smart_crop_trajectory("dummy.mp4", 0.0, 0.5)

        mock_get_yolo.assert_called_once()
        self.assertGreater(traj["engine_hits"]["yolo"], 0)

    def test_speaker_target_selector_locks_on_split_screen_instead_of_ping_pong(self):
        selector = _SpeakerTargetSelector(ana_w=640, switch_streak=4, dominance_ratio=1.35)
        left = {"cx": 160.0, "cy": 140.0, "w": 60.0, "h": 80.0, "area": 4800.0, "engine": "yunet"}
        right_slightly_bigger = {"cx": 480.0, "cy": 140.0, "w": 62.0, "h": 81.0, "area": 5022.0, "engine": "yunet"}

        # Frame 1: kunci ke kanan karena sedikit lebih besar saat awal
        d0, switched0 = selector.select([right_slightly_bigger, left])
        self.assertEqual(d0["cx"], 480.0)
        self.assertFalse(switched0)

        # Frame berikutnya: kiri sedikit lebih besar (fluktuasi kecil < 1.35x) -> tetap terkunci di kanan
        left_slightly_bigger = {"cx": 162.0, "cy": 140.0, "w": 63.0, "h": 81.0, "area": 5103.0, "engine": "yunet"}
        for _ in range(6):
            d, switched = selector.select([left_slightly_bigger, right_slightly_bigger])
            self.assertEqual(d["cx"], 480.0)
            self.assertFalse(switched)

    @patch("backend.renderer._probe_video_stream", return_value=(1920, 1080, 30.0))
    @patch("backend.renderer._iter_ffmpeg_frames")
    @patch("backend.renderer._get_yunet_detector")
    @patch("backend.renderer._detect_faces_yunet")
    def test_compute_trajectory_split_screen_does_not_average_to_center_seam(
        self, mock_detect_yunet, mock_get_yunet, mock_iter_frames, _mock_probe
    ):
        # 30 frame = 6 sampel (stride=5). Area kiri & kanan bergantian sedikit lebih besar.
        frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(30)]
        mock_iter_frames.return_value = iter(frames)
        mock_get_yunet.return_value = MagicMock()

        left_init = {"cx": 160.0, "cy": 140.0, "w": 61.0, "h": 81.0, "area": 4950.0, "engine": "yunet"}
        left = {"cx": 160.0, "cy": 140.0, "w": 60.0, "h": 80.0, "area": 4800.0, "engine": "yunet"}
        right = {"cx": 480.0, "cy": 140.0, "w": 60.0, "h": 80.0, "area": 4900.0, "engine": "yunet"}
        left_big = {"cx": 160.0, "cy": 140.0, "w": 61.0, "h": 82.0, "area": 5002.0, "engine": "yunet"}
        mock_detect_yunet.side_effect = [
            [left_init, right],     # sample 0: left lebih besar -> kunci left
            [right, left],          # sample 1: right sedikit lebih besar (< 1.35x)
            [left_big, right],      # sample 2
            [right, left],          # sample 3
            [left_big, right],      # sample 4
            [right, left],          # sample 5
        ]
        # Di sample 0, left dan right -> _detect_faces_yunet biasanya sudah sorted descending,
        # jadi sample 0 dengan [left, right] mengunci left (cx=160 -> 480px di 1920w).
        with patch("backend.renderer.select_tracking_engine", return_value={
            "requested": "auto", "primary": "yunet", "fallback": None, "device": "cpu", "available": ["yunet"]
        }):
            traj = compute_smart_crop_trajectory("dummy.mp4", 0.0, 1.0)

        # Center crop X1 untuk cx=480 dengan crop_w=606 -> 480 - 303 = 177 (bukan tengah seam ~657!)
        mean_crop_x = float(np.mean(traj["positions"]))
        self.assertLess(mean_crop_x, 300.0)

    def test_detect_persons_yolo_filters_merged_wide_box_and_refines_head(self):
        fake_yolo = MagicMock()
        fake_res = MagicMock()
        # Box 1: terlalu lebar (0..550 dari 640 = 86%, gabungan 2 orang/meja) -> harus dibuang
        # Box 2: orang kiri (20..260 = 240px = 37.5%) -> diterima & di-refine ke posisi wajah
        fake_res.boxes.__len__.return_value = 2
        fake_res.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([
            [0.0, 20.0, 550.0, 350.0],
            [20.0, 40.0, 260.0, 350.0],
        ])
        fake_yolo.predict.return_value = [fake_res]

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        with patch("backend.renderer._refine_person_head_with_yunet", return_value={"cx": 95.0, "cy": 100.0, "w": 45.0, "h": 60.0}):
            dets = _detect_persons_yolo(fake_yolo, frame, 640, 360, device="cpu", yunet_detector=MagicMock())

        self.assertEqual(len(dets), 1)
        self.assertAlmostEqual(dets[0]["cx"], 95.0)

    def test_mouth_activity_tracker_switches_to_speaking_speaker(self):
        tracker = _MouthActivityTracker(ana_w=640, ema_alpha=0.6)
        selector = _SpeakerTargetSelector(ana_w=640, switch_streak=4, speech_switch_streak=3, min_hold_samples=1)

        quiet_mouth = np.zeros((16, 24), dtype=np.float32)
        forehead = np.zeros((12, 24), dtype=np.float32)
        rng = np.random.default_rng(42)

        # Sample 0: kunci ke kiri (keduanya diam)
        d0 = [
            {"cx": 160.0, "area": 5000.0, "mouth_patch": quiet_mouth, "forehead_patch": forehead, "engine": "yunet"},
            {"cx": 480.0, "area": 4900.0, "mouth_patch": quiet_mouth, "forehead_patch": forehead, "engine": "yunet"},
        ]
        tracker.update(d0)
        chosen, _ = selector.select(d0)
        self.assertEqual(chosen["cx"], 160.0)

        # Sample 1..4: kanan mulai berbicara (patch mulut berubah aktif), kiri tetap diam
        switched_to_right = False
        for i in range(4):
            speaking_mouth = (rng.uniform(-25.0, 25.0, (16, 24))).astype(np.float32)
            dets = [
                {"cx": 160.0, "area": 5000.0, "mouth_patch": quiet_mouth, "forehead_patch": forehead, "engine": "yunet"},
                {"cx": 480.0, "area": 4900.0, "mouth_patch": speaking_mouth, "forehead_patch": forehead, "engine": "yunet"},
            ]
            tracker.update(dets)
            chosen, switched_far = selector.select(dets)
            if chosen is not None and chosen["cx"] == 480.0 and switched_far:
                switched_to_right = True
                break
        self.assertTrue(switched_to_right)

    def test_classify_shot_modes_switches_wide_or_empty_shots_to_fit_blur(self):
        # Shot 0..30: close-up wajah besar (w=80 dari 640 = 12.5%) -> smart_crop
        # Shot 30..60: zoom-out / wide (w=24 dari 640 = 3.75%) -> fit_blur
        # Shot 60..90: tidak ada wajah terdeteksi -> fit_blur
        sample_indices = [0, 5, 10, 30, 35, 40, 60, 65, 70]
        sample_meta = [
            {"count": 1, "min_cx": 200.0, "max_cx": 200.0, "max_w": 80.0, "is_face": True},
            {"count": 1, "min_cx": 202.0, "max_cx": 202.0, "max_w": 78.0, "is_face": True},
            {"count": 1, "min_cx": 198.0, "max_cx": 198.0, "max_w": 82.0, "is_face": True},
            {"count": 2, "min_cx": 150.0, "max_cx": 490.0, "max_w": 24.0, "is_face": True},
            {"count": 2, "min_cx": 150.0, "max_cx": 490.0, "max_w": 22.0, "is_face": True},
            {"count": 2, "min_cx": 150.0, "max_cx": 490.0, "max_w": 25.0, "is_face": True},
            {"count": 0, "min_cx": None, "max_cx": None, "max_w": 0.0, "is_face": False},
            {"count": 0, "min_cx": None, "max_cx": None, "max_w": 0.0, "is_face": False},
            {"count": 0, "min_cx": None, "max_cx": None, "max_w": 0.0, "is_face": False},
        ]
        modes = _classify_shot_modes(90, [30, 60], sample_indices, sample_meta, ana_w=640)
        self.assertEqual(modes, [
            {"start": 0, "end": 30, "mode": "smart_crop"},
            {"start": 30, "end": 90, "mode": "fit_blur"},
        ])

    def test_build_hybrid_smart_crop_graph_composites_fit_blur_segments(self):
        traj = {
            "fps": 30.0,
            "orig_w": 1920,
            "orig_h": 1080,
            "crop_w": 606,
            "crop_h": 1080,
            "crop_y": 0,
            "positions": [100.0] * 30 + [1188.0] * 30 + [100.0] * 30,
            "shot_modes": [
                {"start": 0, "end": 30, "mode": "smart_crop"},
                {"start": 30, "end": 60, "mode": "fit_blur"},
                {"start": 60, "end": 90, "mode": "smart_crop"},
            ],
            "fit_blur_zoom": 1.78,
        }
        with tempfile.TemporaryDirectory() as d:
            cmd_path = os.path.join(d, "cmd.txt")
            graph = build_hybrid_smart_crop_graph(traj, cmd_path, "eq=contrast=1.1", out_label="vsub")
        self.assertIn("split=3[crop_in][bg_in][fg_in]", graph)
        self.assertIn("scale=1922:-2", graph)
        # Posisi segmen fit_blur berada di kanan (x=1188 -> cx=1491) -> fit_x dikunci ke sisi kanan (842), bukan tengah (421)
        self.assertIn("crop=1080:min(ih\\,1920):842:0", graph)
        self.assertIn("boxblur=25:5", graph)
        self.assertIn("overlay=0:0:enable='between(t,", graph)
        self.assertTrue(graph.endswith("[vsub]"))

    def test_classify_shot_modes_handles_zoom_out_without_scene_cut(self):
        # 1 shot tanpa cut_frames: close-up -> zoom-out (body-only / wide) >= 4 sample -> close-up >= 3 sample
        sample_indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        sample_meta = [
            {"count": 1, "min_cx": 200.0, "max_cx": 200.0, "max_w": 80.0, "is_face": True},
            {"count": 1, "min_cx": 202.0, "max_cx": 202.0, "max_w": 78.0, "is_face": True},
            {"count": 1, "min_cx": 204.0, "max_cx": 204.0, "max_w": 82.0, "is_face": True},
            {"count": 1, "min_cx": 320.0, "max_cx": 320.0, "max_w": 90.0, "is_face": False},
            {"count": 1, "min_cx": 320.0, "max_cx": 320.0, "max_w": 90.0, "is_face": False},
            {"count": 2, "min_cx": 140.0, "max_cx": 500.0, "max_w": 24.0, "is_face": True},
            {"count": 0, "min_cx": None, "max_cx": None, "max_w": 0.0, "is_face": False},
            {"count": 1, "min_cx": 210.0, "max_cx": 210.0, "max_w": 78.0, "is_face": True},
            {"count": 1, "min_cx": 212.0, "max_cx": 212.0, "max_w": 80.0, "is_face": True},
            {"count": 1, "min_cx": 215.0, "max_cx": 215.0, "max_w": 82.0, "is_face": True},
            {"count": 1, "min_cx": 216.0, "max_cx": 216.0, "max_w": 80.0, "is_face": True},
            {"count": 1, "min_cx": 218.0, "max_cx": 218.0, "max_w": 79.0, "is_face": True},
            {"count": 1, "min_cx": 220.0, "max_cx": 220.0, "max_w": 81.0, "is_face": True},
        ]
        modes = _classify_shot_modes(65, [], sample_indices, sample_meta, ana_w=640)
        self.assertEqual(modes, [
            {"start": 0, "end": 15, "mode": "smart_crop"},
            {"start": 15, "end": 35, "mode": "fit_blur"},
            {"start": 35, "end": 65, "mode": "smart_crop"},
        ])

    def test_detect_faces_yunet_filters_top_wall_portrait_and_single_wide_switches_fit_blur(self):
        # Wajah lukisan/poster di pojok atas (cy=28/360 = 0.078 < 0.10) harus diabaikan
        fake_det = MagicMock()
        fake_det.detect.return_value = (
            1,
            np.array([
                [600.0, 10.0, 32.0, 36.0, 0.90],   # top-right wall portrait (cy=28)
                [240.0, 80.0, 68.0, 88.0, 0.94],   # pembicara nyata (cy=124)
            ], dtype=np.float32),
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        faces = _detect_faces_yunet(fake_det, frame, 640, 360, min_w_ratio=0.04, max_cy_ratio=0.75, min_cy_ratio=0.10)
        self.assertEqual(len(faces), 1)
        self.assertAlmostEqual(faces[0]["cx"], 274.0)

        # Shot wide ketika hanya 1 wajah tampak dengan lebar 6% (< 7%) -> fit_blur
        modes = _classify_shot_modes(
            30,
            [],
            [0, 5, 10, 15, 20],
            [{"count": 1, "min_cx": 490.0, "max_cx": 490.0, "max_w": 39.0, "is_face": True}] * 5,
            ana_w=640,
        )
        self.assertEqual(modes, [{"start": 0, "end": 30, "mode": "fit_blur"}])

    def test_detect_faces_yunet_rejects_low_score_and_weird_aspect_objects(self):
        fake_det = MagicMock()
        fake_det.detect.return_value = (
            1,
            np.array([
                [300.0, 120.0, 28.0, 32.0, 0.48],   # objek latar skor rendah (<0.55)
                [260.0, 100.0, 120.0, 40.0, 0.85],  # rasio aneh melebar (40/120 = 0.33 < 0.75)
                [310.0, 100.0, 30.0, 38.0, 0.62],   # objek sekunder lemah saat ada wajah utama kuat
                [100.0, 90.0, 68.0, 88.0, 0.93],    # wajah pembicara nyata
            ], dtype=np.float32),
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        faces = _detect_faces_yunet(fake_det, frame, 640, 360, min_w_ratio=0.04, max_cy_ratio=0.75, min_cy_ratio=0.10)
        self.assertEqual(len(faces), 1)
        self.assertAlmostEqual(faces[0]["cx"], 134.0)

    def test_classify_shot_modes_ignores_single_cut_miss_and_keeps_large_yolo_closeup(self):
        # Sampel 0 tepat di cut gagal deteksi (count=0), tapi sampel 1..5 adalah close-up -> tetap smart_crop penuh
        modes = _classify_shot_modes(
            30,
            [],
            [0, 5, 10, 15, 20, 25],
            [
                {"count": 0, "min_cx": None, "max_cx": None, "max_w": 0.0, "is_face": False},
                {"count": 1, "min_cx": 200.0, "max_cx": 200.0, "max_w": 75.0, "is_face": True},
                {"count": 1, "min_cx": 202.0, "max_cx": 202.0, "max_w": 75.0, "is_face": True},
                {"count": 1, "min_cx": 200.0, "max_cx": 200.0, "max_w": 75.0, "is_face": True},
                {"count": 1, "min_cx": 200.0, "max_cx": 200.0, "max_w": 75.0, "is_face": True},
                {"count": 1, "min_cx": 200.0, "max_cx": 200.0, "max_w": 75.0, "is_face": True},
            ],
            ana_w=640,
        )
        self.assertEqual(modes, [{"start": 0, "end": 30, "mode": "smart_crop"}])

        # 1 orang besar close-up dari YOLO (max_w = 180/640 = 28% >= 22%) saat menoleh -> tetap smart_crop
        modes_yolo = _classify_shot_modes(
            30,
            [],
            [0, 5, 10, 15, 20, 25],
            [{"count": 1, "min_cx": 320.0, "max_cx": 320.0, "max_w": 180.0, "is_face": False}] * 6,
            ana_w=640,
        )
        self.assertEqual(modes_yolo, [{"start": 0, "end": 30, "mode": "smart_crop"}])

    @patch("backend.renderer._probe_video_stream", return_value=(1920, 1080, 30.0))
    @patch("backend.renderer._iter_ffmpeg_frames")
    @patch("backend.renderer._get_yunet_detector")
    @patch("backend.renderer._detect_faces_yunet")
    def test_compute_trajectory_suppresses_isolated_single_sample_spike(
        self, mock_detect_yunet, mock_get_yunet, mock_iter_frames, _mock_probe
    ):
        frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(25)]
        mock_iter_frames.return_value = iter(frames)
        mock_get_yunet.return_value = MagicMock()
        real_face = {"cx": 150.0, "cy": 140.0, "w": 65.0, "h": 82.0, "area": 5330.0, "score": 0.94, "engine": "yunet"}
        spike_obj = {"cx": 480.0, "cy": 140.0, "w": 65.0, "h": 82.0, "area": 5330.0, "score": 0.90, "engine": "yunet"}
        mock_detect_yunet.side_effect = [
            [real_face],
            [real_face],
            [spike_obj],   # 1-sample spike di tengah shot -> harus dibuang
            [real_face],
            [real_face],
        ]
        with patch("backend.renderer.select_tracking_engine", return_value={
            "requested": "auto", "primary": "yunet", "fallback": None, "device": "cpu", "available": ["yunet"]
        }):
            traj = compute_smart_crop_trajectory("dummy.mp4", 0.0, 0.83)
        # Target tetap di sisi kiri (150*3 - 303 = 147), tidak melompat ke kanan
        self.assertLess(float(np.max(traj["positions"])), 200.0)

    def test_yunet_landmark_validation_rejects_scrambled_non_face_objects(self):
        fake_det = MagicMock()
        fake_det.detect.return_value = (
            1,
            np.array([
                # 1) Pareidolia objek (mikrofon/cangkir): mulut (y=105) di atas mata (y=155) -> ditolak!
                [280.0, 90.0, 60.0, 80.0, 295.0, 155.0, 325.0, 155.0, 310.0, 130.0, 298.0, 105.0, 322.0, 105.0, 0.84],
                # 2) Landmark keluar jauh dari bounding box -> ditolak!
                [400.0, 90.0, 60.0, 80.0, 320.0, 115.0, 440.0, 115.0, 430.0, 135.0, 415.0, 155.0, 445.0, 155.0, 0.88],
                # 3) Objek kecil di latar (w=28 < 5.5% * 640) dengan skor 0.62 (< 0.70) -> ditolak!
                [500.0, 110.0, 28.0, 36.0, 507.0, 122.0, 520.0, 122.0, 514.0, 130.0, 508.0, 139.0, 520.0, 139.0, 0.62],
                # 4) Wajah manusia valid (mata -> hidung -> mulut) -> diterima!
                [140.0, 80.0, 70.0, 90.0, 158.0, 112.0, 192.0, 113.0, 175.0, 132.0, 160.0, 152.0, 190.0, 152.0, 0.93],
            ], dtype=np.float32),
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        faces = _detect_faces_yunet(fake_det, frame, 640, 360, min_w_ratio=0.04, max_cy_ratio=0.75, min_cy_ratio=0.10)
        self.assertEqual(len(faces), 1)
        self.assertAlmostEqual(faces[0]["cx"], 175.0)

    def test_static_prop_wall_poster_never_steals_focus_when_speaker_turns_away(self):
        tracker = _MouthActivityTracker(ana_w=640, ema_alpha=0.5)
        selector = _SpeakerTargetSelector(ana_w=640, switch_streak=4)
        rng = np.random.default_rng(7)
        frozen_mouth = np.zeros((16, 24), dtype=np.float32)
        forehead = np.zeros((12, 24), dtype=np.float32)

        # 6 sampel pertama: pembicara hidup di kiri (aktif bicara) + poster dinding statis di kanan
        for i in range(6):
            live_mouth = rng.uniform(-20.0, 20.0, (16, 24)).astype(np.float32)
            dets = [
                {
                    "cx": 180.0 + (i % 2) * 1.5, "cy": 145.0, "w": 72.0, "h": 92.0,
                    "area": 6624.0, "mouth_patch": live_mouth, "forehead_patch": forehead, "engine": "yunet",
                },
                {
                    "cx": 470.0, "cy": 110.0, "w": 42.0, "h": 54.0,
                    "area": 2268.0, "mouth_patch": frozen_mouth, "forehead_patch": forehead, "engine": "yunet",
                },
            ]
            tracker.update(dets)
            chosen, _ = selector.select(dets)
            self.assertAlmostEqual(chosen["cx"], 180.0 + (i % 2) * 1.5)

        # Pembicara kiri menoleh/minum selama 8 sampel (> switch_streak), hanya poster kanan yang terdeteksi
        for _ in range(8):
            only_poster = [{
                "cx": 470.0, "cy": 110.0, "w": 42.0, "h": 54.0,
                "area": 2268.0, "mouth_patch": frozen_mouth, "forehead_patch": forehead, "engine": "yunet",
            }]
            tracker.update(only_poster)
            chosen, switched = selector.select(only_poster)
            # Kamera TIDAK boleh pindah menyorot poster dinding!
            self.assertIsNone(chosen)
            self.assertFalse(switched)

    @patch("backend.renderer._probe_video_stream", return_value=(1920, 1080, 30.0))
    @patch("backend.renderer._iter_ffmpeg_frames")
    @patch("backend.renderer._get_yunet_detector")
    @patch("backend.renderer._detect_faces_yunet")
    def test_mid_shot_speaker_switch_preserves_earlier_speaker_and_cuts_without_center_glide(
        self, mock_detect_yunet, mock_get_yunet, mock_iter_frames, _mock_probe
    ):
        # 120 frame (4 detik, 1 shot tanpa cut kamera): 0-2s kiri bicara, 2-4s kanan bicara
        frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(120)]
        mock_iter_frames.return_value = iter(frames)
        mock_get_yunet.return_value = MagicMock()

        rng = np.random.default_rng(99)
        quiet = np.zeros((16, 24), dtype=np.float32)
        fh = np.zeros((12, 24), dtype=np.float32)
        samples = []
        for s in range(24):
            talk = rng.uniform(-30.0, 30.0, (16, 24)).astype(np.float32)
            if s < 12:
                samples.append([
                    {"cx": 150.0, "cy": 140.0, "w": 66.0, "h": 84.0, "area": 5544.0, "mouth_patch": talk, "forehead_patch": fh, "engine": "yunet"},
                    {"cx": 490.0, "cy": 140.0, "w": 64.0, "h": 82.0, "area": 5248.0, "mouth_patch": quiet, "forehead_patch": fh, "engine": "yunet"},
                ])
            else:
                samples.append([
                    {"cx": 150.0, "cy": 140.0, "w": 66.0, "h": 84.0, "area": 5544.0, "mouth_patch": quiet, "forehead_patch": fh, "engine": "yunet"},
                    {"cx": 490.0, "cy": 140.0, "w": 64.0, "h": 82.0, "area": 5248.0, "mouth_patch": talk, "forehead_patch": fh, "engine": "yunet"},
                ])
        mock_detect_yunet.side_effect = samples

        with patch("backend.renderer.select_tracking_engine", return_value={
            "requested": "auto", "primary": "yunet", "fallback": None, "device": "cpu", "available": ["yunet"]
        }):
            traj = compute_smart_crop_trajectory("dummy.mp4", 0.0, 4.0)

        pos = np.asarray(traj["positions"])
        # Paruh awal (pembicara kiri berbicara) TIDAK boleh tertimpa oleh backfill ke kanan!
        self.assertLess(float(np.mean(pos[:50])), 220.0)
        # Paruh akhir mengunci pembicara kanan (490*3 - 303 = 1167)
        self.assertGreater(float(np.mean(pos[90:])), 1050.0)
        # Virtual cut: tidak boleh melayang lama di area tengah meja (580..740)
        center_frames = int(np.sum((pos >= 580.0) & (pos <= 740.0)))
        self.assertEqual(center_frames, 0)

    @patch("backend.renderer._probe_video_stream", return_value=(1920, 1080, 30.0))
    @patch("backend.renderer._iter_ffmpeg_frames")
    @patch("backend.renderer._get_yunet_detector")
    @patch("backend.renderer._detect_faces_yunet")
    def test_multi_sample_burst_distraction_is_suppressed(
        self, mock_detect_yunet, mock_get_yunet, mock_iter_frames, _mock_probe
    ):
        # 45 frame = 9 sampel: 3 sampel berturut-turut melompat ke objek kanan lalu kembali ke pembicara kiri
        frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(45)]
        mock_iter_frames.return_value = iter(frames)
        mock_get_yunet.return_value = MagicMock()
        real_face = {"cx": 150.0, "cy": 140.0, "w": 65.0, "h": 82.0, "area": 5330.0, "score": 0.94, "engine": "yunet"}
        distraction = {"cx": 490.0, "cy": 150.0, "w": 95.0, "h": 120.0, "area": 11400.0, "score": 0.91, "engine": "yunet"}
        mock_detect_yunet.side_effect = [
            [real_face], [real_face], [real_face],
            [distraction], [distraction], [distraction],
            [real_face], [real_face], [real_face],
        ]
        with patch("backend.renderer.select_tracking_engine", return_value={
            "requested": "auto", "primary": "yunet", "fallback": None, "device": "cpu", "available": ["yunet"]
        }):
            traj = compute_smart_crop_trajectory("dummy.mp4", 0.0, 1.5)
        self.assertLess(float(np.max(traj["positions"])), 200.0)

    @patch("backend.renderer._probe_video_stream", return_value=(1920, 1080, 30.0))
    @patch("backend.renderer._iter_ffmpeg_frames")
    @patch("backend.renderer._get_yunet_detector")
    @patch("backend.renderer._detect_faces_yunet")
    def test_adaptive_fit_blur_zoom_uses_1x_for_broll_and_fits_wide_two_speakers(
        self, mock_detect_yunet, mock_get_yunet, mock_iter_frames, _mock_probe
    ):
        mock_get_yunet.return_value = MagicMock()

        # Kasus A: murni B-roll / presentasi tanpa wajah -> fit_blur_zoom = 1.0x (utuh, tidak potong tengah)
        mock_iter_frames.return_value = iter([np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(30)])
        mock_detect_yunet.return_value = []
        with patch("backend.renderer.select_tracking_engine", return_value={
            "requested": "auto", "primary": "yunet", "fallback": None, "device": "cpu", "available": ["yunet"]
        }):
            traj_broll = compute_smart_crop_trajectory("dummy.mp4", 0.0, 1.0)
        self.assertEqual(traj_broll["shot_modes"], [{"start": 0, "end": 30, "mode": "fit_blur"}])
        self.assertAlmostEqual(traj_broll["fit_blur_zoom"], 1.0)

        # Kasus B: Wide 2-shot (kiri cx=110, kanan cx=530 di 640w) -> zoom <= 1.25x agar kedua pembicara utuh
        mock_iter_frames.return_value = iter([np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(30)])
        mock_detect_yunet.return_value = [
            {"cx": 110.0, "cy": 150.0, "w": 36.0, "h": 46.0, "area": 1656.0, "score": 0.90, "engine": "yunet"},
            {"cx": 530.0, "cy": 150.0, "w": 36.0, "h": 46.0, "area": 1656.0, "score": 0.90, "engine": "yunet"},
        ]
        with patch("backend.renderer.select_tracking_engine", return_value={
            "requested": "auto", "primary": "yunet", "fallback": None, "device": "cpu", "available": ["yunet"]
        }):
            traj_wide = compute_smart_crop_trajectory("dummy.mp4", 0.0, 1.0)
        self.assertEqual(traj_wide["shot_modes"], [{"start": 0, "end": 30, "mode": "fit_blur"}])
        self.assertLessEqual(traj_wide["fit_blur_zoom"], 1.25)

    @patch("backend.renderer._probe_video_stream", return_value=(1080, 1920, 30.0))
    @patch("backend.renderer._iter_ffmpeg_frames")
    def test_native_vertical_video_uses_passthrough_without_fit_blur(
        self, mock_iter_frames, _mock_probe
    ):
        traj = compute_smart_crop_trajectory("vertical_source.mp4", 0.0, 2.0)
        self.assertEqual(traj["engine"], "native_vertical")
        self.assertEqual(traj["shot_modes"], [{"start": 0, "end": 60, "mode": "smart_crop"}])
        self.assertEqual(len(traj["positions"]), 60)
        self.assertTrue(np.all(np.asarray(traj["positions"]) == 0.0))
        mock_iter_frames.assert_not_called()

    @patch("backend.renderer._probe_video_stream", return_value=(1920, 1080, 30.0))
    @patch("backend.renderer._iter_ffmpeg_frames")
    @patch("backend.renderer._get_yunet_detector")
    @patch("backend.renderer._detect_faces_yunet")
    def test_empty_first_shot_uses_speaker_anchor_instead_of_center_table(
        self, mock_detect_yunet, mock_get_yunet, mock_iter_frames, _mock_probe
    ):
        # Frame 0..19 gelap/kosong, cut di frame 20 ke pembicara kiri (cx=150 -> 450px)
        f_a = np.zeros((360, 640, 3), dtype=np.uint8)
        f_b = np.full((360, 640, 3), 180, dtype=np.uint8)
        frames = [f_a.copy() for _ in range(20)] + [f_b.copy() for _ in range(25)]
        mock_iter_frames.return_value = iter(frames)
        mock_get_yunet.return_value = MagicMock()
        left_face = {"cx": 150.0, "cy": 140.0, "w": 70.0, "h": 90.0, "area": 6300.0, "score": 0.95, "engine": "yunet"}
        # Sampel di frame 0, 5, 10, 15 (kosong), lalu frame 20 (cut), 20%5==0, 25, 30, 35, 40 (left_face)
        mock_detect_yunet.side_effect = [[], [], [], [], [left_face], [left_face], [left_face], [left_face], [left_face]]
        with patch("backend.renderer.select_tracking_engine", return_value={
            "requested": "auto", "primary": "yunet", "fallback": None, "device": "cpu", "available": ["yunet"]
        }):
            traj = compute_smart_crop_trajectory("dummy.mp4", 0.0, 1.5)
        # Tidak pernah jatuh ke tengah meja (657px); seluruh trajektori memakai anchor kiri (~147px)
        self.assertLess(float(np.max(traj["positions"])), 220.0)

    def test_dark_studio_localized_camera_cut_is_detected(self):
        # 80% latar studio gelap identik, hanya blok pembicara pindah dari kiri ke kanan di frame 25
        cam_left = np.full((90, 160, 3), (15, 15, 20), dtype=np.uint8)
        cam_left[20:75, 15:50] = (95, 130, 185)
        cam_right = np.full((90, 160, 3), (15, 15, 20), dtype=np.uint8)
        cam_right[20:75, 110:145] = (95, 130, 185)
        det = SceneCutDetector(fps=30)
        cuts = [i for i in range(50) if det.update(i, cam_left if i < 25 else cam_right)]
        self.assertEqual(cuts, [25])

    def test_yolo_accepts_wide_closeup_solo_speaker_and_rejects_low_table_clutter(self):
        fake_yolo = MagicMock()
        fake_res = MagicMock()
        # Box 1: objek bawah meja / kaki (by1=230 > 0.58*360=208.8) -> harus dibuang
        # Box 2: solo close-up bahu lebar (120..500 = 380px = 59.4% > 50% lama) + kepala valid -> diterima!
        fake_res.boxes.__len__.return_value = 2
        fake_res.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([
            [200.0, 230.0, 340.0, 355.0],
            [120.0, 30.0, 500.0, 355.0],
        ])
        fake_res.boxes.conf.cpu.return_value.numpy.return_value = np.array([0.85, 0.92])
        fake_yolo.predict.return_value = [fake_res]

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        with patch("backend.renderer._refine_person_head_with_yunet", return_value={"cx": 305.0, "cy": 105.0, "w": 82.0, "h": 104.0}):
            dets = _detect_persons_yolo(fake_yolo, frame, 640, 360, device="cpu", yunet_detector=MagicMock())
        self.assertEqual(len(dets), 1)
        self.assertAlmostEqual(dets[0]["cx"], 305.0)



class NaturalPaddingTest(unittest.TestCase):
    def test_tail_does_not_leak_into_next_word(self):
        # Kata saat ini selesai di 20.00, kata berikutnya mulai di 20.05 (gap cuma 50ms)
        segments = [
            {"words": [{"word": "terakhir", "start": 18.0, "end": 20.0}]},
            {"words": [{"word": "berikutnya", "start": 20.05, "end": 21.0}]},
        ]
        lead, tail = _natural_padding(segments, clip_start=18.0, clip_end=20.0)
        # 20.0 + tail HARUS kurang dari 20.05 (tidak boleh menabrak kata berikutnya)
        self.assertLess(20.0 + tail, 20.05)
        self.assertGreaterEqual(tail, 0.0)

    def test_tail_allows_natural_decay_when_silence_follows(self):
        # Ada jeda hening panjang setelah 20.0 (kata berikutnya baru di 25.0)
        segments = [
            {"words": [{"word": "terakhir", "start": 18.0, "end": 20.0}]},
            {"words": [{"word": "berikutnya", "start": 25.0, "end": 26.0}]},
        ]
        lead, tail = _natural_padding(segments, clip_start=18.0, clip_end=20.0)
        self.assertGreater(tail, 0.1)
        self.assertLessEqual(tail, 0.50)


if __name__ == "__main__":
    unittest.main()
