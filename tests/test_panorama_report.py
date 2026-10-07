"""Real-video and rendered-output checks; no pose/model inference is needed."""
import copy
import importlib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont


def api():
    return importlib.import_module("scoring.panorama_report")


def make_pose(n=11):
    lm = np.full((n, 33, 3), np.nan)
    vis = np.zeros((n, 33))
    for joint, xy in {2: (.48, .22), 5: (.52, .22), 11: (.40, .36),
                      12: (.60, .36), 23: (.43, .60), 24: (.57, .60),
                      25: (.42, .84), 26: (.58, .84)}.items():
        lm[:, joint] = [*xy, 0]
        vis[:, joint] = 1
    return dict(n=n, width=240, height=160, lm=lm, world=lm.copy(), vis=vis)


def make_report(name="sample.avi", n=11):
    return dict(video=dict(name=name, fps=10.0, frames=n), metrics=dict(
        headup=dict(raw=12.345678901, unit="deg", score=4.126, reason=None,
                    valid_events=3, valid_frames=99),
        trunk=dict(raw=148.123456789, unit="deg", score=8.765, reason=None,
                   valid_frames=n),
        shoulder=dict(raw=37.123456789, unit="deg/s", score=6.321, reason=None,
                      valid_pairs=10)), total=7.123456789,
        weights=dict(headup=.1, trunk=.8, shoulder=.1),
        quality=dict(provisional=True, warnings=["Inspect subject identity."]))


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "실제 frame # clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (240, 160))
    assert writer.isOpened(), "This test needs OpenCV's bundled MJPEG encoder"
    colors = []
    for i in range(11):
        rgb = (40 + i * 14, 60 + i * 8, 100 + i * 10)
        colors.append(rgb)
        frame = np.full((160, 240, 3), rgb[::-1], np.uint8)
        cv2.circle(frame, (120, 80), 12, (200, 25, 200), -1)
        writer.write(frame)
    writer.release()
    return path, colors


@pytest.fixture
def drawn_text(monkeypatch):
    """Observe actual text drawing while leaving Pillow's rasterization intact."""
    result = []
    original = ImageDraw.ImageDraw.text

    def text(draw, xy, value, *args, **kwargs):
        if draw._image.width > 1:
            result.append(str(value))
        return original(draw, xy, value, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", text)
    return result


def render_item(video, pose=None, report=None):
    return dict(video_path=video[0], pose=pose if pose is not None else make_pose(),
                report=report if report is not None else make_report(video[0].name))


def color_extent(pixels, rgb):
    mask = np.max(np.abs(pixels.astype(float) - rgb), axis=-1) < 8
    yy, xx = np.nonzero(mask)
    assert len(xx) > 80, f"Missing source color {rgb}"
    return xx.min(), yy.min(), xx.max(), yy.max()


def test_six_real_panels_preserve_source_clock_through_missing_poses(video, tmp_path, drawn_text):
    pose = make_pose()
    pose["lm"][[2, 6, 10]] = np.nan
    pose["vis"][[2, 6, 10]] = 0
    target = tmp_path / "new folder" / "panorama.png"
    api().render_panorama([render_item(video, pose)], target)
    with Image.open(target) as figure:
        pixels = np.array(figure.convert("RGB"))
        assert figure.info["dpi"][0] >= 299
        assert tuple(pixels[0, 0]) == (255, 255, 255)
    extents = [color_extent(pixels, video[1][i]) for i in [0, 2, 4, 6, 8, 10]]
    assert all(left[2] < right[0] for left, right in zip(extents, extents[1:]))
    text = "\n".join(drawn_text)
    for frame, seconds in [(0, "0.00"), (2, "0.20"), (4, "0.40"),
                           (6, "0.60"), (8, "0.80"), (10, "1.00")]:
        assert any(f"#{frame}" in line and f"{seconds} s" in line for line in drawn_text)
    assert text.count("No pose") + text.count("포즈 없음") == 3
    assert "4.13" in text and "8.77" in text and "6.32" in text and "7.12" in text
    assert "schematic" in text.lower() or "개략" in text
    assert "provisional" in text.lower() or "잠정" in text
    assert "pelvis" not in text.lower()


def test_blank_pose_keeps_picture_aspect_and_has_no_geometry(video, tmp_path, drawn_text):
    pose = make_pose()
    pose["lm"][:] = np.nan
    pose["vis"][:] = np.nan
    path = tmp_path / "blank.png"
    api().render_panorama([render_item(video, pose)], path)
    pixels = np.array(Image.open(path).convert("RGB"))
    text = "\n".join(drawn_text)
    assert text.count("No pose") + text.count("포즈 없음") == 6
    for i in [0, 2, 4, 6, 8, 10]:
        x0, y0, x1, y1 = color_extent(pixels, video[1][i])
        # Full 240x160 source must be contained, never stretched to portrait.
        assert abs((x1 - x0 + 1) / (y1 - y0 + 1) - 1.5) < .06
        patch = pixels[y0:y1 + 1, x0:x1 + 1]
        for rgb in [(209, 49, 61), (37, 99, 190), (235, 180, 20)]:
            assert not np.all(patch == rgb, axis=-1).any()


def test_valid_geometry_is_colored_and_does_not_mutate_pose(video, tmp_path):
    item = render_item(video)
    before = copy.deepcopy(item)
    target = tmp_path / "overlay.png"
    api().render_panorama([item], target)
    pixels = np.array(Image.open(target).convert("RGB"))
    # Colors must occur inside the actual first source panel, not just a legend.
    x0, y0, x1, y1 = color_extent(pixels, video[1][0])
    patch = pixels[y0:y1 + 1, x0:x1 + 1]
    for color in [(209, 49, 61), (37, 99, 190), (235, 180, 20)]:
        assert np.all(patch == color, axis=-1).sum() > 15
    for field in ["lm", "world", "vis"]:
        np.testing.assert_equal(item["pose"][field], before["pose"][field])
    assert item["report"] == before["report"]


def test_invisible_or_nonfinite_geometry_is_not_drawn(video, tmp_path):
    pose = make_pose()
    pose["vis"][:, 11] = .1
    pose["lm"][:, 2, 0] = np.inf
    pose["lm"][:, 0] = [1e12, -1e12, 0]  # invisible outlier cannot steer crop
    target = tmp_path / "partial.png"
    api().render_panorama([render_item(video, pose)], target)
    pixels = np.array(Image.open(target).convert("RGB"))
    x0, y0, x1, y1 = color_extent(pixels, video[1][0])
    patch = pixels[y0:y1 + 1, x0:x1 + 1]
    assert not np.all(patch == (209, 49, 61), axis=-1).any()
    assert not np.all(patch == (235, 180, 20), axis=-1).any()
    assert np.all(patch == (37, 99, 190), axis=-1).sum() > 15


def test_multiple_rows_and_missing_scores_are_kept(video, tmp_path, drawn_text):
    report = make_report("second")
    report["metrics"]["headup"].update(raw=None, score=None, reason="Insufficient valid touches")
    report["total"] = None
    report["quality"]["warnings"].append("Pose coverage low: 6/11 source frames.")
    path = tmp_path / "rows.png"
    api().render_panorama([render_item(video), render_item(video, report=report)], path)
    text = "\n".join(drawn_text)
    assert text.count("#0") == 2
    assert "N/A" in text and "Insufficient valid touches" in text
    assert "Pose coverage low: 6/11 source frames." in text
    assert Image.open(path).height > 950


def test_single_requested_frame_and_short_clip_repeat_real_frames(video, tmp_path, drawn_text):
    api().render_panorama([render_item(video)], tmp_path / "one.png", n_frames=1)
    assert sum("#0" in s for s in drawn_text) == 1
    drawn_text.clear()
    api().render_panorama([render_item(video)], tmp_path / "repeat.png", n_frames=12)
    labels = [s for s in drawn_text if s.startswith("#")]
    assert len(labels) == 12
    assert labels[0].startswith("#0 ") and labels[-1].startswith("#10 ")
    assert "repeat" in " ".join(drawn_text).lower() or "반복" in " ".join(drawn_text)


@pytest.mark.parametrize("n_frames", [0, -2, 2.5, True, "6"])
def test_bad_panel_count_is_rejected_before_writing(video, tmp_path, n_frames):
    target = tmp_path / "unchanged.png"
    target.write_bytes(b"keep existing result")
    with pytest.raises(ValueError, match="n_frames"):
        api().render_panorama([render_item(video)], target, n_frames=n_frames)
    assert target.read_bytes() == b"keep existing result"


def test_missing_or_truncated_video_fails_explicitly_without_replacing_output(video, tmp_path):
    target = tmp_path / "previous.png"
    target.write_bytes(b"keep")
    missing = render_item(video)
    missing["video_path"] = tmp_path / "missing.avi"
    with pytest.raises((OSError, ValueError), match="(?i)(open|read|video)"):
        api().render_panorama([missing], target)
    item = render_item(video, pose=make_pose(14), report=make_report(n=14))
    with pytest.raises((OSError, ValueError), match="(?i)(frame|decode|read)"):
        api().render_panorama([item], target)
    assert target.read_bytes() == b"keep"


@pytest.mark.parametrize("change", ["fps", "frames", "shape", "dimensions", "empty"])
def test_invalid_metadata_or_pose_is_not_silently_compacted(video, tmp_path, change):
    item = render_item(video)
    items = [item]
    if change == "fps":
        item["report"]["video"]["fps"] = float("nan")
    elif change == "frames":
        item["report"]["video"]["frames"] = 7
    elif change == "shape":
        item["pose"]["lm"] = item["pose"]["lm"][:3]
    elif change == "dimensions":
        item["pose"]["width"] = 0
    else:
        items = []
    with pytest.raises(ValueError):
        api().render_panorama(items, tmp_path / "invalid.png")


def test_font_override_and_english_fallback_keep_nonlatin_filenames(video, tmp_path, drawn_text, monkeypatch):
    module = api()
    # Force the documented no-Korean-font branch; the actual rasterizer still runs.
    monkeypatch.setattr(module, "_find_font", lambda: None)
    module.render_panorama([render_item(video)], tmp_path / "한글 output.png")
    text = " ".join(drawn_text)
    assert "Head-up" in text and "Shoulder speed" in text and "Total" in text
    assert "DRIBBLING MOTION" in text
    assert "Raw: 12.35 deg" in text and "Raw: 37.12 deg/s" in text
    assert "Valid events: 3" in text and "Valid frames: 11 / 11" in text
    assert "Valid frame pairs: 10" in text
    assert "Video 1" in text
    with pytest.raises(ValueError, match="font"):
        module.render_panorama([render_item(video)], tmp_path / "font.png",
                               font_path=tmp_path / "not-a-font.ttf")
    font_path = tmp_path / "pillow-test-font.ttf"
    font_path.write_bytes(ImageFont.load_default(size=20).path.getvalue())
    module.render_panorama([render_item(video)], tmp_path / "custom.png", font_path=font_path)
    assert (tmp_path / "custom.png").is_file()


def test_korean_font_localizes_panorama_and_preserves_sample_counts(video, tmp_path, drawn_text):
    font = api()._find_font()
    if font is None:
        pytest.skip("No local Korean font; English fallback is covered independently")
    report = make_report("한국어 원본 동영상.avi")
    report["metrics"]["headup"]["candidate_events"] = 7
    api().render_panorama([render_item(video, report=report)], tmp_path / "한국어.png", font_path=font)
    text = " ".join(drawn_text)
    for label in ["드리블 동작 분석", "헤드업", "몸통 각도", "어깨 회전 속도", "총점", "점수 / 10"]:
        assert label in text
    assert "한국어 원본 동영상.avi" in text
    assert "측정값: 12.35 deg" in text and "측정값: 37.12 deg/s" in text
    assert "유효 사건: 3 / 후보 7" in text
    assert "유효 프레임: 11 / 11" in text and "유효 프레임 쌍: 10" in text
    assert "고정 가중치" in text
    assert "DRIBBLING MOTION" not in text and "SCORE / 10" not in text


def test_panorama_summarizes_only_known_generic_warnings_and_html_keeps_all(video, tmp_path, drawn_text, monkeypatch):
    monkeypatch.setattr(api(), "_find_font", lambda: None)
    generic = [
        "Provisional frozen relative scores; not validated skill grades. No grade training or per-video refitting.",
        "Automatic numeric masks do not validate subject identity or left/right landmark identity; inspect the source video.",
        "Head ball proxies retain fixed pixel thresholds (80 px jumps, 40 px/frame interpolation, 200 px ankle distance, 10 px prominence); source width 240 px and camera framing can change acceptance.",
        "Head ranges include upward and downward motion; ball direction changes are proxies, not confirmed contacts or gaze.",
    ]
    issue = "Suspected subject switch at source frame #6."
    reason = "Insufficient valid touches"
    reports = [make_report("first"), make_report("second")]
    for report in reports:
        report["metrics"]["headup"].update(raw=None, score=None, valid_events=0, reason=reason)
        report["total"] = None
        report["quality"]["warnings"] = generic + [issue, issue, "headup: " + reason]
    before = copy.deepcopy(reports)
    path = tmp_path / "concise.png"
    api().render_panorama([render_item(video, report=r) for r in reports], path)
    text = " ".join(drawn_text)
    for warning in generic:
        assert warning not in text
    assert text.count(issue) == 2  # Still present for each affected video.
    assert text.count(reason) == 2  # In the metric panels, without duplicate row notes.
    assert text.count("not validated skill grades") == 1
    assert text.count("Fixed weights") == 2
    for meaning in ["subject", "left/right", "pixel", "framing", "contacts", "gaze", "schematic"]:
        assert meaning in text
    assert text.count("Valid events: 0") == 2
    assert reports == before
    html = tmp_path / "full.html"
    api().write_html_report(reports, path, html)
    html_text = " ".join(ParsedHTML(html.read_text(encoding="utf-8")).words)
    for warning in generic + [issue, "headup: " + reason]:
        assert html_text.count(warning) >= 2


def test_unrecognized_warning_with_generic_prefix_is_retained(video, tmp_path, drawn_text):
    warning = "Automatic numeric masks do not validate subject identity: possible switch at frame #4."
    report = make_report()
    report["quality"]["warnings"] = [warning]
    api().render_panorama([render_item(video, report=report)], tmp_path / "unknown-warning.png")
    assert warning in " ".join(drawn_text)


def test_missing_sample_count_displays_na_instead_of_zero(video, tmp_path, drawn_text, monkeypatch):
    monkeypatch.setattr(api(), "_find_font", lambda: None)
    report = make_report()
    report["metrics"]["headup"].pop("valid_events")
    report["metrics"]["trunk"].pop("valid_frames")
    report["metrics"]["shoulder"].pop("valid_pairs")
    api().render_panorama([render_item(video, report=report)], tmp_path / "missing-counts.png")
    assert " ".join(drawn_text).count("Valid samples: N/A") == 3


def test_long_korean_names_and_na_reasons_fit_inside_rendered_rows(video, tmp_path, monkeypatch):
    report = make_report("공격수의 드리블 측정 장면 및 전체 원본 동영상 " * 8)
    reason = "측정 불가: 유효한 어깨 좌표와 연속 프레임이 부족합니다. " * 8
    for metric in report["metrics"].values():
        metric.update(raw=None, score=None, reason=reason)
    report["total"] = None
    report["quality"]["warnings"] = [reason, "Inspect the complete source clip. " * 12]
    original = ImageDraw.ImageDraw.text
    actual_bounds = []

    def record(draw, xy, text, *args, **kwargs):
        # Ignore the 1x1 measurement surface; assert on real row/figure drawing.
        if draw._image.width > 1:
            box = draw.textbbox(xy, text, font=kwargs.get("font"), anchor=kwargs.get("anchor"))
            actual_bounds.append((box, draw._image.size, str(text)))
        return original(draw, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record)
    path = tmp_path / "long-korean.png"
    api().render_panorama([render_item(video, report=report)], path)
    assert actual_bounds
    for (left, top, right, bottom), (width, height), text in actual_bounds:
        assert 0 <= left <= right <= width, text
        assert 0 <= top <= bottom <= height, text
    assert Image.open(path).height > 1100


def test_actual_video_count_and_dimensions_are_checked(video, tmp_path):
    item = render_item(video, pose=make_pose(7), report=make_report(n=7))
    with pytest.raises(ValueError, match="frames"):
        api().render_panorama([item], tmp_path / "short-metadata.png")
    item = render_item(video)
    item["pose"]["width"] = 480
    with pytest.raises(ValueError, match="dimensions"):
        api().render_panorama([item], tmp_path / "wrong-dimensions.png")


def test_nonfinite_report_values_remain_na(video, tmp_path, drawn_text):
    report = make_report()
    for metric in report["metrics"].values():
        metric.update(raw=float("nan"), score=float("inf"), reason="Unavailable")
    report["total"] = float("nan")
    image = tmp_path / "nonfinite.png"
    api().render_panorama([render_item(video, report=report)], image)
    path = tmp_path / "nonfinite.html"
    api().write_html_report([report], image, path)
    text = " ".join(drawn_text)
    assert text.count("N/A") >= 7
    parsed = ParsedHTML(path.read_text(encoding="utf-8"))
    assert "N/A" in parsed.words
    assert not any(value in {"nan", "inf"} for value in parsed.words)


class ParsedHTML(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.elements = []
        self.words = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.words.append(data)


def test_html_escapes_all_user_fields_and_links_image_offline(tmp_path):
    folder = tmp_path / "figures"
    folder.mkdir()
    image = folder / '한글 # 100% " &.png'
    Image.new("RGB", (20, 20), "white").save(image)
    report = make_report('<script>alert("video")</script> & 한글')
    report["metrics"]["headup"].update(score=None, raw=None,
        reason='<img src=x onerror="alert(1)">')
    report["metrics"]["shoulder"]["unit"] = "<svg onload=alert(2)>"
    report["quality"]["warnings"] = ['<script>alert("warning")</script>']
    report["total"] = None
    target = tmp_path / "report folder" / "index.html"
    api().write_html_report([report], image, target)
    parsed = ParsedHTML(target.read_text(encoding="utf-8"))
    assert not any(tag in {"script", "svg", "iframe", "link"} for tag, _ in parsed.elements)
    images = [attrs for tag, attrs in parsed.elements if tag == "img"]
    assert len(images) == 1
    src = images[0]["src"]
    assert not urlsplit(src).scheme and not src.startswith("/")
    assert (target.parent / unquote(src)).resolve() == image.resolve()
    assert "%23" in src and "%25" in src and "%22" in src
    text = " ".join(parsed.words)
    assert report["video"]["name"] in text
    assert report["metrics"]["headup"]["reason"] in text
    assert report["quality"]["warnings"][0] in text
    assert "148.123456789" in text and "37.123456789" in text
    assert "8.765" in text and "6.321" in text and "N/A" in text
    assert "provisional" in text.lower() and "fixed" in text.lower()
    assert "0.1" in text and "0.8" in text
    assert "valid_frames" in text and "valid_pairs" in text


def test_html_keeps_each_report_exact_scores_weights_and_quality(tmp_path):
    image = tmp_path / "figure.png"
    Image.new("RGB", (2, 2)).save(image)
    first, second = make_report("one"), make_report("two")
    second["weights"] = dict(headup=.3, trunk=.6, shoulder=.1)
    second["quality"]["pose_fraction"] = .123456789
    second["quality"]["warnings"] = []
    target = tmp_path / "page.html"
    api().write_html_report([first, second], image, target)
    parsed = ParsedHTML(target.read_text(encoding="utf-8"))
    text = " ".join(parsed.words)
    assert "one" in text and "two" in text
    assert "7.123456789" in text and "12.345678901" in text
    assert "pose_fraction" in text and "0.123456789" in text
    assert "0.3" in text and "0.6" in text
    assert sum(tag == "table" for tag, _ in parsed.elements) >= 1


def test_html_requires_reports_and_existing_image_without_overwrite(tmp_path):
    target = tmp_path / "existing.html"
    target.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError):
        api().write_html_report([], tmp_path / "none.png", target)
    with pytest.raises((ValueError, OSError)):
        api().write_html_report([make_report()], tmp_path / "none.png", target)
    assert target.read_text() == "keep"


def test_output_never_overwrites_source_video_or_image(video, tmp_path):
    before = video[0].read_bytes()
    with pytest.raises(ValueError):
        api().render_panorama([render_item(video)], video[0])
    assert video[0].read_bytes() == before
    image = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(image)
    before = image.read_bytes()
    with pytest.raises(ValueError):
        api().write_html_report([make_report()], image, image)
    assert image.read_bytes() == before
