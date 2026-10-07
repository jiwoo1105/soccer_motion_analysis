"""Offline, model-free figures and HTML for the provisional three-metric report.

Landmarks are normalized source-image coordinates on the uncompressed source
frame clock. The 2D drawing is schematic: measurements and scores are taken
verbatim from ``report``, never estimated from these projected lines. Only
OpenCV, NumPy and Pillow are required; no models or scoring modules are loaded.
"""
from collections.abc import Mapping
from functools import lru_cache
from html import escape
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re
import tempfile
import unicodedata
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin


_METRICS = (("headup", "Head-up"), ("trunk", "Trunk"),
            ("shoulder", "Shoulder speed"))
_RED = (209, 49, 61)
_BLUE = (37, 99, 190)
_YELLOW = (235, 180, 20)
_INK = (27, 35, 43)
_MUTED = (88, 97, 105)
_RULE = (218, 224, 229)
_TILE = (232, 348)
_NOTICE = ("Provisional scores; not validated skill grades. "
           "Inspect subject identity and pose quality.")
_LABELS = {
    "title": ("DRIBBLING MOTION", "드리블 동작 분석"),
    "subtitle": ("Source frames / pose overlays / provisional relative scores",
                 "원본 프레임 / 자세 오버레이 / 잠정 상대 점수"),
    "headup": ("Head-up", "헤드업"),
    "trunk": ("Trunk", "몸통 각도"),
    "shoulder": ("Shoulder speed", "어깨 회전 속도"),
    "score": ("SCORE / 10", "점수 / 10"),
    "raw": ("Raw", "측정값"),
    "reason": ("Reason", "사유"),
    "unavailable": ("Measurement unavailable", "측정 불가"),
    "total": ("Total", "총점"),
    "total_na": ("Total N/A: all three component scores are required.",
                 "총점 N/A: 세 지표의 점수가 모두 필요합니다."),
    "no_pose": ("No pose", "포즈 없음"),
    "weights": ("Fixed weights", "고정 가중치"),
    "quality": ("Quality", "품질"),
    "repeat": ("Short clip: source frames repeat to fill all panels.",
               "짧은 영상: 패널 수를 채우기 위해 원본 프레임을 반복 표시했습니다."),
    "events": ("Valid events", "유효 사건"),
    "candidates": ("candidates", "후보"),
    "frames": ("Valid frames", "유효 프레임"),
    "pairs": ("Valid frame pairs", "유효 프레임 쌍"),
    "samples": ("Valid samples", "유효 표본"),
    "limitation": ("Provisional relative scores; not validated skill grades. Inspect subject, left/right identity and pose quality.",
                   "잠정 상대 점수이며 검증된 실력 등급이 아닙니다. 대상 선수·좌우 관절·포즈를 확인하세요."),
    "head_limitation": ("Head-up uses fixed pixel thresholds and is sensitive to framing; ball events do not confirm contacts or gaze, and head ranges include both motion directions.",
                        "헤드업은 고정 픽셀 기준과 영상 구도에 영향을 받습니다. 공 이동은 접촉·시선을 확정하지 않으며, 머리 각도 범위는 상·하 움직임을 모두 포함합니다."),
    "clock": ("Source frames: zero-based indices; time = frame / source fps. Missing poses retain the real image.",
              "원본 프레임 번호는 0부터 시작하며 시간 = 프레임 / fps입니다. 포즈가 없어도 원본 이미지를 유지합니다."),
    "overlay": ("2D overlays are schematic, not measured 3D angles. Red: shoulders; blue: trunk/hips/knees; yellow: head vector.",
                "2D 선은 개략 표시이며 측정된 3D 각도가 아닙니다. 빨강: 어깨 · 파랑: 몸통/엉덩이/무릎 · 노랑: 머리 방향"),
}
# Only these known boilerplate messages are summarized in the shared footer.
# Never discard an unfamiliar warning merely because it repeats across videos.
_COMMON_WARNINGS = {
    "Inspect subject identity.",
    "Provisional frozen relative scores; not validated skill grades. No grade training or per-video refitting.",
    "Automatic numeric masks do not validate subject identity or left/right landmark identity; inspect the source video.",
    "Head ranges include upward and downward motion; ball direction changes are proxies, not confirmed contacts or gaze.",
}
_PIXEL_WARNING = re.compile(
    re.escape("Head ball proxies retain fixed pixel thresholds (80 px jumps, 40 px/frame interpolation, "
              "200 px ankle distance, 10 px prominence); source width ")
    + r"[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?"
    + re.escape(" px and camera framing can change acceptance.")
)


def _find_font():
    """Find a locally installed Korean font; no downloading or redistribution."""
    candidates = [
        Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
        Path("/Library/Fonts/AppleGothic.ttf"),
    ]
    for root in (Path("/Library/Fonts"), Path.home() / "Library/Fonts",
                 Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
                 Path.home() / ".local/share/fonts", Path("C:/Windows/Fonts")):
        if root.is_dir():
            candidates.extend(sorted(root.rglob("NotoSans*CJK*.*")))
            candidates.extend(sorted(root.rglob("NotoSans*KR*.*")))
    for path in candidates:
        if path.is_file() and path.suffix.lower() in {".ttf", ".otf", ".ttc"}:
            try:
                ImageFont.truetype(str(path), 20)
                return path
            except OSError:
                continue
    return None


class _Fonts:
    def __init__(self, path):
        explicit = path is not None
        selected = path if explicit else _find_font()
        self.path = str(selected) if selected is not None else None
        if self.path is not None:
            try:
                font = ImageFont.truetype(self.path, 20)
            except (OSError, ValueError) as exc:
                raise ValueError(f"Cannot load font: {self.path}") from exc
        else:
            for candidate in ("DejaVuSans.ttf",
                              "/System/Library/Fonts/Supplemental/Arial.ttf"):
                try:
                    font = ImageFont.truetype(candidate, 20)
                    self.path = candidate
                    break
                except OSError:
                    continue
            else:
                font = ImageFont.load_default(size=20)
        self.korean = bytes(font.getmask("가")) != bytes(font.getmask("\U0010ffff"))

    @lru_cache(maxsize=12)
    def at(self, size):
        return (ImageFont.truetype(self.path, size) if self.path else
                ImageFont.load_default(size=size))

    def text(self, value):
        value = unicodedata.normalize("NFC", str(value))
        if not self.korean:
            # Preserve reasons in a readable escaped form instead of empty boxes.
            value = value.encode("ascii", "backslashreplace").decode("ascii")
        return value

    def label(self, key):
        return _LABELS[key][int(self.korean)]


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def _value(value, decimals=None):
    """Exact numeric text for HTML, fixed precision only when requested by PNG."""
    if value is None or isinstance(value, Real) and not math.isfinite(value):
        return "N/A"
    if decimals is not None and _finite(value):
        return f"{value:.{decimals}f}"
    return str(value)


def _validate_report(report):
    if not isinstance(report, Mapping) or not isinstance(report.get("video"), Mapping):
        raise ValueError("report must contain video metadata")
    video = report["video"]
    if not _finite(video.get("fps")) or video["fps"] <= 0:
        raise ValueError("report.video.fps must be finite and positive")
    count = video.get("frames")
    if not isinstance(count, Integral) or isinstance(count, bool) or count <= 0:
        raise ValueError("report.video.frames must be a positive integer")
    metrics = report.get("metrics", {})
    if not isinstance(metrics, Mapping) or any(
            not isinstance(metrics.get(key), Mapping) for key, _ in _METRICS):
        raise ValueError("report.metrics must contain headup, trunk and shoulder")


def _validate_pose(pose, count):
    if not isinstance(pose, Mapping) or pose.get("n") != count:
        raise ValueError("pose.n must match report.video.frames on the source clock")
    for key in ("width", "height"):
        if not _finite(pose.get(key)) or pose[key] <= 0:
            raise ValueError(f"pose.{key} must be finite and positive")
    for key, shape in (("lm", (count, 33, 3)), ("vis", (count, 33))):
        if np.asarray(pose.get(key)).shape != shape:
            raise ValueError(f"pose.{key} must have shape {shape}; do not compact missing frames")


def _same_file(left, right):
    return left.resolve() == right.resolve() or (
        left.exists() and right.exists() and os.path.samefile(left, right))


def _read_frames(path, count, indices, pose):
    """Sequential decoding avoids codec seek rounding and valid-pose compaction."""
    capture = cv2.VideoCapture(str(path))
    frames = {}
    try:
        if not capture.isOpened():
            raise OSError(f"Cannot open source video: {path}")
        selected = set(indices)
        for index in range(count):
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                raise OSError(f"Cannot read source frame #{index} of {count} from video: {path}")
            height, width = frame.shape[:2]
            if width != pose["width"] or height != pose["height"]:
                raise ValueError(f"Video dimensions {width}x{height} disagree with pose for {path}")
            if index in selected:
                frames[index] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ok, _ = capture.read()
        if ok:
            raise ValueError(f"Video has more source frames than report.video.frames={count}: {path}")
    finally:
        capture.release()
    return frames


def _panel(frame, landmarks, visibility):
    """Crop around finite visible points, contain with padding, then draw in 2D."""
    points = np.asarray(landmarks, float)
    visibility = np.asarray(visibility, float)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(visibility) & (visibility >= .5)
    xy = points[:, :2] * frame.size
    # Out-of-image predictions do not define a meaningful source-image crop.
    valid &= (xy[:, 0] >= 0) & (xy[:, 0] < frame.width)
    valid &= (xy[:, 1] >= 0) & (xy[:, 1] < frame.height)
    if valid.any():
        lo, hi = xy[valid].min(axis=0), xy[valid].max(axis=0)
        span = np.maximum(hi - lo, [frame.width * .12, frame.height * .25])
        center = (lo + hi) / 2
        crop_width, crop_height = span * [1.55, 1.32]
        crop_width = max(crop_width, crop_height * _TILE[0] / _TILE[1])
        crop_height = max(crop_height, crop_width * _TILE[1] / _TILE[0])
        left = max(0, int(math.floor(center[0] - crop_width / 2)))
        top = max(0, int(math.floor(center[1] - crop_height / 2)))
        right = min(frame.width, int(math.ceil(center[0] + crop_width / 2)))
        bottom = min(frame.height, int(math.ceil(center[1] + crop_height / 2)))
    else:
        left, top, right, bottom = 0, 0, frame.width, frame.height
    cropped = frame.crop((left, top, right, bottom))
    scale = min(_TILE[0] / cropped.width, _TILE[1] / cropped.height)
    size = (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale)))
    resized = cropped.resize(size, Image.Resampling.LANCZOS)
    offset = ((_TILE[0] - size[0]) // 2, (_TILE[1] - size[1]) // 2)
    tile = Image.new("RGB", _TILE, "white")
    tile.paste(resized, offset)
    projected = (xy - [left, top]) * [size[0] / cropped.width, size[1] / cropped.height] + offset
    draw = ImageDraw.Draw(tile)
    drawn = False

    def segment(a, b, color):
        nonlocal drawn
        if valid[a] and valid[b]:
            p, q = tuple(projected[a]), tuple(projected[b])
            draw.line([p, q], fill=color, width=4)
            for x, y in (p, q):
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            drawn = True

    for a, b in ((11, 23), (12, 24), (23, 24), (23, 25), (24, 26)):
        segment(a, b, _BLUE)
    segment(11, 12, _RED)
    if valid[[2, 5, 11, 12]].all():
        start = (projected[11] + projected[12]) / 2
        end = (projected[2] + projected[5]) / 2
        draw.line([tuple(start), tuple(end)], fill=_YELLOW, width=4)
        direction = end - start
        length = np.linalg.norm(direction)
        if length > 0:
            direction /= length
            normal = np.array([-direction[1], direction[0]])
            tip = end - direction * min(11, length / 3)
            draw.polygon([tuple(end), tuple(tip + normal * 4), tuple(tip - normal * 4)], fill=_YELLOW)
        drawn = True
    return tile, drawn


def _wrap(draw, text, font, width):
    """Wrap words, including long paths/tokens, without clipping user reasons."""
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}" if line else word
            if draw.textlength(candidate, font=font) <= width:
                line = candidate
                continue
            if line:
                lines.append(line)
                line = ""
            for char in word:
                if line and draw.textlength(line + char, font=font) > width:
                    lines.append(line)
                    line = ""
                line += char
        lines.append(line)
    return lines


def _paragraph(draw, text, x, y, width, fonts, size=18, color=_MUTED):
    font = fonts.at(size)
    for line in _wrap(draw, fonts.text(text), font, width):
        draw.text((x, y), line, font=font, fill=color)
        y += size + 8
    return y


def _warnings(report):
    quality = report.get("quality", {})
    warnings = quality.get("warnings", []) if isinstance(quality, Mapping) else quality
    if warnings is None:
        return []
    return [str(warnings)] if isinstance(warnings, str) else [str(w) for w in warnings]


def _row_warnings(report):
    """Keep all data-specific issues, without reprinting visible metric reasons."""
    seen = set(_COMMON_WARNINGS)
    for key, _ in _METRICS:
        reason = report["metrics"][key].get("reason")
        if reason:
            seen.update((str(reason), f"{key}: {reason}"))
    result = []
    for warning in _warnings(report):
        if warning not in seen and not _PIXEL_WARNING.fullmatch(warning):
            result.append(warning)
        seen.add(warning)
    return result


def _weights(report, korean=False):
    labels = [(key, _LABELS[key][int(korean)]) for key, _ in _METRICS]
    weights = report.get("weights", {})
    if isinstance(weights, Mapping):
        return [(label, weights.get(key)) for key, label in labels]
    if isinstance(weights, (list, tuple)) and len(weights) == 3:
        return [(label, value) for (_, label), value in zip(labels, weights)]
    return [(label, None) for _, label in labels]


def _sample_text(metric, key, source_frames, fonts):
    count_key, label_key = {
        "headup": ("valid_events", "events"),
        "trunk": ("valid_frames", "frames"),
        "shoulder": ("valid_pairs", "pairs"),
    }[key]
    if count_key not in metric:
        return f"{fonts.label('samples')}: {_value(metric.get('valid_samples'))}"
    text = f"{fonts.label(label_key)}: {_value(metric[count_key])}"
    if key == "headup" and "candidate_events" in metric:
        text += f" / {fonts.label('candidates')} {_value(metric['candidate_events'])}"
    elif key == "trunk":
        text += f" / {_value(metric.get('source_frames', source_frames))}"
    return text


def _metric_panel(draw, report, x, y, width, fonts):
    draw.text((x, y), fonts.label("score"), font=fonts.at(17), fill=_MUTED)
    y += 34
    for key, _ in _METRICS:
        metric = report["metrics"][key]
        draw.text((x, y), fonts.label(key), font=fonts.at(23), fill=_INK)
        score = _value(metric.get("score"), 2)
        draw.text((x + width, y - 4), score, font=fonts.at(29), fill=_INK, anchor="ra")
        y += 34
        y = _paragraph(draw, f"{fonts.label('raw')}: {_value(metric.get('raw'), 2)} {metric.get('unit', '')}",
                       x, y, width, fonts, size=17)
        y = _paragraph(draw, _sample_text(metric, key, report['video']['frames'], fonts),
                       x, y, width, fonts, size=16)
        reason = metric.get("reason")
        if reason or score == "N/A":
            y = _paragraph(draw, f"{fonts.label('reason')}: {reason or fonts.label('unavailable')}", x, y, width, fonts, size=17)
        y += 8
    draw.line((x, y, x + width, y), fill=_RULE, width=2)
    y += 17
    draw.text((x, y), fonts.label("total"), font=fonts.at(26), fill=_INK)
    draw.text((x + width, y - 5), _value(report.get("total"), 2), font=fonts.at(36), fill=_INK, anchor="ra")
    y += 43
    if _value(report.get("total")) == "N/A":
        y = _paragraph(draw, fonts.label("total_na"), x, y, width, fonts, size=17)
    return y


def _atomic_write(path, writer):
    """Finish a sibling temporary file before replacing any existing output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(descriptor)
    temporary = Path(name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render_panorama(items, output_path, n_frames=6, font_path=None):
    """Write a 300-dpi PNG of real source frames and supplied scores; return Path.

    Samples are rounded ``linspace(0, frames-1, n_frames)`` source indices (zero
    based), including both endpoints. Short clips repeat frames with their real
    labels. Missing poses remain in place without overlays. ``lm`` is normalized
    N x 33 x 3 and ``vis`` is N x 33; finite points with visibility >= .5 are used.
    A decoder failure or inconsistent frame count raises rather than omitting a
    panel. ``font_path`` names a local TTF/OTF/TTC font; none is copied or fetched.
    """
    if not isinstance(n_frames, Integral) or isinstance(n_frames, bool) or n_frames <= 0:
        raise ValueError("n_frames must be a positive integer")
    items = list(items)
    if not items:
        raise ValueError("items must contain at least one video")
    output = Path(output_path)
    for item in items:
        _validate_report(item["report"])
        _validate_pose(item["pose"], item["report"]["video"]["frames"])
        if _same_file(output, Path(item["video_path"])):
            raise ValueError("Output must not overwrite a source video")
    fonts = _Fonts(font_path)
    margin, gap, panel_gap, metric_width = 48, 14, 32, 470
    frames_width = n_frames * _TILE[0] + (n_frames - 1) * gap
    width = margin * 2 + frames_width + panel_gap + metric_width
    metric_x = margin + frames_width + panel_gap
    rows = []
    for row_index, item in enumerate(items, 1):
        report, pose = item["report"], item["pose"]
        count, fps = report["video"]["frames"], report["video"]["fps"]
        indices = np.rint(np.linspace(0, count - 1, n_frames)).astype(int).tolist()
        frames = _read_frames(Path(item["video_path"]), count, indices, pose)
        name = unicodedata.normalize("NFC", str(report["video"].get("name", Path(item["video_path"]).name)))
        title = f"{row_index:02d}  {name}"
        if not fonts.korean and not name.isascii():
            title = f"{row_index:02d}  Video {row_index} | " + name.encode("ascii", "replace").decode("ascii")
        # Measure text with the same layout pass used for the final row image.
        measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        title_end = _paragraph(measure, title, margin, 22, width - 2 * margin, fonts, 25, _INK)
        top = title_end + 22
        metric_end = _metric_panel(measure, report, metric_x, top, metric_width, fonts)
        content_end = max(top + _TILE[1] + 58, metric_end)
        notes = [fonts.label("weights") + ": " + "; ".join(
            f"{label} {_value(value)}" for label, value in _weights(report, fonts.korean))]
        notes.extend(f"{fonts.label('quality')}: {warning}" for warning in _row_warnings(report))
        if count < n_frames:
            notes.append(fonts.label("repeat"))
        y = content_end + 18
        for note in notes:
            y = _paragraph(measure, note, margin, y, width - 2 * margin, fonts, 17)
        row = Image.new("RGB", (width, y + 28), "white")
        draw = ImageDraw.Draw(row)
        draw.line((margin, 0, width - margin, 0), fill=_RULE, width=2)
        _paragraph(draw, title, margin, 22, width - 2 * margin, fonts, 25, _INK)
        for column, index in enumerate(indices):
            x = margin + column * (_TILE[0] + gap)
            tile, has_overlay = _panel(frames[index], pose["lm"][index], pose["vis"][index])
            row.paste(tile, (x, top))
            draw.rectangle((x, top, x + _TILE[0] - 1, top + _TILE[1] - 1), outline=_RULE)
            draw.text((x, top + _TILE[1] + 11), f"#{index}  |  {index / fps:.2f} s", font=fonts.at(18), fill=_INK)
            if not has_overlay:
                label = fonts.label("no_pose")
                label_width = draw.textlength(label, font=fonts.at(17))
                draw.rectangle((x + 7, top + 8, x + label_width + 20, top + 37), fill="white")
                draw.text((x + 13, top + 11), label, font=fonts.at(17), fill=_MUTED)
        _metric_panel(draw, report, metric_x, top, metric_width, fonts)
        y = content_end + 18
        for note in notes:
            y = _paragraph(draw, note, margin, y, width - 2 * margin, fonts, 17)
        rows.append(row)
    header_height = 158
    footer_lines = [fonts.label(key) for key in ("limitation", "head_limitation", "clock", "overlay")]
    footer_measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    footer_height = 24
    for line in footer_lines:
        footer_height = _paragraph(footer_measure, line, margin, footer_height,
                                   width - 2 * margin, fonts, 17)
    figure = Image.new("RGB", (width, header_height + sum(row.height for row in rows) + footer_height + 24), "white")
    draw = ImageDraw.Draw(figure)
    draw.text((margin, 32), fonts.label("title"), font=fonts.at(34), fill=_INK)
    _paragraph(draw, fonts.label("subtitle"), margin, 89, width - 2 * margin, fonts, 20)
    y = header_height
    for row in rows:
        figure.paste(row, (0, y))
        y += row.height
    draw.line((margin, y, width - margin, y), fill=_RULE, width=2)
    y += 24
    for line in footer_lines:
        y = _paragraph(draw, line, margin, y, width - 2 * margin, fonts, 17)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_itxt("Description", _NOTICE + " Videos: " + "; ".join(
        str(item["report"]["video"].get("name", item["video_path"])) for item in items))
    _atomic_write(output, lambda path: figure.save(path, format="PNG", dpi=(300, 300), pnginfo=metadata))
    return output


def _exact(value):
    if isinstance(value, Mapping):
        return "; ".join(f"{key}: {_exact(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return "; ".join(_exact(item) for item in value)
    return _value(value)


def write_html_report(reports, image_path, output_path):
    """Write escaped static HTML with an offline relative image URL; return Path.

    Keep this HTML and the referenced PNG at their relative locations when
    sharing. No scripts, remote assets, web fonts, or model imports are used.
    Table values retain the supplied numeric precision; the PNG rounds to two
    decimals. Extra per-metric fields (including sample counts) and all quality
    fields are preserved as text.
    """
    reports = list(reports)
    if not reports:
        raise ValueError("reports must contain at least one report")
    for report in reports:
        _validate_report(report)
    image, output = Path(image_path), Path(output_path)
    if not image.is_file():
        raise FileNotFoundError(f"Report image does not exist: {image}")
    if _same_file(image, output):
        raise ValueError("HTML output must not overwrite its source image")
    relative = Path(os.path.relpath(image.resolve(), output.parent.resolve())).as_posix()
    image_url = escape(quote(relative, safe="/"), quote=True)
    sections = []
    for index, report in enumerate(reports, 1):
        video = report["video"]
        name = escape(str(video.get("name", f"Video {index}")))
        rows = []
        for key, label in _METRICS:
            metric = report["metrics"][key]
            reason = metric.get("reason") or ("Measurement unavailable" if _value(metric.get("score")) == "N/A" else "—")
            counts = {key: value for key, value in metric.items() if key not in {"raw", "unit", "score", "reason"}}
            values = (label, _value(metric.get("raw")), metric.get("unit", ""),
                      _value(metric.get("score")), reason, _exact(counts) or "—")
            rows.append("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in values) + "</tr>")
        total = escape(_value(report.get("total")))
        weights = escape("; ".join(f"{label}: {_value(value)}" for label, value in _weights(report)))
        quality = escape(_exact(report.get("quality", {})) or "No quality details supplied.")
        sections.append(f"""<section aria-labelledby="video-{index}">
<h2 id="video-{index}">{index:02d} · {name}</h2>
<p>{escape(_value(video['frames']))} source frames · {escape(_value(video['fps']))} fps</p>
<div class="table-scroll"><table><caption>Supplied measurements and scores: {name}</caption>
<thead><tr><th scope="col">Metric</th><th scope="col">Raw value</th><th scope="col">Unit</th>
<th scope="col">Score / 10</th><th scope="col">Reason</th><th scope="col">Samples / details</th></tr></thead>
<tbody>{''.join(rows)}<tr class="total"><th scope="row">Total</th><td colspan="2">Weighted score</td>
<td>{total}</td><td colspan="2">{'All three component scores are required.' if total == 'N/A' else 'As supplied by the report.'}</td></tr></tbody></table></div>
<p><strong>Fixed weights for this report:</strong> {weights}</p>
<p><strong>Quality:</strong> {quality}</p></section>""")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dribbling motion report</title>
<style>
:root {{ color-scheme: light; font-family: system-ui, sans-serif; color: #1b232b; background: white; }}
body {{ margin: 0 auto; max-width: 1500px; padding: 40px 28px; line-height: 1.55; }}
h1 {{ font-size: 2rem; margin-bottom: .4rem; }} h2 {{ font-size: 1.3rem; overflow-wrap: anywhere; }}
p, figcaption {{ overflow-wrap: anywhere; }} .notice, figcaption {{ color: #586169; }}
figure {{ margin: 32px 0; }} img {{ display: block; width: 100%; height: auto; }}
figcaption {{ font-size: .9rem; margin-top: 12px; }} section {{ border-top: 1px solid #dae0e5; padding: 20px 0; }}
.table-scroll {{ overflow-x: auto; }} table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
caption {{ text-align: left; font-weight: 600; padding: 10px 0; }}
th, td {{ padding: 12px; border-bottom: 1px solid #dae0e5; text-align: left; vertical-align: top; overflow-wrap: anywhere; }}
thead th {{ background: #f5f7f8; }} .total {{ font-weight: 600; }}
@media print {{ body {{ max-width: none; padding: 0; font-size: 10pt; }} section {{ break-inside: avoid; }} }}
</style></head><body>
<header><h1>Dribbling motion report</h1><p class="notice">{escape(_NOTICE)}</p>
<p>Fixed calibration and weights are not fitted to these video labels. Any weight override is recorded per video.
Missing measurements remain N/A. Displayed figure scores use two decimals; tables retain supplied precision.</p></header>
<figure><img src="{image_url}" alt="Sequential real source frames with schematic pose overlays and reported scores">
<figcaption>Frames use zero-based source indices and seconds (frame / fps). Missing poses retain the real frame.
Short clips may repeat labeled source frames. Red: shoulders; blue: trunk/hips/knees; yellow: head vector.
The 2D overlay is schematic, not a measured 3D angle.</figcaption></figure>
{''.join(sections)}
<footer><p class="notice">Offline report: keep this HTML and its local PNG at their relative locations.</p></footer>
</body></html>
"""
    _atomic_write(output, lambda path: path.write_text(document, encoding="utf-8"))
    return output
