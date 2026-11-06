# core/ui/confetti.py
# -*- coding: utf-8 -*-
"""
Confetti Animation on Success — v1.0

A lightweight, beautiful, physics-based confetti animation that plays
when the user gets a successful result (inference complete, PDF saved, etc.).

Features:
    - Pure QPainter-based (no external dependencies)
    - Physics simulation (gravity, drag, rotation, wobble)
    - Theme-aware colors (uses theme primary/secondary/accent/success)
    - Multiple burst patterns: center, top, corners, celebration
    - Respects MotionSettings (reduced motion → no animation)
    - Auto-cleanup (QTimer-based lifecycle, no memory leaks)
    - FPS-capped (60 FPS) for low CPU usage
    - Safe: never crashes even if parent is destroyed

Usage:
    from core.ui.confetti import ConfettiOverlay, celebrate

    # Simple one-liner (fire-and-forget):
    celebrate(parent_widget, theme=THEMES["theme_dark_aqua"])

    # Or with customization:
    overlay = ConfettiOverlay(
        parent=main_window,
        theme=theme_dict,
        particle_count=120,
        duration_ms=2500,
        burst_pattern="center",
    )
    overlay.start()
"""

import math
import random
from typing import List, Optional, Dict, Tuple

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import (
    Qt, QTimer, QPoint, QPointF, QRect, QRectF, QObject, pyqtSignal,
)
from PyQt5.QtGui import (
    QPainter, QColor, QBrush, QPen, QPainterPath, QPolygonF,
)


# ================================================================
# ✅ Motion Settings (safe wrapper)
# ================================================================

def _motion_allowed() -> bool:
    """Check if animations are allowed (respects reduced-motion setting)."""
    try:
        from core.ui.micro_interactions import MotionSettings
        return not MotionSettings.get().is_reduced()
    except Exception:
        return True


# ================================================================
# ✅ Particle (single confetti piece)
# ================================================================

class ConfettiParticle:
    """
    یک تکه‌ی کاغذرنگی (confetti).

    فیزیک:
        - position (x, y)
        - velocity (vx, vy) — با گرانش و drag
        - rotation (angle) + angular velocity
        - wobble (نوسان افقی ملایم برای حس کاغذ)
        - size, color, shape

    شکل‌ها:
        - "rect"    — مستطیل کاغذی (کلاسیک)
        - "circle"  — دایره‌ی کوچک
        - "ribbon"  — روبان باریک (مستطیل بلند)
        - "star"    — ستاره‌ی ۵ پر
    """

    __slots__ = (
        "x", "y", "vx", "vy", "angle", "angular_velocity",
        "wobble_phase", "wobble_speed", "wobble_amp",
        "size_w", "size_h", "color", "shape",
        "life", "max_life", "alpha",
    )

    def __init__(
        self,
        x: float, y: float,
        vx: float, vy: float,
        color: QColor,
        shape: str = "rect",
        size: float = 8.0,
        max_life: float = 2.5,
    ):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.angle = random.uniform(0.0, 360.0)
        self.angular_velocity = random.uniform(-480.0, 480.0)

        # ✅ wobble (نوسان افقی برای حس کاغذ سبک)
        self.wobble_phase = random.uniform(0.0, 2.0 * math.pi)
        self.wobble_speed = random.uniform(6.0, 14.0)
        self.wobble_amp = random.uniform(8.0, 22.0)

        # ✅ ابعاد (مستطیل / روبان)
        if shape == "ribbon":
            self.size_w = size * 0.5
            self.size_h = size * 2.2
        elif shape == "circle":
            self.size_w = size * 0.7
            self.size_h = size * 0.7
        elif shape == "star":
            self.size_w = size * 1.1
            self.size_h = size * 1.1
        else:  # rect
            self.size_w = size
            self.size_h = size * 0.6

        self.color = QColor(color)
        self.shape = shape

        self.life = 0.0
        self.max_life = float(max_life)
        self.alpha = 1.0

    # ------------------------------------------------------------
    # Update — یه قدم فیزیک
    # ------------------------------------------------------------
    def update(self, dt: float) -> bool:
        """
        یه قدم فیزیک (dt = ثانیه).
        Returns:
            True اگه زنده، False اگه باید حذف شه.
        """
        # ✅ life
        self.life += dt
        if self.life >= self.max_life:
            return False

        # ✅ گرانش
        self.vy += 480.0 * dt

        # ✅ drag (مقاومت هوا)
        drag = 1.0 - (1.8 * dt)
        if drag < 0.0:
            drag = 0.0
        self.vx *= drag
        self.vy *= drag

        # ✅ wobble (نوسان افقی ملایم)
        self.wobble_phase += self.wobble_speed * dt
        wobble_dx = math.sin(self.wobble_phase) * self.wobble_amp * dt

        # ✅ position
        self.x += self.vx * dt + wobble_dx
        self.y += self.vy * dt

        # ✅ rotation
        self.angle += self.angular_velocity * dt

        # ✅ fade-out در ۳۰٪ آخر عمر
        life_ratio = self.life / self.max_life
        if life_ratio > 0.7:
            self.alpha = max(0.0, 1.0 - (life_ratio - 0.7) / 0.3)
        else:
            self.alpha = 1.0

        return True

    # ------------------------------------------------------------
    # Draw — رسم روی painter
    # ------------------------------------------------------------
    def draw(self, painter: QPainter):
        try:
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing, True)

            c = QColor(self.color)
            c.setAlphaF(max(0.0, min(1.0, self.alpha)))

            painter.translate(self.x, self.y)
            painter.rotate(self.angle)

            if self.shape == "circle":
                painter.setBrush(QBrush(c))
                painter.setPen(Qt.NoPen)
                r = self.size_w / 2.0
                painter.drawEllipse(QPointF(0, 0), r, r)

            elif self.shape == "star":
                path = self._make_star_path(self.size_w * 0.5, 5)
                painter.setBrush(QBrush(c))
                painter.setPen(Qt.NoPen)
                painter.drawPath(path)

            else:
                # rect / ribbon
                w = self.size_w
                h = self.size_h
                rect = QRectF(-w / 2.0, -h / 2.0, w, h)
                painter.setBrush(QBrush(c))
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(rect, 1.2, 1.2)

            painter.restore()
        except Exception:
            try:
                painter.restore()
            except Exception:
                pass

    @staticmethod
    def _make_star_path(radius: float, points: int) -> QPainterPath:
        """ساخت مسیر ستاره‌ی ۵ پر."""
        path = QPainterPath()
        inner = radius * 0.42
        for i in range(points * 2):
            r = radius if i % 2 == 0 else inner
            ang = math.pi / 2.0 + i * math.pi / points
            x = r * math.cos(ang)
            y = -r * math.sin(ang)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        path.closeSubpath()
        return path


# ================================================================
# ✅ ConfettiOverlay (Widget شفاف روی پنجره)
# ================================================================

class ConfettiOverlay(QWidget):
    """
    Overlay شفاف که ذرات رو روی خودش رسم می‌کنه.

    - تمام‌صفحه روی parent
    - WA_TransparentForMouseEvents → جلوی کلیک رو نمی‌گیره
    - WA_TranslucentBackground → پس‌زمینه شفاف
    - QTimer 60 FPS برای انیمیشن
    - بعد از پایان، خودش رو حذف می‌کنه
    """

    FPS = 60
    TICK_MS = 1000 // FPS  # 16ms

    # ✅ الگوهای burst
    PATTERN_CENTER = "center"
    PATTERN_TOP = "top"
    PATTERN_CORNERS = "corners"
    PATTERN_CELEBRATION = "celebration"
    PATTERN_FOUNTAIN = "fountain"

    def __init__(
        self,
        parent: QWidget,
        theme: Optional[Dict] = None,
        particle_count: int = 120,
        duration_ms: int = 2500,
        burst_pattern: str = "center",
        gravity_scale: float = 1.0,
    ):
        super().__init__(parent)

        self._theme = theme or {}
        self._particle_count = max(10, min(400, int(particle_count)))
        self._duration_ms = max(800, min(8000, int(duration_ms)))
        self._burst_pattern = burst_pattern
        self._gravity_scale = float(gravity_scale)

        self._particles: List[ConfettiParticle] = []
        self._elapsed_ms = 0
        self._stopped = False

        # ✅ تنظیمات widget
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.setFocusPolicy(Qt.NoFocus)

        # ✅ هم‌سایز با parent
        self.setGeometry(parent.rect())
        self.raise_()

        # ✅ تایمر
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._on_tick)

        # ✅ رنگ‌های theme-aware
        self._palette = self._build_palette()

        # ✅ حالت متوقف
        self.setVisible(False)

    # ============================================================
    # Palette — رنگ‌ها از theme
    # ============================================================
    def _build_palette(self) -> List[QColor]:
        """
        رنگ‌های confetti از theme.
        اگه theme نبود، fallback زیبا.
        """
        t = self._theme or {}
        colors: List[QColor] = []

        for key in ("primary", "secondary", "accent", "success", "warning"):
            val = t.get(key)
            if isinstance(val, str):
                try:
                    colors.append(QColor(val))
                except Exception:
                    pass

        # ✅ اضافه‌کردن چند رنگ شاد برای تنوع
        extras = [
            "#FFD700",  # gold
            "#FF6B9D",  # pink
            "#4ECDC4",  # turquoise
            "#FFA07A",  # light salmon
            "#C084FC",  # light purple
            "#F472B6",  # pink
            "#34D399",  # green
            "#00E5FF",  # cyan
        ]
        for hexcol in extras:
            try:
                colors.append(QColor(hexcol))
            except Exception:
                continue

        # ✅ حذف تکراری‌ها (بر اساس RGB)
        seen = set()
        unique = []
        for c in colors:
            key = (c.red(), c.green(), c.blue())
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique or [QColor("#FF6B9D"), QColor("#4ECDC4"), QColor("#FFD700")]

    def _random_color(self) -> QColor:
        return random.choice(self._palette)

    # ============================================================
    # Spawn — تولید ذرات بر اساس pattern
    # ============================================================
    def _spawn_particles(self):
        w = max(1, self.width())
        h = max(1, self.height())
        n = self._particle_count

        if self._burst_pattern == self.PATTERN_TOP:
            self._spawn_from_line(n, y=0, w=w, spread_x=1.0)
        elif self._burst_pattern == self.PATTERN_CORNERS:
            self._spawn_from_corners(n, w=w, h=h)
        elif self._burst_pattern == self.PATTERN_CELEBRATION:
            # ✅ ترکیب center + top + corners
            self._spawn_from_center(n // 2, w=w, h=h)
            self._spawn_from_line(n // 4, y=0, w=w, spread_x=1.0)
            self._spawn_from_corners(n // 4, w=w, h=h)
        elif self._burst_pattern == self.PATTERN_FOUNTAIN:
            self._spawn_fountain(n, w=w, h=h)
        else:  # PATTERN_CENTER
            self._spawn_from_center(n, w=w, h=h)

    def _spawn_from_center(self, count: int, w: int, h: int):
        cx = w / 2.0
        cy = h / 2.0
        for _ in range(count):
            ang = random.uniform(0.0, 2.0 * math.pi)
            speed = random.uniform(280.0, 620.0) * self._gravity_scale
            vx = math.cos(ang) * speed
            vy = math.sin(ang) * speed - 180.0  # یه کم به بالا
            shape = self._random_shape()
            size = random.uniform(6.0, 12.0)
            max_life = random.uniform(1.8, 3.2)
            p = ConfettiParticle(
                x=cx + random.uniform(-15, 15),
                y=cy + random.uniform(-15, 15),
                vx=vx, vy=vy,
                color=self._random_color(),
                shape=shape,
                size=size,
                max_life=max_life,
            )
            self._particles.append(p)

    def _spawn_from_line(self, count: int, y: float, w: int, spread_x: float = 1.0):
        for _ in range(count):
            x = random.uniform(0.0, w * spread_x)
            ang = random.uniform(math.radians(240), math.radians(300))  # به سمت پایین
            speed = random.uniform(200.0, 480.0) * self._gravity_scale
            vx = math.cos(ang) * speed + random.uniform(-60, 60)
            vy = abs(math.sin(ang) * speed)
            shape = self._random_shape()
            size = random.uniform(6.0, 12.0)
            max_life = random.uniform(2.0, 3.5)
            p = ConfettiParticle(
                x=x, y=y,
                vx=vx, vy=vy,
                color=self._random_color(),
                shape=shape,
                size=size,
                max_life=max_life,
            )
            self._particles.append(p)

    def _spawn_from_corners(self, count: int, w: int, h: int):
        corners = [
            (0, 0, math.radians(45)),         # top-left → پایین-راست
            (w, 0, math.radians(135)),         # top-right → پایین-چپ
            (0, h, math.radians(-45)),         # bottom-left → بالا-راست
            (w, h, math.radians(-135)),        # bottom-right → بالا-چپ
        ]
        per_corner = max(1, count // 4)
        for (cx, cy, base_ang) in corners:
            for _ in range(per_corner):
                ang = base_ang + random.uniform(-0.5, 0.5)
                speed = random.uniform(320.0, 680.0) * self._gravity_scale
                vx = math.cos(ang) * speed
                vy = math.sin(ang) * speed
                shape = self._random_shape()
                size = random.uniform(6.0, 12.0)
                max_life = random.uniform(1.8, 3.0)
                p = ConfettiParticle(
                    x=cx, y=cy,
                    vx=vx, vy=vy,
                    color=self._random_color(),
                    shape=shape,
                    size=size,
                    max_life=max_life,
                )
                self._particles.append(p)

    def _spawn_fountain(self, count: int, w: int, h: int):
        """فواره از پایین-وسط به بالا."""
        cx = w / 2.0
        cy = h
        for _ in range(count):
            ang = random.uniform(math.radians(60), math.radians(120))  # به سمت بالا
            speed = random.uniform(500.0, 900.0) * self._gravity_scale
            vx = math.cos(ang) * speed + random.uniform(-80, 80)
            vy = -abs(math.sin(ang) * speed)
            shape = self._random_shape()
            size = random.uniform(6.0, 12.0)
            max_life = random.uniform(2.0, 3.5)
            p = ConfettiParticle(
                x=cx + random.uniform(-30, 30),
                y=cy,
                vx=vx, vy=vy,
                color=self._random_color(),
                shape=shape,
                size=size,
                max_life=max_life,
            )
            self._particles.append(p)

    @staticmethod
    def _random_shape() -> str:
        r = random.random()
        if r < 0.55:
            return "rect"
        elif r < 0.75:
            return "ribbon"
        elif r < 0.90:
            return "circle"
        else:
            return "star"

    # ============================================================
    # Public API
    # ============================================================
    def start(self):
        """شروع انیمیشن."""
        if self._stopped:
            return
        if not _motion_allowed():
            # ✅ reduced motion → نذار انیمیشن اجرا شه
            self._cleanup()
            return
        try:
            # ✅ اطمینان از geometry درست
            parent = self.parentWidget()
            if parent is not None:
                self.setGeometry(parent.rect())

            self._spawn_particles()
            if not self._particles:
                self._cleanup()
                return

            self._elapsed_ms = 0
            self.setVisible(True)
            self.raise_()
            self._timer.start(self.TICK_MS)
        except Exception as e:
            print(f"⚠️ ConfettiOverlay.start failed: {e}")
            self._cleanup()

    def stop(self):
        """توقف فوری."""
        self._stopped = True
        try:
            self._timer.stop()
        except Exception:
            pass
        self._particles.clear()
        self.setVisible(False)

    # ============================================================
    # Internals
    # ============================================================
    def _on_tick(self):
        if self._stopped:
            return
        try:
            dt = self.TICK_MS / 1000.0
            self._elapsed_ms += self.TICK_MS

            # ✅ آپدیت ذرات
            alive: List[ConfettiParticle] = []
            for p in self._particles:
                if p.update(dt):
                    # ✅ حذف ذراتی که از صفحه خارج شدن
                    if -50 < p.y < self.height() + 80 and -80 < p.x < self.width() + 80:
                        alive.append(p)
            self._particles = alive

            # ✅ پایان
            if not self._particles or self._elapsed_ms >= self._duration_ms:
                self._timer.stop()
                self._cleanup()
                return

            self.update()
        except Exception as e:
            print(f"⚠️ ConfettiOverlay._on_tick failed: {e}")
            self._cleanup()

    def _cleanup(self):
        """پاک‌سازی و حذف خود."""
        self._stopped = True
        try:
            self._timer.stop()
        except Exception:
            pass
        self._particles.clear()
        self.setVisible(False)
        try:
            self.deleteLater()
        except Exception:
            pass

    def paintEvent(self, event):
        if not self._particles:
            return
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            for p in self._particles:
                p.draw(painter)
            painter.end()
        except Exception:
            try:
                painter.end()
            except Exception:
                pass

    def resizeEvent(self, event):
        """اگه parent تغییر اندازه داد، هم‌سایز شو."""
        super().resizeEvent(event)
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect())


# ================================================================
# ✅ celebrate() — یکی‌خطی برای استفاده
# ================================================================

def celebrate(
    parent: QWidget,
    theme: Optional[Dict] = None,
    pattern: str = "celebration",
    particle_count: int = 120,
    duration_ms: int = 2500,
    gravity_scale: float = 1.0,
) -> Optional[ConfettiOverlay]:
    """
    نمایش یه انیمیشن confetti روی parent.

    Args:
        parent: ویجتی که confetti روش رسم می‌شه (معمولاً MainWindow.centralWidget()).
        theme: دیکشنری تم (برای رنگ‌ها).
        pattern: "center" | "top" | "corners" | "celebration" | "fountain".
        particle_count: تعداد ذرات (پیش‌فرض ۱۲۰).
        duration_ms: مدت (پیش‌فرض ۲.۵ ثانیه).
        gravity_scale: ضریب گرانش (۱.۰ = عادی).

    Returns:
        ConfettiOverlay instance یا None اگه نشد.

    Example:
        >>> celebrate(self.centralWidget(), THEMES["theme_dark_aqua"])
        >>> celebrate(self, pattern="fountain", particle_count=80)
    """
    if parent is None:
        return None
    if not _motion_allowed():
        return None

    try:
        overlay = ConfettiOverlay(
            parent=parent,
            theme=theme,
            particle_count=particle_count,
            duration_ms=duration_ms,
            burst_pattern=pattern,
            gravity_scale=gravity_scale,
        )
        overlay.start()
        return overlay
    except Exception as e:
        print(f"⚠️ celebrate failed: {e}")
        return None


# ================================================================
# ✅ celebration_patterns — helper برای انتخاب pattern
# ================================================================

CELEBRATION_PATTERNS = {
    "success":       "celebration",  # برای موفقیت‌های مهم (inference, PDF)
    "gentle":        "center",       # برای موفقیت‌های کوچک‌تر
    "achievement":   "fountain",     # برای دستاوردها (first result, streak)
    "welcome":       "top",          # برای خوش‌آمدگویی (login, signup)
}


def get_pattern_for(event_type: str) -> str:
    """نقشه‌ی رویداد → pattern."""
    return CELEBRATION_PATTERNS.get(event_type, "celebration")
