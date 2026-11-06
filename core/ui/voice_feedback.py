# core/ui/voice_feedback.py
# -*- coding: utf-8 -*-
"""
Voice Feedback — Offline Text-to-Speech for the Liver AI Pipeline.

Uses pyttsx3 (offline, no API key, no internet needed).

Features:
    - Multi-language support (EN, FA, AR, RU, DE, FR, ES, ...)
    - Thread-safe queue
    - Toggleable (enabled/disabled)
    - Volume control
    - Graceful fallback if pyttsx3 not installed

Usage:
    from core.ui.voice_feedback import VoiceManager

    vm = VoiceManager.get()
    vm.set_language("English")
    vm.speak("Welcome, John!")
    vm.speak_key("inference_ok", grid=2, conf=95.0)
"""

import os
import sys
import threading
import queue
import time
from typing import Optional, Dict, Any

# ✅ Optional dependency — graceful fallback
try:
    import pyttsx3

    HAS_TTS = True
except ImportError:
    HAS_TTS = False
    print("⚠️ pyttsx3 not installed — Voice Feedback disabled")
    print("   → Install with: pip install pyttsx3")

# ================================================================
# ✅ Language → TTS voice mapping
# ================================================================

# pyttsx3 روی ویندوز: زبانه‌ها با اسم voice مشخص می‌شن
# این mapping رو استفاده می‌کنیم که در runtime بهترین voice رو پیدا کنیم
LANG_VOICE_HINTS = {
    "English": ["english", "en-us", "en_us", "en-gb", "david", "zira", "mark"],
    "USA": ["english", "en-us", "david", "zira"],
    "Persian": ["persian", "farsi", "fa-ir", "fa_ir", "iran"],
    "Arabic": ["arabic", "ar-sa", "ar_sa", "ar-eg", "hoda"],
    "Morocco": ["arabic", "ar-sa", "ar-ma", "hoda"],
    "Russian": ["russian", "ru-ru", "ru_ru", "irina", "pavel"],
    "Germany": ["german", "de-de", "de_de", "hedda", "stefan"],
    "France": ["french", "fr-fr", "fr_fr", "hortense", "paul"],
    "Italy": ["italian", "it-it", "it_it", "elsa", "cosimo"],
    "Portugal": ["portuguese", "pt-pt", "pt_pt", "maria", "helder"],
    "Brazil": ["portuguese", "pt-br", "pt_br", "maria", "daniel"],
    "Greece": ["greek", "el-gr", "el_gr", "stefanos"],
    "Turkey": ["turkish", "tr-tr", "tr_tr", "tolga"],
    "China": ["chinese", "zh-cn", "zh_cn", "huihui", "yaoyao"],
    "Japan": ["japanese", "ja-jp", "ja_jp", "haruka", "ayumi"],
    "SouthKorea": ["korean", "ko-kr", "ko_kr", "heami"],
    "NorthKorea": ["korean", "ko-kr", "ko_kr", "heami"],
    "India": ["hindi", "en-in", "hi-in", "heera"],
    "Ukranian": ["ukrainian", "uk-ua", "uk_ua"],
    "Switzerland": ["german", "de-ch", "de_de"],
    "Sweden": ["swedish", "sv-se", "sv_se"],
    "Wales": ["welsh", "en-gb", "en_gb"],
    "Nederland": ["dutch", "nl-nl", "nl_nl"],
    "Argentina": ["spanish", "es-ar", "es_es", "helena"],
    "Spain": ["spanish", "es-es", "es_es", "helena"],
}


# ================================================================
# ✅ VoiceManager (singleton, thread-safe)
# ================================================================

class VoiceManager:
    """
    Manager صداگذاری — singleton.

    چرا thread جدا؟
        چون pyttsx3 وقتی speak() می‌زنی، تا آخر صدا block می‌کنه.
        اگه توی UI thread اجرا بشه → UI freeze.
        پس یه worker thread داریم که از queue می‌خونه.
    """

    _instance = None

    def __init__(self):
        self._enabled = True
        self._volume = 0.8
        self._rate = 170  # سرعت (کلمات در دقیقه)
        self._language = "English"
        self._engine = None
        self._voice_id = None
        self._queue = queue.Queue()
        self._worker = None
        self._stop_flag = False
        self._lock = threading.Lock()

        if not HAS_TTS:
            return

        # ✅ شروع worker thread
        self._start_worker()

    @classmethod
    def get(cls) -> "VoiceManager":
        if cls._instance is None:
            cls._instance = VoiceManager()
        return cls._instance

    # ============================================================
    # ✅ Worker thread
    # ============================================================
    def _start_worker(self):
        """شروع thread صداگذاری."""
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    def _run_loop(self):
        """حلقه‌ی اصلی worker — از queue می‌خونه و speak می‌کنه."""
        while not self._stop_flag:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                text, lang = item
                self._do_speak(text, lang)
            except Exception as e:
                print(f"⚠️ Voice worker error: {e}")
            finally:
                try:
                    self._queue.task_done()
                except Exception:
                    pass

    def _do_speak(self, text: str, lang: str):
        """اجرای واقعی speak — در worker thread."""
        try:
            # ✅ ساخت engine جدید برای هر speak (راه‌حل crash در ویندوز)
            engine = pyttsx3.init()
            engine.setProperty("volume", self._volume)
            engine.setProperty("rate", self._rate)

            # ✅ پیدا کردن voice مناسب زبان
            voice_id = self._find_voice(engine, lang)
            if voice_id:
                engine.setProperty("voice", voice_id)

            engine.say(text)
            engine.runAndWait()

            # ✅ آزادسازی
            try:
                engine.stop()
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️ TTS speak failed: {e}")

    def _find_voice(self, engine, lang: str) -> Optional[str]:
        """پیدا کردن voice مناسب زبان."""
        try:
            voices = engine.getProperty("voices")
            if not voices:
                return None

            hints = LANG_VOICE_HINTS.get(lang, LANG_VOICE_HINTS["English"])

            # ✅ ۳ مرحله: اول voice.id رو چک کن، بعد voice.name، بعد voice.languages
            for hint in hints:
                hint_lower = hint.lower()
                for v in voices:
                    # چک کردن voice.id
                    vid = (getattr(v, "id", "") or "").lower()
                    if hint_lower in vid:
                        return v.id
                    # چک کردن voice.name
                    vname = (getattr(v, "name", "") or "").lower()
                    if hint_lower in vname:
                        return v.id
                    # چک کردن languages
                    vlangs = getattr(v, "languages", []) or []
                    for vl in vlangs:
                        try:
                            vl_str = vl.decode("utf-8", errors="ignore") if isinstance(vl, bytes) else str(vl)
                            if hint_lower in vl_str.lower():
                                return v.id
                        except Exception:
                            pass

            # ✅ fallback: اولین voice
            return voices[0].id
        except Exception as e:
            print(f"⚠️ _find_voice failed: {e}")
            return None

    # ============================================================
    # ✅ Public API
    # ============================================================
    def set_enabled(self, enabled: bool):
        """فعال/غیرفعال کردن."""
        self._enabled = bool(enabled)

    def is_enabled(self) -> bool:
        return self._enabled

    def set_volume(self, volume: float):
        """تنظیم صدا (0.0 تا 1.0)."""
        self._volume = max(0.0, min(1.0, float(volume)))

    def set_rate(self, rate: int):
        """تنظیم سرعت (کلمات در دقیقه، 50-300)."""
        self._rate = max(50, min(300, int(rate)))

    def set_language(self, lang: str):
        """تنظیم زبان — برای next utterance."""
        self._language = lang

    def speak(self, text: str, lang: Optional[str] = None):
        """
        صداگذاری یک متن.

        Args:
            text: متن (باید آماده باشه — ترجمه شده)
            lang: زبان (اگه None، از self._language استفاده می‌کنه)
        """
        if not HAS_TTS or not self._enabled:
            return
        if not text or not text.strip():
            return

        # ✅ اگه صف خیلی پر شده، پاک کن (جلوگیری از انباشتگی)
        try:
            if self._queue.qsize() > 3:
                # پاک کردن queue
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                        self._queue.task_done()
                    except Exception:
                        break
        except Exception:
            pass

        try:
            self._queue.put_nowait((text, lang or self._language))
        except Exception as e:
            print(f"⚠️ Voice queue failed: {e}")

    def stop(self):
        """توقف worker thread."""
        self._stop_flag = True
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass


# ================================================================
# ✅ Translation keys for voice feedback
# ================================================================
# این‌ها رو در TRANSLATIONS اضافه می‌کنم — اینجا فقط تعریف keys

VOICE_KEYS = {
    "voice_welcome": "Welcome, {user}",
    "voice_goodbye": "Goodbye",
    "voice_logged_out": "Logged out successfully",
    "voice_image_loaded": "Image loaded successfully",
    "voice_image_rejected": "This is not a liver ultrasound image",
    "voice_inference_result": "{grid_name}, confidence {conf} percent",
    "voice_pdf_saved": "PDF report saved successfully",
    "voice_error": "An error occurred",
    "voice_settings_applied": "Settings applied",
    "voice_language_changed": "Language changed to {lang}",
    "voice_theme_changed": "Theme changed",
    "voice_2fa_enabled": "Two factor authentication enabled",
    "voice_2fa_disabled": "Two factor authentication disabled",
    "voice_locked": "Session locked",
    "voice_unlocked": "Session unlocked",
    "voice_processing": "Processing image",
    "voice_processing_done": "Processing complete",
    "voice_voice_enabled": "Voice feedback enabled",
    "voice_voice_disabled": "Voice feedback disabled",
}
