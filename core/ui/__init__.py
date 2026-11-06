# core/ui/__init__.py
# -*- coding: utf-8 -*-

from .font_manager import (
    FontSettings, FontManager,
    load_font_settings, save_font_settings,
    list_available_fonts,
    DEFAULT_FAMILY, DEFAULT_SIZE, DEFAULT_SCALE,
    MIN_SCALE, MAX_SCALE,
)
from .micro_interactions import (
    MotionSettings,
    shake_widget, add_focus_glow, show_success_check,
    animate_count_up, button_press_scale, smooth_scroll_to,
    bounce_in, color_pulse, page_transition_in,
    decorate_input, decorate_button_press,
    highlight_widget_error, highlight_widget_success,
)
from .empty_state import (
    EmptyStateStyle,
    EMPTY_STATE_STYLES,
    make_empty_state_html,
    EmptyStateWidget,
)
from .onboarding import (
    OnboardingStorage,
    OnboardingStep,
    OnboardingTour,
    TOUR_STEPS_GUEST,
    TOUR_STEPS_FULL,
    TOUR_VERSION,
)
from .smart_suggestions import (
    SuggestionStorage,
    Suggestion,
    SuggestionContext,
    SuggestionEngine,
    SmartSuggestionBar,
    CHECK_INTERVAL_MS,
    AUTO_HIDE_MS,
)

__all__ = [
    # Font
    "FontSettings", "FontManager",
    "load_font_settings", "save_font_settings",
    "list_available_fonts",
    "DEFAULT_FAMILY", "DEFAULT_SIZE", "DEFAULT_SCALE",
    "MIN_SCALE", "MAX_SCALE",
    # Micro-interactions
    "MotionSettings",
    "shake_widget", "add_focus_glow", "show_success_check",
    "animate_count_up", "button_press_scale", "smooth_scroll_to",
    "bounce_in", "color_pulse", "page_transition_in",
    "decorate_input", "decorate_button_press",
    "highlight_widget_error", "highlight_widget_success",
    # Empty state
    "EmptyStateStyle", "EMPTY_STATE_STYLES",
    "make_empty_state_html", "EmptyStateWidget",
    # Onboarding
    "OnboardingStorage", "OnboardingStep", "OnboardingTour",
    "TOUR_STEPS_GUEST", "TOUR_STEPS_FULL", "TOUR_VERSION",
    # Smart suggestions
    "SuggestionStorage", "Suggestion", "SuggestionContext",
    "SuggestionEngine", "SmartSuggestionBar",
    "CHECK_INTERVAL_MS", "AUTO_HIDE_MS",
]