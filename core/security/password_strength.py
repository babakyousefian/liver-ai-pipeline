# core/security/password_strength.py
# -*- coding: utf-8 -*-
"""
Password strength calculation — pure function, testable.

Algorithm:
1. Base score from length (0-50 points)
2. Bonus for character variety (0-30 points)
3. Bonus for length beyond minimum (0-20 points)
4. Penalty for common patterns (-20 points each)
5. Penalty for sequences (-10 points each)

Final score (0-100) maps to 5 levels:
- 0-19:  VERY_WEAK
- 20-39: WEAK
- 40-59: MEDIUM
- 60-79: STRONG
- 80-100: VERY_STRONG
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


# ============================================================
# Rules (must match validate_password in GUI_update.py)
# ============================================================

MIN_LENGTH = 8
MAX_LENGTH = 16
MIN_LOWERCASE = 1
MIN_UPPERCASE = 1
MIN_DIGITS = 4
MIN_SPECIAL = 4

SPECIAL_CHARS = r"!@#$%^&*(),.?\":{}|<>_\-+=\[\]\\/;`~"

# ✅ Common weak patterns (case-insensitive)
COMMON_PATTERNS = [
    "123456", "12345678", "123456789", "12345", "1234",
    "password", "pass", "qwerty", "qwertyuiop", "asdfgh",
    "zxcvbn", "admin", "administrator", "root", "user",
    "welcome", "login", "letmein", "master", "monkey",
    "dragon", "sunshine", "princess", "football", "baseball",
    "abc123", "iloveyou", "trustno1", "000000", "111111",
]

# ✅ Common years (birth years, etc.)
COMMON_YEARS = [str(y) for y in range(1900, 2030)]


class StrengthLevel(str, Enum):
    """5 strength levels."""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


# ✅ Level → emoji
LEVEL_EMOJI = {
    StrengthLevel.VERY_WEAK.value: "🔴",
    StrengthLevel.WEAK.value: "🟠",
    StrengthLevel.MEDIUM.value: "🟡",
    StrengthLevel.STRONG.value: "🟢",
    StrengthLevel.VERY_STRONG.value: "💚",
}

# ✅ Level → progress (0-100)
LEVEL_PROGRESS = {
    StrengthLevel.VERY_WEAK.value: 15,
    StrengthLevel.WEAK.value: 35,
    StrengthLevel.MEDIUM.value: 55,
    StrengthLevel.STRONG.value: 75,
    StrengthLevel.VERY_STRONG.value: 100,
}


@dataclass
class RuleCheck:
    """Result of a single rule check."""
    key: str                # e.g. "length", "uppercase"
    passed: bool
    current: int = 0        # current count
    required: int = 0       # required count

    @property
    def display(self) -> str:
        """e.g. 'Length (8/8)'"""
        if self.required > 1:
            return f"{self.key} ({self.current}/{self.required})"
        return self.key


@dataclass
class StrengthResult:
    """Result of password strength calculation."""
    score: int                              # 0-100
    level: str                              # StrengthLevel value
    rules: List[RuleCheck] = field(default_factory=list)
    entropy: float = 0.0
    feedback: List[str] = field(default_factory=list)  # e.g. ["Too short", ...]

    @property
    def emoji(self) -> str:
        return LEVEL_EMOJI.get(self.level, "")

    @property
    def progress(self) -> int:
        return LEVEL_PROGRESS.get(self.level, 0)

    @property
    def is_valid(self) -> bool:
        """True if all rules pass."""
        return all(r.passed for r in self.rules)


def calculate_strength(password: str) -> StrengthResult:
    """
    Calculate password strength (pure function, no side effects).

    Args:
        password: the password string

    Returns:
        StrengthResult with score, level, rules, feedback
    """
    # ✅ Handle empty
    if not password:
        return StrengthResult(
            score=0,
            level=StrengthLevel.VERY_WEAK.value,
            rules=_build_rules(password),
            entropy=0.0,
            feedback=[],
        )

    # ✅ Check rules
    rules = _build_rules(password)
    all_passed = all(r.passed for r in rules)

    # ✅ Calculate score
    score = _calculate_score(password, rules)

    # ✅ Map score to level
    level = _score_to_level(score)

    # ✅ Calculate entropy (Shannon approximation)
    entropy = _calculate_entropy(password)

    # ✅ Build feedback
    feedback = _build_feedback(password, rules, score)

    return StrengthResult(
        score=score,
        level=level,
        rules=rules,
        entropy=entropy,
        feedback=feedback,
    )


# ============================================================
# Rule checks
# ============================================================

def _build_rules(password: str) -> List[RuleCheck]:
    """Check all rules against password."""
    length = len(password)
    lowercase = len(re.findall(r"[a-z]", password))
    uppercase = len(re.findall(r"[A-Z]", password))
    digits = len(re.findall(r"\d", password))
    special = len(re.findall(
        r"[!@#$%^&*(),.?\":{}|<>_\-+=\[\]\\/;`~]", password
    ))

    return [
        RuleCheck(
            key="length",
            passed=(MIN_LENGTH <= length <= MAX_LENGTH),
            current=length,
            required=MIN_LENGTH,
        ),
        RuleCheck(
            key="lowercase",
            passed=(lowercase >= MIN_LOWERCASE),
            current=lowercase,
            required=MIN_LOWERCASE,
        ),
        RuleCheck(
            key="uppercase",
            passed=(uppercase >= MIN_UPPERCASE),
            current=uppercase,
            required=MIN_UPPERCASE,
        ),
        RuleCheck(
            key="digits",
            passed=(digits >= MIN_DIGITS),
            current=digits,
            required=MIN_DIGITS,
        ),
        RuleCheck(
            key="special",
            passed=(special >= MIN_SPECIAL),
            current=special,
            required=MIN_SPECIAL,
        ),
    ]


# ============================================================
# Score calculation
# ============================================================

def _calculate_score(password: str, rules: List[RuleCheck]) -> int:
    """
    Calculate score 0-100.

    - Base: length (0-50)
    - Variety: (0-30)
    - Length bonus: (0-20)
    - Penalties for patterns
    """
    length = len(password)
    score = 0.0

    # ✅ Base score from length (0-50)
    # 0 chars → 0, 8 chars → 20, 12 chars → 35, 16+ chars → 50
    score += min(50, (length / MAX_LENGTH) * 50)

    # ✅ Variety bonus (0-30)
    has_lower = any(c.islower() for c in password)
    has_upper = any(c.isupper() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in SPECIAL_CHARS for c in password)
    variety_count = sum([has_lower, has_upper, has_digit, has_special])
    score += (variety_count / 4) * 30

    # ✅ Length bonus (0-20)
    if length >= 12:
        score += 20
    elif length >= 10:
        score += 10
    elif length >= 8:
        score += 5

    # ✅ Penalty: common patterns
    pwd_lower = password.lower()
    for pattern in COMMON_PATTERNS:
        if pattern in pwd_lower:
            score -= 25
            break

    # ✅ Penalty: common years
    for year in COMMON_YEARS:
        if year in password:
            score -= 10
            break

    # ✅ Penalty: sequences (1234, abcd, qwerty)
    if _has_sequence(password):
        score -= 15

    # ✅ Penalty: repeats (aaaa, 1111)
    if _has_repeats(password, min_repeat=3):
        score -= 10

    # ✅ Hard penalty: if any rule fails → cap score
    if not all(r.passed for r in rules):
        # Missing characters → can't be better than medium
        missing_count = sum(1 for r in rules if not r.passed)
        if missing_count >= 2:
            score = min(score, 35)  # cap at WEAK
        else:
            score = min(score, 55)  # cap at MEDIUM

    return max(0, min(100, int(round(score))))


def _score_to_level(score: int) -> str:
    """Map score (0-100) to level."""
    if score < 20:
        return StrengthLevel.VERY_WEAK.value
    if score < 40:
        return StrengthLevel.WEAK.value
    if score < 60:
        return StrengthLevel.MEDIUM.value
    if score < 80:
        return StrengthLevel.STRONG.value
    return StrengthLevel.VERY_STRONG.value


# ============================================================
# Entropy (Shannon, bits)
# ============================================================

def _calculate_entropy(password: str) -> float:
    """
    Approximate entropy in bits using Shannon's formula.
    """
    import math
    if not password:
        return 0.0
    freq = {}
    for c in password:
        freq[c] = freq.get(c, 0) + 1
    length = len(password)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy * length  # total bits


# ============================================================
# Pattern detection
# ============================================================

def _has_sequence(password: str) -> bool:
    """Detect ascending/descending sequences (length >= 4)."""
    if len(password) < 4:
        return False
    pwd_lower = password.lower()
    for i in range(len(pwd_lower) - 3):
        chunk = pwd_lower[i:i + 4]
        # ✅ ascending: 'abcd', '1234'
        if all(ord(chunk[j + 1]) - ord(chunk[j]) == 1 for j in range(3)):
            return True
        # ✅ descending: 'dcba', '4321'
        if all(ord(chunk[j + 1]) - ord(chunk[j]) == -1 for j in range(3)):
            return True
    return False


def _has_repeats(password: str, min_repeat: int = 3) -> bool:
    """Detect repeated characters (e.g. 'aaa', '1111')."""
    if len(password) < min_repeat:
        return False
    for i in range(len(password) - min_repeat + 1):
        chunk = password[i:i + min_repeat]
        if len(set(chunk)) == 1:
            return True
    return False


# ============================================================
# Feedback
# ============================================================

def _build_feedback(
    password: str, rules: List[RuleCheck], score: int
) -> List[str]:
    """
    Build human-readable feedback (list of issues).

    Returns short strings that the UI can translate.
    """
    feedback = []

    for rule in rules:
        if not rule.passed:
            if rule.key == "length":
                if rule.current < MIN_LENGTH:
                    missing = MIN_LENGTH - rule.current
                    feedback.append(f"length_short:{missing}")
                else:
                    overflow = rule.current - MAX_LENGTH
                    feedback.append(f"length_long:{overflow}")
            elif rule.key == "lowercase":
                feedback.append("need_lowercase")
            elif rule.key == "uppercase":
                feedback.append("need_uppercase")
            elif rule.key == "digits":
                missing = MIN_DIGITS - rule.current
                feedback.append(f"need_digits:{missing}")
            elif rule.key == "special":
                missing = MIN_SPECIAL - rule.current
                feedback.append(f"need_special:{missing}")

    # ✅ Pattern warnings (only if valid rules, to avoid noise)
    if all(r.passed for r in rules):
        pwd_lower = password.lower()
        for pattern in COMMON_PATTERNS:
            if pattern in pwd_lower:
                feedback.append(f"common_pattern:{pattern}")
                break
        for year in COMMON_YEARS:
            if year in password:
                feedback.append(f"common_year:{year}")
                break
        if _has_sequence(password):
            feedback.append("has_sequence")
        if _has_repeats(password):
            feedback.append("has_repeats")

    return feedback
