"""
Reaction Classifier (v2.0)
===========================

Multi-dimensional reaction analysis for the Asymmetric RL Layer.

Standard sentiment classifiers fail catastrophically on conversational
reactions because they treat profanity as inherently negative and miss
contextual meaning entirely:
  - "fuck yeah" → classified negative (WRONG — strong positive)
  - "I guess that works" → classified positive (WRONG — lukewarm/disappointed)
  - "..." → no signal (WRONG — disengagement = negative)

This module implements a TWO-TIER classification system:

Tier 1: PatternClassifier (microseconds)
  - Contextual compound patterns, not single-word sentiment
  - Profanity is scored ONLY in context with surrounding words
  - Handles ~60% of reactions with high confidence

Tier 2: LLMReactionClassifier (hundreds of ms)
  - Activated when Tier 1 confidence is below threshold
  - Few-shot prompted LLM that understands conversational nuance
  - Returns structured multi-dimensional signal

Output: ReactionSignal with valence, intensity, directedness, confidence.
The final scalar r for the RL layer is: r = valence * intensity * directedness_weight.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass

import torch


@dataclass
class ReactionSignal:
    """
    Multi-dimensional reaction signal for the RL layer.

    Attributes:
        valence:      [-1, 1] positive/negative feeling
        intensity:    [0, 1] strength of the reaction (0 = no reaction, 1 = extreme)
        directedness: [0, 1] how much this is about the AI's output vs general expression
        confidence:   [0, 1] classifier's confidence in this reading
        interpretation: human-readable reasoning
        tier:         which classifier tier produced this (1=pattern, 2=LLM)
    """

    valence: float = 0.0
    intensity: float = 0.0
    directedness: float = 0.5
    confidence: float = 0.0
    interpretation: str = ""
    tier: int = 0

    def to_reward(self) -> float:
        """
        Convert multi-dimensional signal to scalar reward r in [-1, 1]
        for the asymmetric RL layer.

        The formula:
          r = valence * intensity * directedness_weight

        Where directedness_weight blends between full (1.0 for directed feedback)
        and reduced (0.3 for general expression not about the AI's output).
        """
        directedness_weight = 0.3 + 0.7 * self.directedness
        r = self.valence * self.intensity * directedness_weight
        return max(-1.0, min(1.0, r))

    def to_tensor(self) -> torch.Tensor:
        """Return the scalar reward as a single-element tensor."""
        return torch.tensor(self.to_reward(), dtype=torch.float32)


# =============================================================================
# TIER 1: Pattern-Based Contextual Classifier
# =============================================================================

# Compound patterns: (regex, valence, intensity, directedness, confidence)
# These are tested IN ORDER — first match wins
COMPOUND_PATTERNS: list[tuple[str, float, float, float, float, str]] = [
    # ---- EXPLETIVE INFIXATION (literature: always positive intensifier) ----
    # "abso-fucking-lutely", "fan-fucking-tastic", "un-fucking-believable"
    (
        r"\b\w+-fuck(ing|in)-\w+\b",
        1.0,
        0.95,
        0.7,
        0.95,
        "Expletive infixation -- always a strong positive intensifier (Byrne 2014)",
    ),
    (
        r"\b\w+-damn-\w+\b",
        0.8,
        0.7,
        0.6,
        0.85,
        "Mild expletive infixation -- positive intensifier",
    ),
    # ---- STRONG POSITIVE (profanity as emphasis) ----
    (
        r"\bfuck\s*(yeah|yes|yea|yep)\b",
        1.0,
        0.95,
        0.7,
        0.95,
        "Strong positive -- profanity as enthusiastic emphasis",
    ),
    (
        r"\bhell\s*(yeah|yes|yea|yep)\b",
        1.0,
        0.85,
        0.7,
        0.95,
        "Strong positive -- 'hell yeah' is enthusiastic",
    ),
    (
        r"\bfuck(ing|in)?\s*(amazing|awesome|great|perfect|beautiful|brilliant|incredible|sick|fire)\b",
        1.0,
        0.95,
        0.8,
        0.95,
        "Strong positive -- profanity amplifying positive adjective",
    ),
    (
        r"\bholy\s*(shit|fuck|crap)\b.*\b(good|great|amazing|awesome|nice|perfect)\b",
        1.0,
        0.9,
        0.8,
        0.9,
        "Strong positive -- exclamatory surprise at quality",
    ),
    (
        r"\bholy\s*(shit|fuck|crap)\b",
        0.0,
        0.8,
        0.6,
        0.4,
        "Ambiguous exclamation -- needs LLM (tier 2)",
    ),  # low confidence -> falls through
    # ---- POSITIVE WITH CONTEXT ----
    (
        r"\b(that'?s?|this\s+is)\s+(sick|fire|lit|dope|based|goated)\b",
        1.0,
        0.8,
        0.9,
        0.9,
        "Positive slang -- modern approval",
    ),
    (
        r"\bno\s+way\b.*\b(awesome|amazing|cool|great|sick|perfect)\b",
        1.0,
        0.85,
        0.8,
        0.9,
        "'No way' as positive surprise amplifier",
    ),
    (
        r"\blet'?s?\s+(go|gooo+)\b",
        1.0,
        0.85,
        0.7,
        0.9,
        "Positive excitement -- 'let's go'",
    ),
    (r"\bnailed\s+it\b", 1.0, 0.8, 0.95, 0.95, "Direct positive about output quality"),
    (r"\b(love|loved)\s+(it|this|that)\b", 1.0, 0.85, 0.9, 0.95, "Direct positive"),
    (r"\bgood\s+(job|work|stuff)\b", 0.8, 0.6, 0.9, 0.9, "Direct positive feedback"),
    # ---- HEDGING / LUKEWARM (MUST come before standalone positives) ----
    (r"\bi\s+guess\b", -0.1, 0.3, 0.8, 0.8, "Hedging -- lukewarm, mildly disappointed"),
    (
        r"\b(it'?s?|that'?s?)\s+(ok|okay|fine|alright|whatever)\b",
        -0.1,
        0.25,
        0.8,
        0.8,
        "Damning with faint praise -- functional but disappointing",
    ),
    (r"\b(meh|eh)\b", -0.2, 0.3, 0.7, 0.8, "Explicit disinterest"),
    (
        r"\bwhatever\b",
        -0.2,
        0.3,
        0.6,
        0.75,
        "Dismissive -- disinterest or mild frustration",
    ),
    # ---- NEGATIONS (MUST come before standalone positives) ----
    (
        r"\bnot\s+(great|good|ideal|amazing|what\s+i)\b",
        -0.4,
        0.5,
        0.9,
        0.85,
        "Direct mild negative -- 'not good' / 'not what I wanted'",
    ),
    (
        r"\bnot\s+(bad|terrible)\b",
        0.3,
        0.3,
        0.7,
        0.75,
        "Double negative -- 'not bad' is mildly positive",
    ),
    # ---- STANDALONE POSITIVES (after negation patterns) ----
    (r"\bperfect\b", 0.9, 0.7, 0.85, 0.85, "Strong approval"),
    (
        r"\b(great|awesome|amazing|excellent|fantastic|wonderful|brilliant)\b",
        0.85,
        0.65,
        0.75,
        0.8,
        "Positive adjective",
    ),
    (r"\bnice\b", 0.6, 0.4, 0.7, 0.75, "Mild positive approval"),
    (
        r"\bthat\s+works\b",
        0.5,
        0.3,
        0.85,
        0.8,
        "Functional approval -- meets requirements but not exciting",
    ),
    (
        r"\b(thanks?|thank\s+you|ty|thx)\b",
        0.6,
        0.4,
        0.8,
        0.85,
        "Gratitude -- generally positive",
    ),
    (r"\b(yes|yep|yup|yeah|yea)\s*[!.]*$", 0.5, 0.4, 0.7, 0.7, "Simple affirmation"),
    (
        r"\blmao\b",
        0.2,
        0.4,
        0.5,
        0.5,
        "Amusement -- mildly positive but ambiguous without context",
    ),
    (r"\blol\b", 0.15, 0.3, 0.4, 0.5, "Light amusement -- mildly positive"),
    # ---- MIXED SIGNAL / AMUSED CRITICISM (Force to LLM) ----
    (
        r"\b(lmf?ao|lol|haha).*(wrong|bad|terrible)|(wrong|bad|terrible).*(lmf?ao|lol|haha|funny)\b",
        -0.2,
        0.5,
        0.8,
        0.4,
        "Amused criticism -- negative but softened by humor. Forcing to LLM.",
    ),
    # ---- NEGATIVE (mild to moderate) ----
    (r"\bwhat\s*\?\s*$", -0.3, 0.4, 0.7, 0.6, "Confused/frustrated response"),
    (r"\bwrong\b", -0.5, 0.6, 0.9, 0.8, "Direct negative -- output is incorrect"),
    (r"\bno+\s*[.!]*$", -0.3, 0.4, 0.7, 0.65, "Simple negation -- mild negative"),
    (r"\bnope\b", -0.3, 0.4, 0.75, 0.75, "Casual negation"),
    (
        r"\b(bad|terrible|horrible|awful|trash|garbage)\b",
        -0.8,
        0.75,
        0.85,
        0.85,
        "Strong negative adjective",
    ),
    (
        r"\b(hate|hated)\s+(it|this|that)\b",
        -0.9,
        0.85,
        0.9,
        0.9,
        "Direct strong negative",
    ),
    (r"\buseless\b", -0.8, 0.8, 0.95, 0.9, "Strong negative about utility"),
    (r"\bstop\b", -0.5, 0.6, 0.8, 0.75, "Directive to halt -- frustration indicator"),
    # ---- STRONG NEGATIVE (profanity as aggression) ----
    (
        r"\bfuck\s+(you|off|this|that)\b",
        -0.85,
        0.9,
        0.9,
        0.85,
        "Directed profanity -- strong negative",
    ),
    (
        r"\bfuck\s*no\b",
        -0.3,
        0.5,
        0.7,
        0.5,
        "Ambiguous -- could be emphatic refusal or mild. Low confidence.",
    ),
    (
        r"\bfuck\b",
        0.0,
        0.6,
        0.5,
        0.3,
        "Isolated profanity -- truly ambiguous. Needs context (tier 2).",
    ),
    (r"\bshit\b", 0.0, 0.4, 0.5, 0.3, "Isolated mild profanity -- ambiguous"),
    (r"\bwtf\b", -0.2, 0.5, 0.6, 0.5, "Ambiguous -- could be negative or amazed"),
    # ---- MODERN SLANG / INTERNET SPEAK ----
    (
        r"\bno\s*cap\b",
        0.5,
        0.6,
        0.7,
        0.80,
        "Positive slang -- 'no cap' = 'no lie' = genuine approval",
    ),
    (
        r"\bthat'?s?\s+cap\b",
        -0.4,
        0.5,
        0.8,
        0.80,
        "Negative slang -- 'cap' = 'that's a lie'",
    ),
    (
        r"\bfire\b",
        0.8,
        0.7,
        0.8,
        0.85,
        "Positive slang -- 'fire' = excellent/impressive",
    ),
    (
        r"\b(lit|slaps?|bussin|goated)\b",
        0.8,
        0.7,
        0.8,
        0.85,
        "Positive slang -- modern approval",
    ),
    (r"\bslay(ed)?\b", 0.7, 0.65, 0.75, 0.80, "Positive slang -- 'slay' = did great"),
    (
        r"\b(dead|dying|i'?m\s*dead)\b",
        0.6,
        0.7,
        0.5,
        0.70,
        "Slang amusement -- 'dead' = laughing hard (positive)",
    ),
    (
        r"\bcrying\b",
        0.4,
        0.5,
        0.5,
        0.55,
        "Ambiguous emotional -- could be positive amusement or negative",
    ),
    (
        r"\bmid\b",
        -0.3,
        0.4,
        0.8,
        0.80,
        "Negative slang -- 'mid' = mediocre/unremarkable",
    ),
    (
        r"\bbruh\b",
        -0.15,
        0.35,
        0.6,
        0.65,
        "Mild exasperation -- 'bruh' signals mild negative surprise",
    ),
    (r"\b(big\s+)?[Ww]\b", 0.7, 0.6, 0.7, 0.75, "Positive slang -- 'W' = win"),
    (
        r"\b(big\s+)?[Ll]\b",
        -0.6,
        0.6,
        0.7,
        0.75,
        "Negative slang -- 'L' = loss/failure",
    ),
    (
        r"\bbased\b",
        0.6,
        0.5,
        0.7,
        0.75,
        "Positive slang -- 'based' = admirably true to oneself",
    ),
    (
        r"\bratio\b",
        -0.5,
        0.5,
        0.7,
        0.70,
        "Negative slang -- 'ratio' = you're wrong / unpopular take",
    ),
    (
        r"\bbet\b",
        0.3,
        0.3,
        0.5,
        0.65,
        "Positive acknowledgment -- 'bet' = agreed/understood",
    ),
    # ---- AMUSEMENT ----
    (
        r"\b(lmf?ao|rofl)\b",
        0.5,
        0.6,
        0.4,
        0.75,
        "Strong amusement -- laughing hard (positive)",
    ),
    (r"\b(haha|hehe|heh)\b", 0.3, 0.4, 0.4, 0.70, "Light amusement -- positive"),
    (r"\blol\b", 0.15, 0.3, 0.4, 0.50, "Light amusement -- mildly positive"),
    # ---- DISENGAGEMENT SIGNALS ----
    (r"^\.{2,}$", -0.3, 0.4, 0.5, 0.7, "Silence/ellipsis -- disengagement signal"),
    (
        r"^[.!?]{1,3}$",
        -0.2,
        0.3,
        0.5,
        0.5,
        "Minimal punctuation response -- possible disengagement",
    ),
    (r"^k$", -0.15, 0.2, 0.6, 0.7, "Single 'k' -- dismissive/disengaged"),
    (
        r"^(ok|okay)$",
        0.0,
        0.15,
        0.6,
        0.65,
        "Bare acknowledgment -- neutral but disengaged",
    ),
]


class PatternClassifier:
    """
    Tier 1: Fast contextual pattern matching.

    Uses compound regex patterns that score profanity IN CONTEXT rather
    than treating individual words as inherently positive or negative.

    Returns ReactionSignal with confidence. If confidence < threshold,
    the orchestrator should escalate to Tier 2 (LLM).
    """

    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.patterns = [
            (re.compile(pattern, re.IGNORECASE), v, i, d, c, interp)
            for pattern, v, i, d, c, interp in COMPOUND_PATTERNS
        ]

    def classify(self, text: str) -> ReactionSignal:
        """
        Classify a reaction using contextual pattern matching.

        Post-pattern modifiers (literature-backed):
        - ALL CAPS: +0.15 intensity (shouting = stronger emotion)
        - Character repetition (yesss, nooo): +0.10 intensity
        - Exclamation marks: +0.05 per mark (emphatic punctuation)

        Args:
            text: The user's reaction text

        Returns:
            ReactionSignal with confidence indicating reliability
        """
        text = text.strip()
        if not text:
            return ReactionSignal(
                valence=-0.2,
                intensity=0.3,
                directedness=0.4,
                confidence=0.6,
                interpretation="Empty response -- mild disengagement",
                tier=1,
            )

        # Compute intensity modifiers BEFORE lowering case for pattern match
        intensity_boost = self._compute_intensity_modifiers(text)

        for (
            regex,
            valence,
            intensity,
            directedness,
            confidence,
            interp,
        ) in self.patterns:
            if regex.search(text):
                # Apply intensity modifiers from CAPS, repetition, punctuation
                modified_intensity = min(1.0, intensity + intensity_boost)
                modifiers_str = ""
                if intensity_boost > 0:
                    modifiers_str = f" [+{intensity_boost:.2f} intensity from emphasis]"
                return ReactionSignal(
                    valence=valence,
                    intensity=modified_intensity,
                    directedness=directedness,
                    confidence=confidence,
                    interpretation=interp + modifiers_str,
                    tier=1,
                )

        # No pattern matched — unknown reaction
        return ReactionSignal(
            valence=0.0,
            intensity=0.3,
            directedness=0.5,
            confidence=0.2,
            interpretation="No pattern matched -- needs LLM analysis",
            tier=1,
        )

    @staticmethod
    def _compute_intensity_modifiers(text: str) -> float:
        """
        Compute intensity boost from emphasis signals (literature-backed).

        Three signals validated by research:
        1. ALL CAPS: Shouting = stronger emotion (+0.15)
           - Only if >50% of alpha chars are uppercase and len > 2
        2. Character repetition: "yesss", "nooo" (+0.10)
           - Detected via 3+ consecutive identical characters
        3. Exclamation marks: Emphatic punctuation (+0.05 per mark, max +0.15)

        Returns:
            float: intensity boost to add (0.0 to 0.40 max)
        """
        boost = 0.0

        # 1. ALL CAPS detection (>50% uppercase, length > 2)
        alpha_chars = [c for c in text if c.isalpha()]
        if len(alpha_chars) > 2:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if upper_ratio > 0.5:
                boost += 0.15

        # 2. Character repetition (3+ same char in a row: "yesss", "nooo", "!!!")
        if re.search(r"(.)\1{2,}", text):
            boost += 0.10

        # 3. Exclamation marks (+0.05 per mark, capped at +0.15)
        excl_count = text.count("!")
        if excl_count > 0:
            boost += min(0.15, excl_count * 0.05)

        return min(0.40, boost)  # Hard cap at 0.40 total boost

    def is_confident(self, signal: ReactionSignal) -> bool:
        """Check if this signal is confident enough to use directly."""
        return signal.confidence >= self.confidence_threshold


# =============================================================================
# TIER 2: LLM-Based Contextual Classifier
# =============================================================================

# The few-shot prompt template for the LLM classifier
LLM_CLASSIFIER_PROMPT = """You are analyzing a user's reaction to an AI assistant's response.

CRITICAL RULES:
- Profanity is NOT inherently negative. "Fuck yeah" is strongly positive. "Holy shit that's good" is positive.
- "I guess" and "it's fine" are lukewarm/disappointed, not positive.
- Silence, single characters, or "..." indicate disengagement (mildly negative).
- "No way" can be positive surprise OR negative refusal — use surrounding context.
- Consider the WHOLE reaction, not individual words.

Output exactly this JSON (no other text):
{"valence": <float -1 to 1>, "intensity": <float 0 to 1>, "directedness": <float 0 to 1>, "interpretation": "<brief reasoning>"}

Where:
- valence: -1 (extremely negative) to 1 (extremely positive)
- intensity: 0 (no reaction) to 1 (extreme reaction)
- directedness: 0 (general expression) to 1 (specifically about the AI's output)

EXAMPLES:

User reaction: "fuck yeah"
{"valence": 0.95, "intensity": 0.9, "directedness": 0.7, "interpretation": "Strong positive — profanity as enthusiastic emphasis"}

User reaction: "I guess that works"
{"valence": -0.15, "intensity": 0.3, "directedness": 0.85, "interpretation": "Lukewarm acceptance — meets minimum but user is disappointed"}

User reaction: "fuck no"
{"valence": -0.2, "intensity": 0.5, "directedness": 0.6, "interpretation": "Mild emphatic rejection — profanity is casual emphasis, not rage"}

User reaction: "that's fire bro"
{"valence": 0.9, "intensity": 0.8, "directedness": 0.9, "interpretation": "Strong positive with modern slang"}

User reaction: "what the fuck is this"
{"valence": -0.7, "intensity": 0.8, "directedness": 0.95, "interpretation": "Frustrated confusion about the output"}

User reaction: "..."
{"valence": -0.3, "intensity": 0.35, "directedness": 0.5, "interpretation": "Disengagement — silence indicates dissatisfaction without explicit criticism"}

User reaction: "wrong"
{"valence": -0.6, "intensity": 0.65, "directedness": 0.95, "interpretation": "Direct factual correction — the output is incorrect"}

User reaction: "no way, that's actually perfect"
{"valence": 0.95, "intensity": 0.85, "directedness": 0.95, "interpretation": "Positive surprise — 'no way' amplifies the positivity of 'perfect'"}

User reaction: "lmao you're so wrong it's funny"
{"valence": -0.3, "intensity": 0.5, "directedness": 0.9, "interpretation": "Amused criticism — the output is wrong but user finds it funny, softening the negativity"}

User reaction: "ok"
{"valence": 0.0, "intensity": 0.1, "directedness": 0.5, "interpretation": "Bare acknowledgment — neutral, minimal engagement"}

Now analyze this reaction:

User reaction: "{reaction}"
"""


class LLMReactionClassifier:
    """
    Tier 2: LLM-based nuanced reaction classification.

    Uses a language model to analyze ambiguous reactions that the
    pattern classifier can't handle confidently. Supports both
    local models (via transformers) and API models.

    The LLM receives a carefully crafted few-shot prompt with
    examples covering all the edge cases: profanity as emphasis,
    sarcasm, disengagement, lukewarm approval, etc.
    """

    def __init__(self, model_name: str = "local", api_url: str = None):
        """
        Args:
            model_name: "local" for transformers pipeline, or API model name
            api_url: URL for API-based models (e.g. localhost:11434 for Ollama)
        """
        self.model_name = model_name
        self.api_url = api_url
        self._pipeline = None

    def _get_pipeline(self):
        """Lazy-load the transformers pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline

                self._pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    max_new_tokens=200,
                    do_sample=False,
                )
            except (ImportError, OSError) as e:
                print(f"[ReactionClassifier] LLM pipeline unavailable: {e}")
                return None
        return self._pipeline

    def classify(self, reaction_text: str, context: str = "") -> ReactionSignal:
        """
        Classify a reaction using LLM inference.

        Args:
            reaction_text: The user's reaction
            context: Optional context (the AI's previous response)

        Returns:
            ReactionSignal with tier=2
        """
        prompt = LLM_CLASSIFIER_PROMPT.replace("{reaction}", reaction_text)

        # Try API first (Ollama, vLLM, etc.)
        if self.api_url:
            return self._classify_api(prompt, reaction_text)

        # Try local pipeline
        pipe = self._get_pipeline()
        if pipe is not None:
            return self._classify_pipeline(pipe, prompt, reaction_text)

        # Fallback: return a neutral signal with low confidence
        return ReactionSignal(
            valence=0.0,
            intensity=0.3,
            directedness=0.5,
            confidence=0.1,
            interpretation="LLM unavailable — defaulting to neutral",
            tier=2,
        )

    def _classify_api(self, prompt: str, reaction_text: str) -> ReactionSignal:
        """Classify via API call (Ollama, vLLM, OpenAI-compatible)."""
        try:
            import urllib.request

            payload = json.dumps(
                {
                    "model": (
                        self.model_name if self.model_name != "local" else "llama3"
                    ),
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 200},
                }
            )
            req = urllib.request.Request(
                f"{self.api_url}/api/generate",
                data=payload.encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return self._parse_llm_output(result.get("response", ""), reaction_text)
        except Exception as e:
            return ReactionSignal(
                valence=0.0,
                intensity=0.3,
                directedness=0.5,
                confidence=0.1,
                interpretation=f"API call failed: {e}",
                tier=2,
            )

    def _classify_pipeline(
        self, pipe, prompt: str, reaction_text: str
    ) -> ReactionSignal:
        """Classify via local transformers pipeline."""
        try:
            output = pipe(prompt)[0]["generated_text"]
            # Extract just the generated part (after our prompt)
            generated = output[len(prompt) :]
            return self._parse_llm_output(generated, reaction_text)
        except Exception as e:
            return ReactionSignal(
                valence=0.0,
                intensity=0.3,
                directedness=0.5,
                confidence=0.1,
                interpretation=f"Pipeline inference failed: {e}",
                tier=2,
            )

    def _parse_llm_output(self, text: str, reaction_text: str) -> ReactionSignal:
        """Parse the LLM's JSON output into a ReactionSignal."""
        try:
            # Safely capture JSON dict block handling nested elements securely
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return ReactionSignal(
                    valence=max(-1.0, min(1.0, float(data.get("valence", 0.0)))),
                    intensity=max(0.0, min(1.0, float(data.get("intensity", 0.3)))),
                    directedness=max(
                        0.0, min(1.0, float(data.get("directedness", 0.5)))
                    ),
                    confidence=0.85,  # LLM responses get decent confidence
                    interpretation=data.get("interpretation", "LLM analysis"),
                    tier=2,
                )
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Failed to parse — return low-confidence neutral
        return ReactionSignal(
            valence=0.0,
            intensity=0.3,
            directedness=0.5,
            confidence=0.1,
            interpretation=f"Failed to parse LLM output for: '{reaction_text}'",
            tier=2,
        )


# =============================================================================
# Orchestrator: Two-Tier Reaction Classifier
# =============================================================================


class ReactionClassifier:
    """
    Two-tier reaction classification system.

    Tier 1 (PatternClassifier): Fast regex-based contextual matching.
    Tier 2 (LLMReactionClassifier): Falls back to LLM for ambiguous cases.

    Usage:
        classifier = ReactionClassifier()
        signal = classifier.classify("fuck yeah")
        r = signal.to_reward()  # → ~0.63 (strong positive)

    The output ReactionSignal contains:
      - valence: positive/negative [-1, 1]
      - intensity: reaction strength [0, 1]
      - directedness: about the AI's output? [0, 1]
      - confidence: classifier's confidence [0, 1]
      - interpretation: human-readable reasoning
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        llm_model: str = None,
        llm_api_url: str = None,
    ):
        self.pattern_classifier = PatternClassifier(confidence_threshold)

        # Auto-detect Ollama on localhost if no explicit config
        if llm_model is None and llm_api_url is None:
            ollama_model, ollama_url = self._detect_ollama()
            llm_model = ollama_model or "local"
            llm_api_url = ollama_url
        elif llm_model is None:
            llm_model = "local"

        self.llm_classifier = LLMReactionClassifier(llm_model, llm_api_url)

    @staticmethod
    def _detect_ollama() -> tuple:
        """
        Auto-detect a running Ollama instance on localhost.

        Probes http://localhost:11434/api/tags to see what models are available.
        Prefers smaller models (faster response) that can handle text classification.

        Returns:
            (model_name, api_url) or (None, None) if not available
        """
        try:
            import urllib.request

            req = urllib.request.Request(
                "http://localhost:11434/api/tags",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models", [])
                if not models:
                    return None, None

                # Preference order: small fast models first
                model_names = [m.get("name", "") for m in models]

                # Prefer Q4 quantized or smaller models for fast inference
                preferred = [
                    n
                    for n in model_names
                    if "q4" in n.lower() or "3b" in n.lower() or "1b" in n.lower()
                ]
                if preferred:
                    chosen = preferred[0]
                elif model_names:
                    # Pick the first available model
                    chosen = model_names[0]
                else:
                    return None, None

                return chosen, "http://localhost:11434"
        except Exception:
            return None, None

    def classify(self, reaction_text: str, context: str = "") -> ReactionSignal:
        """
        Classify a user reaction into a multi-dimensional signal.

        Tries Tier 1 (fast patterns) first. If confidence is below
        threshold, escalates to Tier 2 (LLM).

        Args:
            reaction_text: The user's reaction text
            context: Optional — the AI's previous response (for LLM context)

        Returns:
            ReactionSignal with valence, intensity, directedness, confidence
        """
        # Short-circuit empty text unconditionally to prevent LLM hallucination
        if not reaction_text or not reaction_text.strip():
            return ReactionSignal(
                valence=0.0,
                intensity=0.0,
                directedness=0.0,
                confidence=1.0,
                interpretation="Empty string bypass",
                tier=1,
            )

        # Tier 1: Fast pattern matching
        signal = self.pattern_classifier.classify(reaction_text)

        if self.pattern_classifier.is_confident(signal):
            return signal

        # Tier 2: LLM for ambiguous cases
        llm_signal = self.llm_classifier.classify(reaction_text, context)

        # If LLM also fails, blend with pattern signal
        if llm_signal.confidence < signal.confidence:
            return signal

        return llm_signal

    def classify_to_reward(self, reaction_text: str, context: str = "") -> float:
        """
        Convenience: classify and return scalar reward directly.

        Args:
            reaction_text: The user's reaction
            context: Optional AI context

        Returns:
            r in [-1, 1] for the asymmetric RL layer
        """
        return self.classify(reaction_text, context).to_reward()
