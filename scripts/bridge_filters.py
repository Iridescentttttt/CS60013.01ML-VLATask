"""Trajectory filters for the course BridgeData V2 subset."""

from __future__ import annotations

import tensorflow as tf


def _instruction(trajectory: dict) -> tf.Tensor:
    """Return one normalized instruction from an RLDS trajectory."""
    language = tf.reshape(trajectory["language_instruction"], [-1])[0]
    return tf.strings.lower(tf.strings.strip(language))


def course_task_filter(trajectory: dict) -> tf.Tensor:
    """Keep placement trajectories related to the two Task-B skills.

    The two groups are object-on-surface and object-in-container. Other skills
    remain available in the downloaded shard trial but are not used to train.
    """
    text = _instruction(trajectory)
    on_surface = tf.strings.regex_full_match(
        text,
        r".*\b(on|onto|on top of)\b.*\b(towel|cloth|plate|tray|table|surface)\b.*",
    )
    in_container = tf.strings.regex_full_match(
        text,
        r".*\b(in|into|inside)\b.*\b(basket|bowl|pot|pan|container|bin|cup)\b.*",
    )
    return tf.logical_or(on_surface, in_container)


def classify_instruction(text: str) -> str:
    """Python equivalent used for manifests and reports."""
    normalized = " ".join(text.lower().strip().split())
    surfaces = ("towel", "cloth", "plate", "tray", "table", "surface")
    containers = ("basket", "bowl", "pot", "pan", "container", "bin", "cup")
    if any(token in normalized for token in (" on ", " onto ", "on top of")) and any(
        target in normalized for target in surfaces
    ):
        return "on_surface"
    if any(token in normalized for token in (" in ", " into ", " inside ")) and any(
        target in normalized for target in containers
    ):
        return "in_container"
    return "other"
