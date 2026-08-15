# -*- coding: utf-8 -*-
"""Clinical facts, and the code book's own rules for resolving them.

The problem this solves: one clinical concept (ER, say) is described by
several documents that do not agree, and the code book does NOT say "pick the
newest" or "pick the most confident" -- it gives a DIFFERENT rule per field:

    breast SSF1 (ER)        侵襲癌優先於原位癌；原發優先於轉移；單顆多份報告取
                            手術切除最大體積者；多顆取比例最高者
    breast SSF6 (Nottingham) 同上，但多顆取「分數」最高
    breast SSF9 (LVI)       任一份原發部位病理報告記錄有 LVI 即為 010（OR）
    head & neck SSF1        同區域依序 病理 > 手術紀錄 > 影像；
                            不同區域取最大徑（極值，不是來源優先序）
    區域淋巴結侵犯數          同一 lymph node chain 不可加總，不同 chain 可加總

So "different sources complement each other" is true, but it is not one merge
function -- it is one rule per field, and the rules are in the manual.

The model here keeps every OBSERVATION and records which one each field
CHOSE and why, instead of collapsing to a single value at import time:

    Observation   one thing one document said, with its specimen, timing,
                  provenance and confidence
    Resolution    what a given registry field selected, under which rule,
                  and why every other observation was rejected

A Resolution feeds the encoders (facts -> TCR code, facts -> QBC code), so
two submission targets read the same observations and never convert into
each other. See docs/fact_layer_refactor.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary. Deliberately small: every value here is something the code book
# actually distinguishes.
# ─────────────────────────────────────────────────────────────────────────────

SPECIMEN_INVASIVE = 'invasive'
SPECIMEN_IN_SITU = 'in_situ'

SITE_PRIMARY = 'primary'
SITE_METASTATIC = 'metastatic'

TIMING_PRE = 'pre_treatment'
TIMING_POST_NEOADJUVANT = 'post_neoadjuvant'
TIMING_UNKNOWN = 'unknown'

PROCEDURE_RESECTION = 'resection'
PROCEDURE_BIOPSY = 'biopsy'

# Document types, most authoritative first for the fields whose rule is a
# source-priority rule (head & neck SSF1 states this order explicitly).
DOC_PATHOLOGY = 'pathology'
DOC_OPERATIVE = 'operative'
DOC_IMAGING = 'imaging'
DOC_PHYSICIAN_STATEMENT = 'physician_statement'
DOC_PRIOR_SUBMISSION = 'prior_submission'   # 既有 QBC/癌登申報值
DOC_ONCOTYPE = 'oncotype'

DOC_PRIORITY = (DOC_PATHOLOGY, DOC_OPERATIVE, DOC_IMAGING,
                DOC_PHYSICIAN_STATEMENT, DOC_PRIOR_SUBMISSION)

HOSPITAL_REPORTING = 'reporting'     # 申報醫院
HOSPITAL_EXTERNAL = 'external'       # 外院


@dataclass(frozen=True)
class Evidence:
    """Where an observation came from, precise enough to re-read."""
    document_id: str
    document_type: str = DOC_PATHOLOGY
    text: Optional[str] = None
    span: Optional[Tuple[int, int]] = None
    page: Optional[int] = None
    hospital: str = HOSPITAL_REPORTING


@dataclass
class Observation:
    """One thing one document said. NOT an answer to a registry field."""
    concept: str
    value: Any
    qualifiers: Dict[str, Any] = dc_field(default_factory=dict)
    specimen: Optional[str] = None
    site: Optional[str] = None
    timing: str = TIMING_UNKNOWN
    procedure: Optional[str] = None
    specimen_volume_mm: Optional[float] = None
    tumor_id: Optional[str] = None
    doc_type: str = DOC_PATHOLOGY
    hospital: str = HOSPITAL_REPORTING
    method: str = 'ai_extraction'
    confidence: float = 1.0
    evidence: List[Evidence] = dc_field(default_factory=list)

    def describe(self) -> str:
        bits = [f'{self.concept}={self.value}']
        if self.qualifiers:
            bits.append(', '.join(f'{k}={v}' for k, v in self.qualifiers.items()))
        for label, attr in (('標本', 'specimen'), ('部位', 'site'),
                            ('時間', 'timing'), ('處置', 'procedure')):
            got = getattr(self, attr)
            if got and got != TIMING_UNKNOWN:
                bits.append(f'{label}={got}')
        if self.specimen_volume_mm:
            bits.append(f'體積={self.specimen_volume_mm}mm')
        bits.append(f'來源={self.doc_type}')
        return '｜'.join(bits)


@dataclass
class Resolution:
    """What one registry field selected from the observations, and why."""
    field: str
    concept: str
    chosen: Optional[Observation] = None
    forced_code: Optional[str] = None      # rule dictates a code directly
    rule_id: str = ''
    trace: List[str] = dc_field(default_factory=list)
    rejected: List[Tuple[Observation, str]] = dc_field(default_factory=list)
    needs_review: bool = False
    review_reason: Optional[str] = None

    def explain(self) -> str:
        lines = [f'[{self.field}] 規則 {self.rule_id}']
        if self.forced_code:
            lines.append(f'  → 直接編碼 {self.forced_code}')
        elif self.chosen:
            lines.append(f'  → 採用：{self.chosen.describe()}')
        else:
            lines.append('  → 無可用觀察')
        lines.extend(f'  · {step}' for step in self.trace)
        for observation, reason in self.rejected:
            lines.append(f'  ✗ 排除 {observation.describe()}｜理由：{reason}')
        if self.needs_review:
            lines.append(f'  ⚠ 需人工複核：{self.review_reason}')
        return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Rule steps. Each one narrows the candidate set and says what it dropped.
# ─────────────────────────────────────────────────────────────────────────────

Step = Callable[[List[Observation], Resolution], List[Observation]]


def drop_doc_types(*doc_types: str, reason: str) -> Step:
    def step(observations, resolution):
        keep, drop = [], []
        for observation in observations:
            (drop if observation.doc_type in doc_types else keep).append(observation)
        for observation in drop:
            resolution.rejected.append((observation, reason))
        return keep
    return step


def prefer_where(predicate: Callable[[Observation], bool], label: str,
                 reject_reason: str) -> Step:
    """Keep only matching observations -- but only if any match at all.

    This is the shape of almost every code-book rule: "侵襲癌及原位癌皆有檢測
    時，優先摘錄侵襲癌之數值，當侵襲癌未做檢測時，則摘錄原位癌數值."
    """
    def step(observations, resolution):
        matching = [o for o in observations if predicate(o)]
        if not matching or len(matching) == len(observations):
            return observations
        resolution.trace.append(label)
        for observation in observations:
            if observation not in matching:
                resolution.rejected.append((observation, reject_reason))
        return matching
    return step


def prefer_doc_priority(order: Sequence[str] = DOC_PRIORITY,
                        label: str = '依來源優先序') -> Step:
    def step(observations, resolution):
        if len(observations) <= 1:
            return observations
        ranked = {doc: i for i, doc in enumerate(order)}
        best = min(ranked.get(o.doc_type, len(order)) for o in observations)
        keep = [o for o in observations
                if ranked.get(o.doc_type, len(order)) == best]
        if len(keep) != len(observations):
            resolution.trace.append(f'{label}（{order[best]} 優先）')
            for observation in observations:
                if observation not in keep:
                    resolution.rejected.append(
                        (observation, f'來源優先序低於 {order[best]}'))
        return keep
    return step


def prefer_max(key: Callable[[Observation], Any], label: str,
               reject_reason: str) -> Step:
    def step(observations, resolution):
        usable = [o for o in observations if key(o) is not None]
        if len(usable) <= 1:
            return observations
        best = max(key(o) for o in usable)
        keep = [o for o in usable if key(o) == best]
        if len(keep) != len(observations):
            resolution.trace.append(label)
            for observation in observations:
                if observation not in keep:
                    resolution.rejected.append((observation, reject_reason))
        return keep
    return step


def force_code_if(predicate: Callable[[List[Observation]], bool], code: str,
                  rule_ref: str) -> Step:
    """Some situations map to a code directly, not to an observation.

    ER is the clearest case: a case whose only ER value is post-neoadjuvant is
    coded 111/121, and one that converted negative -> positive is 888. No
    single observation carries that meaning; the SET does.
    """
    def step(observations, resolution):
        if resolution.forced_code is None and predicate(observations):
            resolution.forced_code = code
            resolution.trace.append(f'{rule_ref} → 直接編碼 {code}')
        return observations
    return step


def flag_for_review(predicate: Callable[[List[Observation]], bool],
                    reason: str) -> Step:
    def step(observations, resolution):
        if predicate(observations):
            resolution.needs_review = True
            resolution.review_reason = reason
        return observations
    return step


@dataclass
class ResolutionRule:
    """One registry field's rule, transcribed from the code book."""
    field: str
    concept: str
    rule_id: str
    citation: str
    steps: List[Step]
    tie_break_note: str = '仍有多筆等價觀察時，取第一筆並標記人工複核'

    def resolve(self, observations: Sequence[Observation]) -> Resolution:
        resolution = Resolution(field=self.field, concept=self.concept,
                                rule_id=self.rule_id)
        resolution.trace.append(f'碼冊出處：{self.citation}')
        candidates = [o for o in observations if o.concept == self.concept]
        for other in observations:
            if other.concept != self.concept:
                continue
        for step in self.steps:
            candidates = step(list(candidates), resolution)
            if resolution.forced_code:
                break
        if resolution.forced_code:
            return resolution
        if not candidates:
            resolution.needs_review = True
            resolution.review_reason = resolution.review_reason or '沒有可用的觀察'
            return resolution
        if len(candidates) > 1:
            resolution.needs_review = True
            resolution.review_reason = (
                f'碼冊規則走完仍有 {len(candidates)} 筆等價觀察，需人工判斷')
        resolution.chosen = candidates[0]
        for observation in candidates[1:]:
            resolution.rejected.append((observation, '與採用值等價，未被規則區分'))
        return resolution


def resolve(rule: ResolutionRule,
            observations: Sequence[Observation]) -> Resolution:
    return rule.resolve(observations)
