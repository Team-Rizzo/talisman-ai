"""The deterministic floor: what every article is checked against, with no model.

Pure Python and regex. Every validator must reach the same verdict on the same inputs,
so nothing here samples, times out, or calls a model. Results feed both the audit and
the volume credit: an article that fails the floor earns nothing.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from rapidfuzz import fuzz

# Grounding tolerances.
VALUE_REL_TOLERANCE = 0.005
QUOTE_MATCH_RATIO = 0.85
QUOTE_WINDOW_SLACK = 20
QUOTE_MAX_CHARS = 1000
QUOTE_MAX_OVERLAP = 0.50
SPAN_MIN_CHARS = 12
SPAN_MIN_ADJACENT_WORDS = 3
TEXT_STATS_TOLERANCE = 0.01

_MAGNITUDES = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mn": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
    "t": 1e12, "tn": 1e12, "trillion": 1e12,
    # Indian numbering, ordinary in financial copy and absent from the short scale.
    "lakh": 1e5, "lakhs": 1e5, "crore": 1e7, "crores": 1e7,
}

BASIS_POINT = 0.01  # of a percentage point

_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR"}
_CURRENCY_CODES = {"usd", "eur", "gbp", "jpy", "cny", "inr", "chf", "cad", "aud", "krw"}
_CURRENCY_PROXIMITY = 12

_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-", "‑": "-",
    " ": " ", " ": " ", " ": " ", "​": "",
    "…": "...",
}

# A numeric run: digits with any mix of grouping and decimal marks. The marks are
# interpreted afterwards, because their meaning depends on the writer's locale.
_NUMBER_RE = re.compile(r"[-+]?\d[\d.,\u00a0 ]*\d|[-+]?\d")

# Number words. English in full; other languages carry the units and scales that appear
# in financial copy. The corpus's best-served articles are not English, so a
# digits-only parser reads them worst.
_WORD_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    # es / pt
    "uno": 1, "una": 1, "um": 1, "uma": 1, "dos": 2, "dois": 2, "duas": 2,
    "tres": 3, "tris": 3, "cuatro": 4, "quatro": 4, "cinco": 5, "seis": 6,
    "siete": 7, "sete": 7, "ocho": 8, "oito": 8, "nueve": 9, "nove": 9,
    "diez": 10, "dez": 10, "veinte": 20, "vinte": 20, "treinta": 30, "trinta": 30,
    # fr
    "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "sept": 7,
    "huit": 8, "neuf": 9, "dix": 10, "vingt": 20, "trente": 30, "quarante": 40,
    "cinquante": 50,
    # de / it
    "eins": 1, "zwei": 2, "drei": 3, "vier": 4, "funf": 5, "sechs": 6, "sieben": 7,
    "acht": 8, "neun": 9, "zehn": 10, "zwanzig": 20, "dreissig": 30,
    "due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6, "otto": 8, "nove_it": 9,
    # ru / uk / bg
    "один": 1, "одна": 1, "одно": 1, "два": 2, "две": 2, "три": 3, "чотири": 4,
    "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9,
    "десять": 10, "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "едно": 1, "двама": 2, "трима": 3, "чотири_uk": 4, "п'ять": 5,
    # cs / sk / pl
    "jedna": 1, "jeden": 1, "dva": 2, "dvě": 2, "dwa": 2, "tři": 3, "trzy": 3,
    "čtyři": 4, "cztery": 4, "pět": 5, "pięć": 5, "šest": 6, "sześć": 6,
    "sedm": 7, "siedem": 7, "osm": 8, "osiem": 8, "devět": 9, "dziewięć": 9,
    "deset": 10, "dziesięć": 10,
    # nl / de indefinite-article numerals: "een miljard" is "a billion".
    "een": 1, "ein": 1, "eine": 1,
}

_WORD_SCALES = {
    "hundred": 100, "thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12,
    "cien": 100, "ciento": 100, "cem": 100, "cento": 100, "cent": 100, "hundert": 100,
    "mil": 1e3, "mille": 1e3, "tausend": 1e3,
    "millon": 1e6, "millones": 1e6, "milhao": 1e6, "milhoes": 1e6, "million_fr": 1e6,
    "millions": 1e6, "millionen": 1e6, "milione": 1e6, "milioni": 1e6,
    "billon": 1e9, "bilhao": 1e9, "bilhoes": 1e9, "milliard": 1e9, "milliards": 1e9,
    "miliardi": 1e9, "billions": 1e9,
    # nl
    "duizend": 1e3, "miljoen": 1e6, "miljard": 1e9,
    # de. "Billion" is 1e12 in German and 1e9 in English, so only the unambiguous
    # German plural is listed; the shared spelling keeps its English value.
    "billionen": 1e12,
    # ru / uk / bg
    "сто": 100, "тысяча": 1e3, "тысячи": 1e3, "тысяч": 1e3, "тисяча": 1e3,
    "хиляда": 1e3, "хиляди": 1e3,
    "миллион": 1e6, "миллиона": 1e6, "миллионов": 1e6, "мільйон": 1e6, "милион": 1e6,
    "миллиард": 1e9, "миллиарда": 1e9, "миллиардов": 1e9, "мільярд": 1e9,
    "милиард": 1e9,
    # cs / sk / pl
    "sto": 100, "tisíc": 1e3, "tysiąc": 1e3, "tysiące": 1e3, "tysięcy": 1e3,
    "milion": 1e6, "milionů": 1e6, "milionów": 1e6, "miliona": 1e6,
    "miliarda": 1e9, "miliard": 1e9, "miliardy": 1e9,
    # Slavic and Germanic decline; the inflected forms are what appears in copy.
    "miliony": 1e6, "milionu": 1e6, "miliardů": 1e9, "miliardów": 1e9,
    "milionami": 1e6, "milionach": 1e6,
    "milliarden": 1e9, "millionens": 1e6, "miljoenen": 1e6, "miljarden": 1e9,
}

_JOINERS = {"and", "y", "e", "et", "und", "en", "i", "а", "и", "-"}

# CJK scale marks. These follow a digit with no space — 3억 is three hundred million,
# not three — so a digits-only reading is wrong by a factor of the scale rather than
# merely incomplete. Korean is the corpus's highest gold-rate language.
_CJK_SCALES = {
    "십": 1e1, "百": 1e2, "백": 1e2, "千": 1e3, "천": 1e3,
    "万": 1e4, "萬": 1e4, "만": 1e4,
    "億": 1e8, "亿": 1e8, "억": 1e8,
    "兆": 1e12, "조": 1e12,
}

_CJK_CURRENCY = {"원": "KRW", "円": "JPY", "元": "CNY", "圓": "CNY", "위안": "CNY"}

_CURRENCY_WORDS = {
    "dollar": "USD", "dollars": "USD", "usd": "USD", "dolar": "USD", "dolares": "USD",
    "euro": "EUR", "euros": "EUR", "eur": "EUR",
    "pound": "GBP", "pounds": "GBP", "sterling": "GBP", "libras": "GBP",
    "yen": "JPY", "yuan": "CNY", "renminbi": "CNY", "won": "KRW",
    "real": "BRL", "reais": "BRL", "franc": "CHF", "francs": "CHF",
}
_QUOTED_RE = re.compile(r'"[^"]+"')
_SENTENCE_END = re.compile(r"[.!?]")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_TRAILING_PUNCT = ".,;:!?'\"-)]}"


# ---- normalisation ----------------------------------------------------------------

@dataclass(frozen=True)
class Normalized:
    """Normalised text plus the offset map back into the original."""
    text: str
    offsets: Tuple[int, ...]

    def to_original(self, start: int, end: int) -> Tuple[int, int]:
        """Map a span in normalised space back to original character offsets."""
        if not self.offsets:
            return (0, 0)
        start = max(0, min(start, len(self.offsets) - 1))
        end = max(start + 1, min(end, len(self.offsets)))
        first = self.offsets[start]
        last = self.offsets[end - 1] + 1
        return (first, last)


def normalize(text: Optional[str]) -> Normalized:
    """NFKC, punctuation folding, lowercase, whitespace collapse, offsets preserved."""
    if not text:
        return Normalized("", ())

    out: List[str] = []
    offs: List[int] = []
    prev_space = True  # strips leading whitespace

    for i, ch in enumerate(text):
        mapped = _PUNCT_MAP.get(ch)
        if mapped is None:
            mapped = unicodedata.normalize("NFKC", ch)
        if not mapped:
            continue
        mapped = mapped.lower()
        for c in mapped:
            if c.isspace():
                if prev_space:
                    continue
                out.append(" ")
                offs.append(i)
                prev_space = True
            else:
                out.append(c)
                offs.append(i)
                prev_space = False

    while out and out[-1] == " ":
        out.pop()
        offs.pop()
    return Normalized("".join(out), tuple(offs))


def _strip_edges(s: str) -> str:
    return s.strip().strip(_TRAILING_PUNCT).strip()


# ---- number parsing ---------------------------------------------------------------

@dataclass(frozen=True)
class ParsedNumber:
    value: float
    unit: str          # "pct" | "currency:XXX" | "count" | "none"
    offset: int
    # False when the value was inferred rather than read: a compound sum, or the
    # second reading of an ambiguous grouping mark. Inferred candidates are adjudicated
    # rather than granted.
    literal: bool = True
    # True when this value came from a digit carrying a CJK scale mark. Only such
    # components may be summed into a compound; a scale mark merely sitting nearby is
    # not evidence that these two numbers are one figure.
    scaled: bool = False


def _has_cjk_scale_at(text: str, end: int) -> bool:
    """Whether a number ending at `end` is immediately followed by a scale mark."""
    return end < len(text) and text[end] in _CJK_SCALES


def _cjk_scale_after(text: str, end: int) -> Tuple[float, int]:
    """A run of CJK scale marks immediately after a number, multiplied together."""
    scale, i = 1.0, end
    while i < len(text) and text[i] in _CJK_SCALES:
        scale *= _CJK_SCALES[text[i]]
        i += 1
    return scale, i


def _magnitude_after(text: str, end: int) -> Tuple[float, int]:
    """Magnitude word within the next two tokens, and where it ends."""
    cjk, cjk_end = _cjk_scale_after(text, end)
    if cjk != 1.0:
        return cjk, cjk_end
    tail = text[end:end + 32]
    tokens = list(_WORD_RE.finditer(tail))[:2]
    for tok in tokens:
        word = tok.group(0)
        if word in _MAGNITUDES:
            return _MAGNITUDES[word], end + tok.end()
        if word in _WORD_SCALES and _WORD_SCALES[word] >= 1e3:
            return float(_WORD_SCALES[word]), end + tok.end()
    return 1.0, end


def _currency_near(text: str, start: int, end: int) -> Optional[str]:
    left = text[max(0, start - _CURRENCY_PROXIMITY):start]
    right = text[end:end + _CURRENCY_PROXIMITY]
    for sym, code in _CURRENCY_SYMBOLS.items():
        if sym in left or sym in right:
            return code
    for char in left + right:
        if char in _CJK_CURRENCY:
            return _CJK_CURRENCY[char]
    for word in _WORD_RE.findall(left) + _WORD_RE.findall(right):
        if word in _CURRENCY_CODES:
            return word.upper()
        if word in _CURRENCY_WORDS:
            return _CURRENCY_WORDS[word]
    return None


def _readings(token: str) -> List[float]:
    """Every plausible value for a numeric run.

    `1,203` is one thousand two hundred and three to an English writer and one point
    two zero three to a French one, and the text does not say which, so both are
    emitted. Readings beyond the first are marked inferred.
    """
    raw = token.strip().replace("\u00a0", " ").replace(" ", "")
    sign = -1.0 if raw.startswith("-") else 1.0
    raw = raw.lstrip("+-")
    if not raw or not raw[0].isdigit():
        return []

    dots, commas = raw.count("."), raw.count(",")
    out: List[float] = []

    def add(value: Optional[float]) -> None:
        if value is not None and value == value and value not in out:
            out.append(value)

    def grouped(text: str, group: str, decimal: str) -> Optional[float]:
        """Read `text` treating `group` as the thousands mark and `decimal` as the point."""
        body, _, fraction = (text.rpartition(decimal) if decimal and decimal in text
                             else (text, "", ""))
        body = body or text
        if decimal and decimal in text and (not fraction.isdigit() or not body):
            return None
        parts = body.split(group) if group else [body]
        if len(parts) > 1:
            if not parts[0] or len(parts[0]) > 3:
                return None
            if any(len(p) != 3 or not p.isdigit() for p in parts[1:]):
                return None
        if not "".join(parts).isdigit():
            return None
        try:
            return float("".join(parts) + ("." + fraction if fraction else ""))
        except ValueError:
            return None

    if dots and commas:
        # The rightmost mark is the decimal point; the other groups.
        if raw.rfind(".") > raw.rfind(","):
            add(grouped(raw, ",", "."))
        else:
            add(grouped(raw, ".", ","))
    elif commas:
        add(grouped(raw, ",", ""))          # grouping: 1,203 -> 1203
        add(grouped(raw, "", ","))          # decimal:  17,99 -> 17.99
    elif dots:
        add(grouped(raw, "", "."))          # decimal:  17.99 -> 17.99
        add(grouped(raw, ".", ""))          # grouping: 1.203 -> 1203
    else:
        try:
            add(float(raw))
        except ValueError:
            return []

    return [sign * v for v in out]


def _word_numbers(normalized_text: str) -> List[Tuple[float, int, int]]:
    """Spelled-out numbers, as (value, start, end) over the normalised text."""
    tokens = list(_WORD_RE.finditer(normalized_text))
    found: List[Tuple[float, int, int]] = []
    i = 0
    while i < len(tokens):
        word = tokens[i].group(0)
        if word not in _WORD_UNITS and word not in _WORD_SCALES:
            i += 1
            continue

        total = current = 0.0
        used = False
        saw_unit = False
        start = tokens[i].start()
        end = tokens[i].end()
        j = i
        while j < len(tokens):
            w = tokens[j].group(0)
            if w in _WORD_UNITS:
                current += _WORD_UNITS[w]
                used = saw_unit = True
            elif w in _WORD_SCALES:
                # A scale word with nothing in front of it is a figure of speech, not a
                # number. Reading "billion" alone as 1e9 would ground that exact claim
                # against any article that merely uses the word.
                if not saw_unit:
                    break
                scale = _WORD_SCALES[w]
                if scale == 100:
                    current *= 100
                else:
                    total += current * scale
                    current = 0.0
                used = True
            elif w in _JOINERS and used:
                pass                         # "two hundred and five"
            else:
                break
            end = tokens[j].end()
            j += 1

        value = total + current
        if used and value:
            found.append((value, start, end))
        i = max(j, i + 1)
    return found


def parse_numbers(normalized_text: str) -> List[ParsedNumber]:
    """Every numeric token in the text, with its magnitude, unit class and offset."""
    found: List[ParsedNumber] = []

    def emit(value: float, start: int, end: int, literal: bool = True) -> None:
        cjk_scaled = _has_cjk_scale_at(normalized_text, end)
        mag, mag_end = _magnitude_after(normalized_text, end)
        value *= mag
        after = normalized_text[mag_end:mag_end + 2]
        if after.startswith("%") or after.strip().startswith("%"):
            unit = "pct"
        else:
            code = _currency_near(normalized_text, start, mag_end)
            unit = f"currency:{code}" if code else "count"
        found.append(ParsedNumber(value=value, unit=unit, offset=start,
                                  literal=literal, scaled=cjk_scaled))

    for m in _NUMBER_RE.finditer(normalized_text):
        for i, value in enumerate(_readings(m.group(0))):
            emit(value, m.start(), m.end(), literal=(i == 0))

    for value, start, end in _word_numbers(normalized_text):
        emit(value, start, end)

    # Descending CJK-scaled parts written next to each other are one figure: 1조 2천억
    # is 1.2 trillion. Only where scale marks are actually present — otherwise "100 and
    # 20" would offer 120, which is in no sense in the text. Inferred either way.
    scaled = [n for n in found if n.unit != "pct" and n.literal and n.scaled]
    for a, b in zip(scaled, scaled[1:]):
        span = normalized_text[a.offset:b.offset]
        if not (0 < len(span) <= 8 and a.value > b.value > 0):
            continue
        found.append(ParsedNumber(value=a.value + b.value, unit=a.unit,
                                  offset=a.offset, literal=False))
    return found


def unit_magnitude(unit: Optional[str]) -> float:
    """The scale named inside a unit string.

    Models commonly split a figure between the fields, reporting `value: 25.7` with
    `unit: "million euros"`. The article says 25,700,000, so the claim has to be scaled
    by its own unit before it can be matched.

    A basis point is a hundredth of a percentage point, and both sit in the same unit
    class, so the scale has to be carried here or the two compare equal.
    """
    text = (unit or "").lower().replace("_", " ")
    scale = 1.0
    for word in _WORD_RE.findall(text):
        if word in _MAGNITUDES:
            scale *= _MAGNITUDES[word]
        elif word in _WORD_SCALES and _WORD_SCALES[word] >= 1e3:
            scale *= _WORD_SCALES[word]
    words = set(_WORD_RE.findall(text))
    if words & {"bps", "bp"} or "basis point" in text:
        scale *= BASIS_POINT
    return scale


def _unit_class(unit: Optional[str]) -> str:
    u = (unit or "").strip().lower()
    if not u:
        return "none"
    if u.replace("_", " ") in ("%", "pct", "percent", "percentage", "bps", "bp",
                              "basis point", "basis points"):
        return "pct"
    if u in _CURRENCY_CODES:
        return f"currency:{u.upper()}"
    for sym, code in _CURRENCY_SYMBOLS.items():
        if sym in u:
            return f"currency:{code}"
    # "million euros" names a scale and a currency; the currency is the unit.
    for word in _WORD_RE.findall(u):
        if word in _CURRENCY_CODES:
            return f"currency:{word.upper()}"
        if word in _CURRENCY_WORDS:
            return f"currency:{_CURRENCY_WORDS[word]}"
    return "count"


def _units_compatible(claim_unit: str, text_unit: str) -> bool:
    """A percentage needs a percent token, a currency needs a currency marker."""
    if claim_unit == "pct":
        return text_unit == "pct"
    if claim_unit.startswith("currency:"):
        if not text_unit.startswith("currency:"):
            return False
        return claim_unit == text_unit or text_unit == "currency:"
    return text_unit in ("count", "none") or not text_unit.startswith("currency:")


def _stated_precision(value: float) -> float:
    """Half a unit in the claim's own last significant place."""
    s = repr(float(value))
    if "e" in s or "E" in s:
        return abs(value) * VALUE_REL_TOLERANCE
    if "." in s:
        decimals = len(s.split(".")[1].rstrip("0"))
        return 0.5 * (10.0 ** -decimals) if decimals else 0.5
    trailing = len(s) - len(s.rstrip("0"))
    return 0.5 * (10.0 ** trailing)


def value_grounded(claim_value: float, text_value: float) -> bool:
    """A claim matches a text number within tolerance, or at its own stated precision."""
    if text_value == claim_value:
        return True
    rel = abs(claim_value - text_value) / max(abs(text_value), 1e-9)
    if rel <= VALUE_REL_TOLERANCE:
        return True
    return abs(claim_value - text_value) <= _stated_precision(claim_value)


# ---- claims -----------------------------------------------------------------------

def ground_claim(claim, numbers: Sequence[ParsedNumber]) -> bool:
    """True when the claim matches any parsed number, literal or inferred."""
    return ground_kind(claim, numbers) is not None


def ground_kind(claim, numbers: Sequence[ParsedNumber]):
    """How a claim grounds: "literal", "inferred", or None.

    Only a literal match is evidence the article states the figure. An inferred match
    means the reading depended on a guess this parser made — a locale convention or a
    compound — so it earns adjudication rather than automatic validity.
    """
    try:
        value = float(getattr(claim, "value", None))
    except (TypeError, ValueError):
        return False
    raw_unit = getattr(claim, "unit", None)
    unit = _unit_class(raw_unit)

    scale = unit_magnitude(raw_unit)
    # A unit naming a scale leaves the claim ambiguous: "25.7" with unit "million
    # euros" is either 25.7 million or 25.7 labelled loosely. Which one the submitter
    # meant is a guess, and the submitter wrote the unit — so neither reading is
    # evidence, and both are adjudicated.
    ambiguous = scale != 1.0
    candidates = [value] if not ambiguous else [value, value * scale]

    best = None
    for n in numbers:
        if not _units_compatible(unit, n.unit):
            continue
        if any(value_grounded(c, n.value) for c in candidates):
            if n.literal and not ambiguous:
                return "literal"
            best = "inferred"
    return best


# ---- quotes -----------------------------------------------------------------------

@dataclass(frozen=True)
class AlignedQuote:
    start: int          # original-text offsets of the maximal aligned match
    end: int
    ratio: float


def align_quote(article: Normalized, quote_text: str,
                start_offset: Optional[int], end_offset: Optional[int]) -> Optional[AlignedQuote]:
    """Locate a quote in the article and return its maximal match in original offsets.

    The miner's offsets say where to look, never what the answer is: the returned span
    is the maximal match this validator finds, so two miners quoting the same sentence
    key to the same span.
    """
    if not quote_text or len(quote_text) > QUOTE_MAX_CHARS or not article.text:
        return None

    needle = normalize(quote_text).text
    if not needle:
        return None

    search_lo, search_hi = 0, len(article.text)
    if start_offset is not None and end_offset is not None:
        try:
            lo = int(start_offset)
            hi = int(end_offset)
        except (TypeError, ValueError):
            lo = hi = None
        if lo is not None and hi is not None and hi > lo:
            n_lo = _original_to_normalized(article, lo)
            n_hi = _original_to_normalized(article, hi)
            search_lo = max(0, n_lo - QUOTE_WINDOW_SLACK)
            search_hi = min(len(article.text), n_hi + QUOTE_WINDOW_SLACK)
            if search_hi - search_lo < len(needle):
                search_hi = min(len(article.text), search_lo + len(needle) + QUOTE_WINDOW_SLACK)

    window = article.text[search_lo:search_hi]
    if not window:
        return None

    exact = window.find(needle)
    if exact >= 0:
        s = search_lo + exact
        c_start, c_end = canonical_span(article, s, s + len(needle))
        o_start, o_end = article.to_original(c_start, c_end)
        return AlignedQuote(o_start, o_end, 1.0)

    if len(needle) > len(window):
        return None

    hit = fuzz.partial_ratio_alignment(needle, window)
    if hit is None or hit.score / 100.0 < QUOTE_MATCH_RATIO:
        return None

    s = search_lo + hit.dest_start
    e = search_lo + hit.dest_end
    if e <= s:
        return None
    c_start, c_end = canonical_span(article, s, e)
    o_start, o_end = article.to_original(c_start, c_end)
    return AlignedQuote(o_start, o_end, hit.score / 100.0)


def canonical_span(article: Normalized, start: int, end: int) -> Tuple[int, int]:
    """Widen a match to the passage that contains it, in normalised offsets.

    Two submissions quoting different lengths of the same sentence must key to the same
    span, or an honest partial quote scores as a miss against the grader's fuller one.
    """
    text = article.text
    for m in _QUOTED_RE.finditer(text):
        if m.start() <= start and end <= m.end():
            return (m.start() + 1, m.end() - 1)

    lo = 0
    for m in _SENTENCE_END.finditer(text, 0, start):
        lo = m.end()
    tail = _SENTENCE_END.search(text, max(end - 1, 0))
    hi = tail.start() if tail else len(text)

    while lo < hi and text[lo].isspace():
        lo += 1
    while hi > lo and text[hi - 1].isspace():
        hi -= 1
    return (lo, hi) if hi > lo else (start, end)


def _original_to_normalized(article: Normalized, original_offset: int) -> int:
    """First normalised index at or after an original-text offset."""
    lo, hi = 0, len(article.offsets)
    while lo < hi:
        mid = (lo + hi) // 2
        if article.offsets[mid] < original_offset:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _overlap_fraction(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if hi <= lo:
        return 0.0
    shorter = min(a[1] - a[0], b[1] - b[0])
    return (hi - lo) / max(shorter, 1)


# ---- evidence spans ---------------------------------------------------------------

def _first_attr(item, attrs) -> Optional[str]:
    for attr in attrs:
        value = str(getattr(item, attr, None) or "").strip()
        if value:
            return value
    return None


def asset_form(item) -> Optional[str]:
    """What an asset is called. AssetSentiment carries ticker and asset_name."""
    return _first_attr(item, ("ticker", "asset_name", "symbol", "name"))


def entity_form(item) -> Optional[str]:
    """What an entity is called. ExtractedEntity carries name, and optionally a
    ticker — the name is the identity, since that is what a grader returns."""
    return _first_attr(item, ("name", "canonical_name", "asset_name", "ticker"))


def span_supported(article: Normalized, span: Optional[str],
                   surface_form: Optional[str] = None) -> bool:
    """An evidence span must be in the article and carry enough context to be evidence.

    A bare ticker repeated as its own evidence proves nothing, so a short span has to
    bring adjacent words with it.
    """
    if not span:
        return False
    needle = normalize(span).text
    if not needle or needle not in article.text:
        return False
    if len(needle) >= SPAN_MIN_CHARS:
        return True

    words = _WORD_RE.findall(needle)
    surface = normalize(surface_form).text if surface_form else ""
    if surface:
        extra = [w for w in words if w not in _WORD_RE.findall(surface)]
        return len(extra) >= SPAN_MIN_ADJACENT_WORDS
    return len(words) >= SPAN_MIN_ADJACENT_WORDS + 1


# ---- proof of read ----------------------------------------------------------------

def content_hash(text: Optional[str]) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def text_stats(text: Optional[str]) -> Dict[str, int]:
    t = text or ""
    return {
        "chars": len(t),
        "words": len(_WORD_RE.findall(t)),
        "sentences": len([s for s in re.split(r"[.!?]+", t) if s.strip()]),
    }


def stats_within_tolerance(claimed: Optional[Dict[str, int]],
                           actual: Dict[str, int]) -> bool:
    if not claimed:
        return False
    for key, real in actual.items():
        if key not in claimed:
            return False
        try:
            got = float(claimed[key])
        except (TypeError, ValueError):
            return False
        if abs(got - real) > max(1.0, real * TEXT_STATS_TOLERANCE):
            return False
    return True


# ---- the floor --------------------------------------------------------------------

@dataclass
class FloorResult:
    floor_pass: bool
    reason: str = ""
    grounded: Set[int] = field(default_factory=set)
    inferred: Set[int] = field(default_factory=set)
    ungrounded: Set[int] = field(default_factory=set)
    aligned_quotes: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    rejected_quotes: Set[int] = field(default_factory=set)
    span_failures: Set[int] = field(default_factory=set)
    # Surface forms whose evidence did not hold. Excluded from the scored sets, so a
    # claimed asset with no proof earns nothing rather than being recorded and ignored.
    unevidenced_assets: Set[str] = field(default_factory=set)
    unevidenced_entities: Set[str] = field(default_factory=set)

    @property
    def quote_keys(self) -> Set[Tuple[int, int]]:
        return set(self.aligned_quotes.values())


def evaluate(intel, article_text: str, *,
             claimed_hash: Optional[str] = None,
             expected_hash: Optional[str] = None,
             claimed_stats: Optional[Dict[str, int]] = None,
             claim_cap: int = 40) -> FloorResult:
    """Run the floor over one submission. Deterministic, no model, no network.

    Proof of read fails the article outright. Everything else is per-item: an
    ungrounded claim scores nothing, but it does not void the article.
    """
    article = normalize(article_text)
    if not article.text:
        return FloorResult(False, "empty_article")

    if claimed_hash is not None:
        # `expected_hash` lets the caller supply the canonical hash for its own schema;
        # comparing two different hash definitions would fail every honest article.
        wanted = expected_hash if expected_hash is not None else content_hash(article_text)
        if claimed_hash != wanted:
            return FloorResult(False, "content_hash_mismatch")
    if claimed_stats is not None and not stats_within_tolerance(claimed_stats,
                                                               text_stats(article_text)):
        return FloorResult(False, "text_stats_mismatch")

    result = FloorResult(True, "ok")
    numbers = parse_numbers(article.text)

    for i, claim in enumerate(list(getattr(intel, "numeric_claims", None) or [])[:claim_cap]):
        kind = ground_kind(claim, numbers)
        if kind == "literal":
            result.grounded.add(i)
        elif kind == "inferred":
            result.inferred.add(i)      # plausible, but adjudicated rather than granted
        else:
            result.ungrounded.add(i)

    accepted: List[Tuple[int, int]] = []
    for i, quote in enumerate(list(getattr(intel, "quotes", None) or [])[:claim_cap]):
        hit = align_quote(article, getattr(quote, "text", None),
                          getattr(quote, "start_offset", None),
                          getattr(quote, "end_offset", None))
        if hit is None:
            result.rejected_quotes.add(i)
            continue
        span = (hit.start, hit.end)
        if any(_overlap_fraction(span, prev) > QUOTE_MAX_OVERLAP for prev in accepted):
            result.rejected_quotes.add(i)
            continue
        accepted.append(span)
        result.aligned_quotes[i] = span

    assets = list(getattr(intel, "assets", None) or [])
    for i, item in enumerate(assets + list(getattr(intel, "entities", None) or [])):
        spans = getattr(item, "evidence_spans", None)
        if spans is None:
            single = getattr(item, "evidence_span", None) or getattr(item, "evidence", None)
            spans = [single] if single else []
        if not spans:
            continue
        surface = asset_form(item) if i < len(assets) else entity_form(item)
        # One span that carries context is enough to evidence the item.
        if not any(span_supported(article, s, surface) for s in spans):
            result.span_failures.add(i)
            if surface:
                # E18: a failed span fails that item's correctness gate, not the
                # article. Recorded by surface form so the scorer can drop it.
                (result.unevidenced_assets if i < len(assets)
                 else result.unevidenced_entities).add(str(surface).strip().lower())

    return result
