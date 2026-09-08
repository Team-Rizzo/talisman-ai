"""Numbers the floor must read: spelled out, and written in other locales.

The benchmark run of 2026-09-02 measured grounding at 0.63-0.68 against every model.
Roughly an eighth of honest claims were numbers plainly in the article that the parser
did not read, concentrated on non-English articles — which are the corpus's
highest-gold slice. Since the floor gates paid volume, that is denied credit for real
work.
"""

import pytest

from alpharidge_ai.oracle import floor


def values(text):
    return [round(n.value, 4) for n in floor.parse_numbers(floor.normalize(text).text)]


def units(text):
    return {n.unit for n in floor.parse_numbers(floor.normalize(text).text)}


# ---- spelled-out numbers -----------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Six children were rescued.", 6),
    ("It lasted twenty-five years.", 25),
    ("Three thousand returns were filed.", 3000),
    ("Two hundred and fifty thousand barrels.", 250_000),
    ("Nineteen people attended.", 19),
    ("Forty-two units sold.", 42),
])
def test_english_number_words_are_read(text, expected):
    assert expected in values(text)


@pytest.mark.parametrize("text,expected", [
    ("Ventas de tres millones de unidades.", 3_000_000),
    ("Cerca de dois mil clientes.", 2000),
    ("Environ trois millions d euros.", 3_000_000),
    ("Etwa zwei millionen einheiten.", 2_000_000),
])
def test_number_words_in_other_languages_are_read(text, expected):
    assert expected in values(text)


def test_digits_still_win_where_both_appear():
    assert 2 in values("Two ambulances arrived and 2 ambulances left.")


# ---- a scale word alone is not a number --------------------------------------------

def test_a_bare_scale_word_emits_nothing():
    """Reading 'billions' as 1e9 would ground that claim against any article using it."""
    assert values("Losses ran into the billions.") == []
    assert values("The company is worth millions.") == []


def test_a_scale_with_a_unit_still_reads():
    assert 1_000_000 in values("One million shares changed hands.")


# ---- locale formats ----------------------------------------------------------------

def test_a_decimal_comma_is_one_number_not_two():
    got = values("Le prix atteint 17,99 euros.")
    assert 17.99 in got
    assert not (17 in got and 99 in got and 17.99 not in got)


def test_dot_grouping_is_read_as_thousands():
    assert 1_203_400 in values("Ventas de 1.203.400 unidades.")


def test_comma_grouping_still_reads_as_thousands():
    assert 1203 in values("Revenue of 1,203 units.")


def test_an_ambiguous_mark_yields_both_readings():
    """1,203 is one-two-oh-three or one-point-two-oh-three depending on the writer."""
    got = values("The figure was 1,203 today.")
    assert 1203 in got and 1.203 in got


def test_mixed_marks_resolve_by_the_rightmost():
    assert 1234.56 in values("Total 1,234.56 dollars.")
    assert 1234.56 in values("Total 1.234,56 euros.")


# ---- currency words ----------------------------------------------------------------

@pytest.mark.parametrize("text,code", [
    ("It cost 17.99 euros.", "currency:EUR"),
    ("Priced at 20 dollars.", "currency:USD"),
    ("Worth 500 reais.", "currency:BRL"),
    ("About 300 pounds.", "currency:GBP"),
])
def test_currency_words_are_recognised(text, code):
    assert code in units(text)


def test_a_percentage_is_still_a_percentage():
    assert "pct" in units("A queda de 1,5% no periodo.")


# ---- grounding, the reason all of this matters -------------------------------------

@pytest.mark.parametrize("text,claim,unit", [
    ("Six children were rescued from the building.", 6.0, "count"),
    ("Le chiffre atteint 17,99 millions d euros.", 17_990_000.0, "EUR"),
    ("Ventas de 1.203.400 unidades este ano.", 1_203_400.0, "count"),
    ("Two hundred and fifty thousand barrels shipped.", 250_000.0, "count"),
])
def test_an_honest_claim_now_grounds(text, claim, unit):
    import types
    numbers = floor.parse_numbers(floor.normalize(text).text)
    stub = types.SimpleNamespace(value=claim, unit=unit)
    assert floor.ground_claim(stub, numbers)


def test_an_invented_number_still_does_not_ground():
    import types
    numbers = floor.parse_numbers(floor.normalize("Six children were rescued.").text)
    assert not floor.ground_claim(types.SimpleNamespace(value=9_400.0, unit="count"),
                                  numbers)


def test_the_extra_readings_do_not_ground_arbitrary_values():
    import types
    numbers = floor.parse_numbers(floor.normalize("The figure was 1,203 today.").text)
    for wrong in (12.03, 120.3, 1_203_000.0, 99.0):
        assert not floor.ground_claim(
            types.SimpleNamespace(value=wrong, unit="count"), numbers), wrong


# ---- CJK scale marks ---------------------------------------------------------------
# These attach to a digit with no space, so a digits-only reading is wrong by the scale
# rather than merely short: 3억 is three hundred million, not three. Korean is the
# corpus's highest gold-rate language.

@pytest.mark.parametrize("text,expected,unit", [
    ("매출은 3억 원을 기록했다.", 300_000_000, "currency:KRW"),
    ("지난해 12만 명이 방문했다.", 120_000, "count"),
    ("売上高は3億円だった。", 300_000_000, "currency:JPY"),
    ("营收达 5亿元。", 500_000_000, "currency:CNY"),
])
def test_cjk_scaled_numbers_are_read(text, expected, unit):
    assert expected in values(text)
    assert unit in units(text)


def test_chained_cjk_scales_multiply():
    assert 200_000_000_000 in values("거래액 2천억 원")


def test_a_compound_figure_offers_its_sum():
    """1조 2천억 is one figure written in descending parts."""
    got = values("거래액 1조 2천억 원")
    assert 1_200_000_000_000 in got
    assert 1_000_000_000_000 in got          # the parts remain candidates too


def test_a_cjk_claim_grounds():
    import types
    numbers = floor.parse_numbers(floor.normalize("매출은 3억 원을 기록했다.").text)
    assert floor.ground_claim(
        types.SimpleNamespace(value=300_000_000.0, unit="KRW"), numbers)


def test_the_unscaled_digit_no_longer_stands_alone():
    """Reading 3억 as 3 would ground a claim of three against a figure of 300 million."""
    import types
    numbers = floor.parse_numbers(floor.normalize("매출은 3억 원을 기록했다.").text)
    assert not floor.ground_claim(
        types.SimpleNamespace(value=3.0, unit="KRW"), numbers)


def test_ascii_numbers_are_unaffected_by_the_cjk_path():
    got = values("Revenue was $1.2 billion, up 12.5%.")
    assert 1_200_000_000 in got and 12.5 in got


# ---- Slavic and Germanic, the third round ------------------------------------------
# The 2026-09-03 re-grounding put the remaining parser residue at ~5-6% of claims,
# dominated by Slavic spelled-out numbers, with a smaller Germanic magnitude gap.

@pytest.mark.parametrize("text,expected", [
    ("Три человека погибли.", 3),
    ("Выручка два миллиона рублей.", 2_000_000),
    ("Около пять тысяч человек.", 5000),
    ("Tři lidé byli zraněni.", 3),
    ("Dwa miliony klientow.", 2_000_000),
    ("Две хиляди души.", 2000),
])
def test_slavic_number_words_are_read(text, expected):
    assert expected in values(text)


@pytest.mark.parametrize("text,expected", [
    ("Ongeveer 25,7 miljoen euro.", 25_700_000),
    ("Een miljard klanten.", 1_000_000_000),
    ("Rund 20 Millionen Barrel.", 20_000_000),
    ("Etwa 3 Milliarden Euro.", 3_000_000_000),
])
def test_germanic_magnitude_words_are_read(text, expected):
    assert expected in values(text)


def test_the_english_billion_keeps_its_english_value():
    """Billion is 1e12 in German and 1e9 in English; the shared spelling stays English."""
    assert 1_200_000_000 in values("Revenue was $1.2 billion.")
    assert 1_200_000_000_000 not in values("Revenue was $1.2 billion.")


# ---- a magnitude named inside the claim's own unit ---------------------------------

def test_a_claim_scaled_by_its_unit_grounds():
    """Models split a figure across the fields: value 25.7, unit "million euros"."""
    import types
    numbers = floor.parse_numbers(floor.normalize("Omzet van 25,7 miljoen euro.").text)
    assert floor.ground_claim(
        types.SimpleNamespace(value=25.7, unit="million euros"), numbers)


def test_the_unscaled_reading_still_grounds():
    import types
    numbers = floor.parse_numbers(floor.normalize("Precies 25.7 procent.").text)
    assert floor.ground_claim(types.SimpleNamespace(value=25.7, unit="count"), numbers)


def test_unit_magnitude_reads_the_scale():
    assert floor.unit_magnitude("million euros") == 1e6
    assert floor.unit_magnitude("billion USD") == 1e9
    assert floor.unit_magnitude("USD") == 1.0
    assert floor.unit_magnitude(None) == 1.0


def test_a_scaled_unit_still_carries_its_currency():
    import types
    numbers = floor.parse_numbers(floor.normalize("Omzet van 25,7 miljoen euro.").text)
    assert floor.ground_claim(
        types.SimpleNamespace(value=25.7, unit="million euros"), numbers)
    assert not floor.ground_claim(
        types.SimpleNamespace(value=25.7, unit="million dollars"), numbers)


def test_scaling_does_not_ground_an_unrelated_value():
    import types
    numbers = floor.parse_numbers(floor.normalize("Omzet van 25,7 miljoen euro.").text)
    assert not floor.ground_claim(
        types.SimpleNamespace(value=99.9, unit="million euros"), numbers)
