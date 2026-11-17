import os

from algorithms.enhanced_aho_corasick import EnhancedAhoCorasick, load_dictionary
from data.context_data import (
    AMBIGUOUS_TERMS,
    NEGATIVE_CONTEXT,
    POSITIVE_CONTEXT,
    AMBIGUOUS_MEANINGS,
    HOT_STATE_THRESHOLD,
    CONTEXT_WINDOW_K,
    COMMON_ENGLISH_WORDS,
)


def build_engine(common_words=None):
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "medical_dictionary.csv")
    patterns, meanings = load_dictionary(csv_path)
    return EnhancedAhoCorasick(
        patterns,
        dictionary_meaning=meanings,
        ambiguous_terms=AMBIGUOUS_TERMS,
        negative_context=NEGATIVE_CONTEXT,
        positive_context=POSITIVE_CONTEXT,
        ambiguous_meanings=AMBIGUOUS_MEANINGS,
        hot_threshold=HOT_STATE_THRESHOLD,
        context_window_k=CONTEXT_WINDOW_K,
        common_words=common_words,
    )


def test_microgram_symbols_are_recognized_in_prescription_notes():
    engine = build_engine()
    hits = engine.search("Take 500 µg BID")
    terms = {hit["term"] for hit in hits}
    assert "MCG" in terms
    assert "BID" in terms


def test_frequency_symbol_is_recognized_in_enhanced_matching():
    engine = build_engine()
    hits = engine.search("5 days (1-1-1)")
    terms = {hit["term"] for hit in hits}
    assert "1-1-1" in terms
    assert any(hit["category"] == "Symbol" for hit in hits if hit["term"] == "1-1-1")


def test_ten_units_morning_and_night_schedule_is_recognized():
    engine = build_engine()

    hits = engine.search("10u -0 - 10u")

    assert any(
        hit["term"] == "10U-0-10U"
        and hit["category"] == "Symbol"
        and hit["meaning"] == "Take 10 units in the morning and at night (twice daily)"
        for hit in hits
    )


def test_spaced_frequency_symbol_is_recognized_in_enhanced_matching():
    engine = build_engine()

    hits = engine.search("Amlodipine 5 mg #100 S (0 - 0 - 1)")

    assert any(
        hit["term"] == "0-0-1" and hit["category"] == "Symbol"
        for hit in hits
    )


def test_all_three_time_slot_combinations_are_recognized_in_enhanced_matching():
    engine = build_engine()
    schedules = [f"{morning}-{afternoon}-{night}"
                 for morning in (0, 1)
                 for afternoon in (0, 1)
                 for night in (0, 1)]

    hits = engine.search(" ".join(schedules))
    assert {hit["term"] for hit in hits} >= set(schedules)


def test_hash_symbol_is_recognized_in_enhanced_matching():
    engine = build_engine()

    hits = engine.search("Dolcet tablet #9")

    hash_hits = [hit for hit in hits if hit["term"] == "#9"]
    assert len(hash_hits) == 1
    assert hash_hits[0]["category"] == "Symbol"


def test_hash_symbol_absorbs_adjacent_quantity_in_either_order():
    engine = build_engine()

    assert {hit["term"] for hit in engine.search("#9 9#")} >= {"#9", "9#"}


def test_mg_unit_is_detected_as_an_abbreviation_from_dosage_tokens():
    engine = build_engine()

    hits = engine.search("Imoflox 200mg tablet #19")
    terms = {hit["term"] for hit in hits}

    assert "200MG" in terms
    assert "MG" in terms


def test_decimal_dosage_preserves_the_full_strength():
    engine = build_engine()

    hits = engine.search("carvedilol 6.25mg")
    dosage_terms = {
        hit["term"] for hit in hits
        if hit["category"] == "Dosage"
    }

    assert "6.25MG" in dosage_terms
    assert "25MG" not in dosage_terms


def test_slash_dosage_is_detected_from_spaced_input():
    engine = build_engine()

    hits = engine.search("Ibuprofen 200 mg/5mL")

    assert any(
        hit["term"] == "200MG/5ML" and hit["category"] == "Dosage"
        for hit in hits
    )


def test_dosage_with_capsule_word_without_slash_matches_strength_only():
    engine = build_engine()

    hits = engine.search("Amoxicillin 500 mg cap")
    terms = {hit["term"] for hit in hits}

    assert "500MG" in terms
    assert "500MG/CAP" not in terms


def test_strength_with_tablet_or_capsule_suffix_is_a_dosage_form():
    engine = build_engine()

    hits = engine.search("200mg/cap 500mg/tab")
    categories = {
        hit["term"]: hit["category"]
        for hit in hits
        if hit["term"] in {"200MG/CAP", "500MG/TAB"}
    }

    assert categories == {
        "200MG/CAP": "Dosage Form",
        "500MG/TAB": "Dosage Form",
    }


def test_one_tablet_is_recognized_as_dosage():
    engine = build_engine()

    hits = engine.search("Take one tablet every 12 hours")

    assert any(
        hit["term"] == "ONE TABLET" and hit["category"] == "Dosage Form"
        for hit in hits
    )


def test_one_capsule_is_recognized_as_a_dosage_form():
    engine = build_engine()

    hits = engine.search("Take one capsule once daily")

    assert any(
        hit["term"] == "ONE CAPSULE" and hit["category"] == "Dosage Form"
        for hit in hits
    )


def test_one_tab_is_recognized_as_a_dosage_form():
    engine = build_engine()

    hits = engine.search(
        "Take 1 tab, 1 cap, 1 sachet, 1 patch, 1 spray, "
        "1 suppository, and 1 ampule once daily"
    )

    dosage_forms = {
        hit["term"] for hit in hits if hit["category"] == "Dosage Form"
    }

    assert {"1 TAB", "1 CAP", "1 SACHET"} <= dosage_forms
    assert {
        "PATCH",
        "SPRAY",
        "SUPPOSITORY",
        "AMPULE",
    } <= dosage_forms


def test_topical_solution_is_recognized_as_a_dosage_form():
    engine = build_engine()

    hits = engine.search("Clindamycin cream or topical solution")

    assert any(
        hit["term"] == "TOPICAL SOLUTION"
        and hit["category"] == "Dosage Form"
        for hit in hits
    )


def test_ocr_typo_is_detected_as_a_fuzzy_match():
    engine = build_engine()

    fuzzy_hits = [hit for hit in engine.search("moflox") if hit["match_type"] == "fuzzy"]

    assert any(hit["matched"] == "IMOFLOX" for hit in fuzzy_hits)


def test_prescription_instruction_words_are_not_fuzzy_drugs():
    engine = build_engine()

    hits = engine.search("NASOFLO NASAL SPRAY #1 SIG 2 SPRAYS PER NOSTRIL")

    assert not any(hit.get("matched") == "SUPRAX" for hit in hits)


def test_contextual_abbreviations_are_detected_without_roman_text_false_positive():
    engine = build_engine()

    historical = engine.search("Chapter IV discusses the roman numeral century in old manuscripts")
    clinical_or = engine.search("Patient transferred to OR for emergency surgical procedure under anesthesia")
    clinical_ra = engine.search("History of RA, rheumatoid autoimmune disease confirmed by recent lab tests")

    assert not any(hit.get("matched") == "RIOPAN" for hit in historical)
    assert any(hit["term"] == "OR" for hit in clinical_or)
    assert any(hit["term"] == "RA" for hit in clinical_ra)


def test_supplemental_ambiguous_abbreviations_require_context():
    engine = build_engine()

    for term in ("IV", "IM", "OR"):
        assert not any(hit["term"] == term for hit in engine.search(term))

    assert any(hit["term"] == "IV" for hit in engine.search("IV fluid"))
    assert any(hit["term"] == "IM" for hit in engine.search("IM injection"))
    assert any(hit["term"] == "OR" for hit in engine.search("OR surgery"))
    assert any(hit["term"] == "AC" for hit in engine.search("AC"))
    assert any(hit["term"] == "ET" for hit in engine.search("ET"))


def test_clinical_narrative_words_are_not_fuzzy_drugs_or_treatments():
    engine = build_engine()

    hits = engine.search("Patient placed on RA, oxygen saturation stable via nasal cannula")
    hits += engine.search("The odds ratio OR reported in the regression analysis, 95 percent CI")

    assert not any(hit.get("matched") == "KLACID" for hit in hits)
    assert not any(hit.get("matched") == "DIALYSIS" for hit in hits)


def test_typo_in_multi_word_term_keeps_the_full_medical_term():
    engine = build_engine()

    hits = engine.search("METFORMIN HYDROCLORIDE")

    phrase_hits = [hit for hit in hits if hit["match_type"] == "fuzzy"]
    assert any(
        hit["matched"] == "METFORMIN HYDROCHLORIDE"
        and hit["term"] == "METFORMIN HYDROCLORIDE"
        for hit in phrase_hits
    )
    assert not any(hit["term"] == "METFORMIN" for hit in hits)


def test_meal_marker_glyphs_become_one_ascii_marker_group():
    engine = build_engine()

    hits = engine.search("•-- —-")
    marker_hits = [hit for hit in hits if hit["category"] == "Symbol"]

    assert {hit["term"] for hit in marker_hits} >= {"---", "--"}
    assert all(hit["meaning"] == "Take 1 tablet before/after meal/s" for hit in marker_hits)


def test_bullet_is_matched_directly_before_marker_normalization():
    engine = build_engine()

    hits = engine.search("•")

    bullet_hits = [hit for hit in hits if hit["term"] == "-"]
    assert len(bullet_hits) == 1
    assert bullet_hits[0]["matched"] == "•"


def test_bullet_symbol_remains_in_the_csv_dictionary():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "medical_dictionary.csv")
    patterns, _ = load_dictionary(csv_path)

    assert any(term == "•" and category == "Symbol" for term, category, _ in patterns)
    assert any(term == "—" and category == "Symbol" for term, category, _ in patterns)


def test_platexan_correction_is_detected_as_a_drug():
    from data.ocr_corrections import apply_ocr_corrections
    from data.drug_indications import DRUG_INDICATIONS

    text = apply_ocr_corrections("Clopidogrel 75mg\nPlateson #60\n1 - 0 - 0")
    engine = build_engine()

    assert "Platexan" in text
    assert DRUG_INDICATIONS["PLATEXAN"] == "Prevention of blood clots"
    assert any(
        hit["term"] == "PLATEXAN" and hit["category"] == "Drug"
        for hit in engine.search(text)
    )


def test_insuget_has_a_diabetes_indication():
    from data.drug_indications import DRUG_INDICATIONS

    assert DRUG_INDICATIONS["INSUGET"] == "Diabetes"


def test_brand_drug_indications_are_available():
    from data.drug_indications import DRUG_INDICATIONS

    assert DRUG_INDICATIONS["URINORM"] == "Gout and high uric acid (febuxostat)"
    assert DRUG_INDICATIONS["LIVERPRIME"] == "Liver support"
    assert DRUG_INDICATIONS["GLUMET"] == "Type 2 diabetes (metformin)"
    assert DRUG_INDICATIONS["GLYHART"] == "Type 2 diabetes (dapagliflozin)"
    assert DRUG_INDICATIONS["HUMULIN 70/30"] == "Diabetes (premixed human insulin)"


def test_polymyxin_and_sinupret_indications_are_available():
    from data.drug_indications import DRUG_INDICATIONS

    assert DRUG_INDICATIONS["POLYMYXIN"] == "Bacterial ear and eye infections"
    assert DRUG_INDICATIONS["SINUPRET FORTE"] == "Acute and chronic sinusitis symptom relief"


def test_na_hyaluronate_indication_is_available():
    from data.drug_indications import DRUG_INDICATIONS

    assert DRUG_INDICATIONS["NA HYALURONATE"] == "Dry eye relief and ocular surface lubrication"


def test_before_breakfast_and_supper_is_one_frequency():
    engine = build_engine(common_words=COMMON_ENGLISH_WORDS)

    hits = engine.search("10u - 6u\nbefore breakfast & supper")
    meal_frequency = [
        hit for hit in hits
        if hit["term"] == "BEFORE BREAKFAST & SUPPER"
    ]

    assert len(meal_frequency) == 1
    assert meal_frequency[0]["category"] == "Instruction"
    assert not any(hit.get("matched") == "SUPP" for hit in hits)


def test_analyze_reports_supp_and_single_hyphen_as_dropped_false_positives():
    from app import app as flask_app

    response = flask_app.test_client().post(
        "/api/analyze",
        json={
            "mode": "enhanced",
            "text": "Humulin 70/30 or Insuget 70/30 #2\n"
                    "10u - 6u\n10u \u2014 6u\nbefore breakfast & supper",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    dropped = {
        match["term"]: match
        for match in payload["dropped_matches"]
    }

    assert dropped["SUPP"]["drop_reason"] == "false positive"
    assert dropped["-"]["drop_reason"] == "false positive"
    assert dropped["\u2014"]["drop_reason"] == "false positive"
    assert any(
        match["term"] == "BEFORE BREAKFAST & SUPPER"
        and match["category"] == "Instruction"
        for match in payload["matches"]
    )


def test_analyze_includes_gtt_in_abbreviations_panel_for_composite_dosage():
    from app import app as flask_app

    response = flask_app.test_client().post(
        "/api/analyze",
        json={"mode": "enhanced", "text": "Diquafosol (Diquas) O/S\n1 gtt\nq 2 hrs"},
    )

    assert response.status_code == 200
    abbreviations = response.get_json()["abbreviations"]
    assert {"term": "GTT", "meaning": "Drops (guttae; drop unit, commonly used for eye or ear drops)"} in abbreviations


def test_analyze_reports_fuzzy_nebule_match_as_dropped():
    from app import app as flask_app

    text = (
        "Humulin 70/30\n"
        "or Insuget\n"
        "#1\n"
        "10u—0—10u\n\n"
        "Insulin syringe\n"
        "BD G#31 short needle\n"
        "#50\n\n"
        "OR\n\n"
        "Novomix\n"
        "#10\n"
        "10u—0—10u\n\n"
        "BD pen needle 4mm\n"
        "#50"
    )
    response = flask_app.test_client().post(
        "/api/analyze",
        json={"mode": "enhanced", "text": text},
    )

    assert response.status_code == 200
    payload = response.get_json()
    nebule_matches = [
        match for match in payload["dropped_matches"]
        if match["term"] == "NEBULE"
    ]

    assert len(nebule_matches) == 2
    assert all(match["confidence"] == "fuzzy" for match in nebule_matches)
    assert all(match["drop_reason"] == "fuzzy match" for match in nebule_matches)
    assert not any(
        match["matched_dictionary_term"] == "NEBULE" and not match.get("dropped")
        for match in payload["matches"]
    )


def test_analyze_keeps_fuzzy_drugs_in_identified_terms():
    from app import app as flask_app

    response = flask_app.test_client().post(
        "/api/analyze",
        json={
            "mode": "enhanced",
            "text": "Mupirocin\n+ Bethamethason\n(Foskina-B)\n"
                    "S Apply 2x/day x 7 days",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    identified_drugs = {
        match["matched_dictionary_term"]: match
        for match in payload["matches"]
        if match["category"] == "Drug" and not match.get("dropped")
    }

    assert identified_drugs["BETAMETHASONE"]["confidence"] == "fuzzy"
    assert identified_drugs["FOSKINA-B"]["confidence"] == "exact"
    assert {
        match["instruction"]
        for match in identified_drugs.values()
    } == {1}
    assert not any(
        match["matched_dictionary_term"] in {"BETAMETHASONE", "FOSKINA-B"}
        for match in payload["dropped_matches"]
    )


def test_analyze_keeps_higher_scoring_overlapping_dosage_term():
    from app import app as flask_app

    text = (
        "Dibencozide 3mg\n"
        "(Heraclene forte cap)\n"
        "#30\n"
        "S 1 cap once daily\n\n"
        "Ibuprofen 200 mg/tab\n"
        "S 1 q 6 hrs"
    )
    response = flask_app.test_client().post(
        "/api/analyze",
        json={"mode": "enhanced", "text": text},
    )

    assert response.status_code == 200
    payload = response.get_json()
    identified_one_cap = next(
        match for match in payload["matches"]
        if match["matched_dictionary_term"] == "1 CAP"
    )
    dropped_cap = next(
        match for match in payload["dropped_matches"]
        if match["matched_dictionary_term"] == "CAP"
    )

    assert identified_one_cap["score"] > dropped_cap["score"]
    assert dropped_cap["drop_reason"] == "overlapping"


def test_six_hours_q_and_sodium_symbol_are_recognized():
    engine = build_engine()

    six_hours = engine.search("q 6 hrs")
    sodium = engine.search("Na hyaluronate")

    assert any(hit["term"] == "6 HRS" and hit["category"] == "Frequency" for hit in six_hours)
    assert any(
        hit["term"] == "Q"
        and hit["category"] == "Symbol"
        and hit["meaning"] == "Every or each"
        for hit in six_hours
    )
    assert any(
        hit["term"] == "NA"
        and hit["category"] == "Symbol"
        and hit["meaning"] == "Sodium"
        for hit in sodium
    )


def test_prescription_17_frequency_phrases_are_recognized():
    from app import app as flask_app

    response = flask_app.test_client().post(
        "/api/analyze",
        json={
            "mode": "enhanced",
            "text": "Salbutamol nebule\nS 1 nebule every 8 hrs for 3-5 days",
        },
    )

    assert response.status_code == 200
    frequencies = {
        match["term"]
        for match in response.get_json()["matches"]
        if match["category"] == "Frequency" and not match.get("dropped")
    }
    durations = {
        match["term"]
        for match in response.get_json()["matches"]
        if match["category"] == "Duration" and not match.get("dropped")
    }
    assert "EVERY 8 HRS" in frequencies
    assert "3-5 DAYS" in durations


def test_duration_stays_with_the_medication_before_the_next_entry():
    from app import app as flask_app

    text = (
        "Ipratropium/Salbutamol nebule #15\n"
        "sig: nebulize every 8 hours\n"
        "Prednisone 50mg tab #5\n"
        "sig: 1 tab once daily\n"
        "for 5 days only\n"
        "Paracetamol 500mg tab #10\n"
        "sig: 1 tab once every 4 hours as needed\n"
        "for fever > 37.8°C"
    )
    response = flask_app.test_client().post(
        "/api/analyze",
        json={"mode": "enhanced", "text": text},
    )

    assert response.status_code == 200
    matches = [
        match for match in response.get_json()["matches"]
        if not match.get("dropped")
    ]
    duration = next(
        match for match in matches if match["matched_dictionary_term"] == "FOR 5 DAYS"
    )
    prednisone = next(
        match for match in matches
        if match["matched_dictionary_term"] == "PREDNISONE"
    )
    paracetamol = next(
        match for match in matches
        if match["matched_dictionary_term"] == "PARACETAMOL"
    )

    assert duration["instruction"] == prednisone["instruction"]
    assert duration["instruction"] != paracetamol["instruction"]


def test_priority_score_uses_all_four_requested_components():
    engine = build_engine()

    hits = engine.search("MONTELUKAST 10MG/TAB #14 SIG 1 TAB AT BEDTIME")
    scores = {hit["term"]: hit["priority_score"] for hit in hits}

    expected_scores = {
        "MONTELUKAST": 1.0,
        "AT BEDTIME": (10 / 11 + 3) / 4.0,
        "10MG/TAB": (8 / 11 + 3) / 4.0,
        "1 TAB": (5 / 11 + 3) / 4.0,
        "SIG": (3 / 11 + 3) / 4.0,
        "#14": (3 / 11 + 3) / 4.0,
    }
    assert set(scores) == set(expected_scores)
    assert all(abs(scores[term] - expected) < 1e-12
               for term, expected in expected_scores.items())


def test_eye_and_ear_laterality_abbreviations_are_recognized():
    engine = build_engine(common_words=COMMON_ENGLISH_WORDS)

    eye = {hit["term"]: hit for hit in engine.search("Instill 1 gtt OD, OS, OU every 6 hours")}
    assert eye["OS"]["category"] == "Abbreviation"
    assert eye["OS"]["meaning"] == "Left eye (oculus sinister)"
    assert eye["OU"]["meaning"] == "Both eyes (oculus uterque)"
    assert eye["OD"]["meaning"] == "Right eye (oculus dexter)"

    ear = {hit["term"]: hit for hit in engine.search("Ear drops: 3 drops AS, AD, AU for otitis")}
    assert ear["AS"]["meaning"] == "Left ear (auris sinistra)"
    assert ear["AD"]["meaning"] == "Right ear (auris dextra)"
    assert ear["AU"]["meaning"] == "Both ears (auris utraque)"


def test_as_in_ordinary_english_is_not_tagged_as_left_ear():
    engine = build_engine(common_words=COMMON_ENGLISH_WORDS)

    hits = engine.search("Paracetamol 500 mg as needed for fever, as directed")
    assert not any(hit["term"] == "AS" for hit in hits)


def test_output_is_sectioned_per_prescription_instruction():
    from app import app as flask_app

    text = (
        "Dibencozide       3 mg\n(Heraclene forte cap)\n#30\nS 1 cap once daily\n\n"
        "Ibuprofen 200 mg/5ml\nS 5ml q 6 hrs"
    )
    response = flask_app.test_client().post("/api/analyze", json={"mode": "enhanced", "text": text})
    matches = [m for m in response.get_json()["matches"] if not m.get("dropped")]
    section = {m["matched_dictionary_term"]: m["instruction"] for m in matches}

    # generic + brand alias in parentheses stay in the same instruction
    assert section["DIBENCOZIDE"] == section["HERACLENE FORTE"] == section["ONCE DAILY"] == 1
    assert section["IBUPROFEN"] == section["6 HRS"] == 2


def test_parenthetical_brands_after_quantity_stay_in_the_same_entry():
    from app import app as flask_app

    text = (
        "Febuxostat 80mg #100\n"
        "(Urinorm)\n"
        "1 tab 1x a day\n\n"
        "Liverprime #100\n"
        "1 tab 3x a day\n\n"
        "Metformin 500mg #200\n"
        "(Glumet)\n"
        "1 tab 2x a day\n\n"
        "Dapagliflozin 10mg #100\n"
        "1 tab after lunch"
    )
    response = flask_app.test_client().post(
        "/api/analyze",
        json={"mode": "enhanced", "text": text},
    )

    assert response.status_code == 200
    matches = [match for match in response.get_json()["matches"] if not match.get("dropped")]
    drug_entries = {
        match["matched_dictionary_term"]: match["instruction"]
        for match in matches
        if match["category"] == "Drug"
    }

    assert drug_entries == {
        "FEBUXOSTAT": 1,
        "URINORM": 1,
        "LIVERPRIME": 2,
        "METFORMIN": 3,
        "GLUMET": 3,
        "DAPAGLIFLOZIN": 4,
    }


def test_sinupret_forte_after_completed_prescription_starts_second_entry():
    from app import app as flask_app

    text = (
        "Polymyxin #1\n"
        "Instill 3 drops to the left ear 3x a day for 7 days\n\n"
        "(Sinupret Forte) #2\n"
        "1 tablet 3x a day for 7 days\n\n"
        "Celecoxib 200mg\n"
        "(Aubrex)\n"
        "Take 1 capsule 2x a day for pain"
    )
    response = flask_app.test_client().post(
        "/api/analyze",
        json={"mode": "enhanced", "text": text},
    )
    assert response.status_code == 200

    matches = [match for match in response.get_json()["matches"] if not match.get("dropped")]
    sections = {
        match["matched_dictionary_term"]: match["instruction"]
        for match in matches
        if match["category"] == "Drug"
    }

    assert sections["POLYMYXIN"] == 1
    assert sections["SINUPRET FORTE"] == 2
    assert sections["CELECOXIB"] == 3
    assert sections["AUBREX"] == 3


def test_humulin_and_insuget_stay_in_the_same_instruction():
    from app import app as flask_app

    text = (
        "Humulin 70/30\n"
        "or Insuget 70/30                            ____ #2\n"
        "10u - 6u\n"
        "before breakfast & supper\n\n"
        "Atorvastatin 40mg\n"
        "Lipitor                                            ____ #60\n"
        "0 - 0 - 1\n\n"
        "carvedilol 6.25 mg                         ____ #120\n"
        "1 - 0 - 1\n\n"
        "Clopidogrel 75mg\n"
        "Platexan                                         ____ #60\n"
        "1 - 0 - 0"
    )
    response = flask_app.test_client().post(
        "/api/analyze",
        json={"mode": "enhanced", "text": text},
    )
    assert response.status_code == 200

    matches = [m for m in response.get_json()["matches"] if not m.get("dropped")]
    section = {m["matched_dictionary_term"]: m["instruction"] for m in matches}
    label = {m["matched_dictionary_term"]: m["category_label"] for m in matches}

    assert section["HUMULIN 70/30"] == 1
    assert section["INSUGET 70/30"] == 1
    assert next(
        match["instruction_type"]
        for match in matches
        if match["matched_dictionary_term"] == "INSUGET 70/30"
    ) == "medication"
    assert section["ATORVASTATIN"] == section["LIPITOR"] == 2
    assert section["CARVEDILOL"] == 3
    assert section["CLOPIDOGREL"] == section["PLATEXAN"] == 4
    assert label["HUMULIN 70/30"] == "Drug 1 (Generic Name)"
    assert label["INSUGET 70/30"] == "Drug 1 (Brand Name)"
    assert label["ATORVASTATIN"] == "Drug 2 (Generic Name)"
    assert label["LIPITOR"] == "Drug 2 (Brand Name)"

    ordered_terms = [
        match["matched_dictionary_term"]
        for match in matches
        if match["instruction"] == 1
    ]
    assert ordered_terms[:2] == ["HUMULIN 70/30", "INSUGET 70/30"]


def test_repeated_sig_is_kept_for_each_medication_entry():
    from app import app as flask_app

    text = (
        "Oflox\n"
        "#19\n"
        "SIG\n"
        "2x a day\n\n"
        "Dolcet\n"
        "#9\n"
        "SIG\n"
        "3x a day"
    )
    response = flask_app.test_client().post(
        "/api/analyze",
        json={"mode": "enhanced", "text": text},
    )
    assert response.status_code == 200

    sig_matches = [
        match for match in response.get_json()["matches"]
        if match["matched_dictionary_term"] == "SIG"
    ]
    assert len(sig_matches) == 2
    assert {match["instruction"] for match in sig_matches} == {1, 2}


def test_nac_starts_the_second_medication_entry():
    from app import app as flask_app

    text = (
        "Nasoflo nasal spray #1\n"
        "Sig: 2 sprays per nostril once daily\n\n"
        "NAC 600mg tab #10\n"
        "Sig: 1 tab in 1 glass water 2x/day (1-0-1)"
    )
    response = flask_app.test_client().post(
        "/api/analyze",
        json={"mode": "enhanced", "text": text},
    )
    assert response.status_code == 200

    matches = [match for match in response.get_json()["matches"] if not match.get("dropped")]
    nac = next(match for match in matches if match["matched_dictionary_term"] == "NAC")
    sigs = [match for match in matches if match["matched_dictionary_term"] == "SIG"]

    assert nac["category"] == "Drug"
    assert nac["instruction"] == 2
    assert nac["category_label"] == "Drug 2 (Generic Name)"
    assert {match["instruction"] for match in sigs} == {1, 2}


def test_separate_brands_after_sig_start_new_medication_entries():
    from app import app as flask_app

    text = (
        "Nasoflo nasal spray #1\n"
        "Sig: 2 sprays per nostril once daily\n\n"
        "NAC 600mg tab #10\n"
        "Sig: 1 tab in 1 glass water 2x/day (1-0-1)\n\n"
        "Kamillosan throat spray #1\n"
        "Sig: 1-2 sprays per orem 3x/day before meals AND as needed\n"
        "for throat itch\n\n"
        "Bactidol gargle #3\n"
        "Sig: gargle 10ml 3x/day after meals"
    )
    response = flask_app.test_client().post(
        "/api/analyze",
        json={"mode": "enhanced", "text": text},
    )
    assert response.status_code == 200

    matches = [match for match in response.get_json()["matches"] if not match.get("dropped")]
    entries = {
        match["matched_dictionary_term"]: match["instruction"]
        for match in matches
        if match["matched_dictionary_term"] in {"NASOFLO", "NAC", "KAMILLOSAN", "BACTIDOL"}
    }

    assert entries == {
        "NASOFLO": 1,
        "NAC": 2,
        "KAMILLOSAN": 3,
        "BACTIDOL": 4,
    }