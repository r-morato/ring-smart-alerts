from ring_smart_alerts.detector import Detection, summarize


def d(label, conf=0.9, detail=None):
    return Detection(label, conf, detail)


def test_empty():
    assert summarize([]) == "nothing recognised"


def test_single_consonant():
    assert summarize([d("person")]) == "a person"


def test_single_vowel():
    assert summarize([d("umbrella")]) == "an umbrella"


def test_two():
    assert summarize([d("person"), d("dog")]) == "a person and a dog"


def test_three():
    assert summarize([d("person"), d("dog"), d("car")]) == "a person, a dog and a car"


def test_detail_replaces_label():
    assert summarize([d("person", detail="child")]) == "a child"


def test_detail_hedged_gender_gets_an():
    assert (
        summarize([d("person", detail="adult (looks like a man)")])
        == "an adult (looks like a man)"
    )


def test_two_refined_people():
    got = summarize([d("person", detail="adult (looks like a woman)"), d("person", detail="child")])
    assert got == "an adult (looks like a woman) and a child"
