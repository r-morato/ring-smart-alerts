from ring_smart_alerts.detector import Detection, summarize


def d(label, conf=0.9):
    return Detection(label, conf)


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
