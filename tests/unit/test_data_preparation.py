from data_prep.src import data_preparation as dp


def test_clean_text():
    text = "Hello @user! Visit http://example.com #tag &amp; nice"
    cleaned = dp.clean_text(text)
    assert "@" not in cleaned
    assert "http" not in cleaned
    assert "#" not in cleaned
    assert "&amp;" not in cleaned


def test_map_label_str():
    assert dp.map_label("negative") == 0
    assert dp.map_label("neutral") == 1
    assert dp.map_label("positive") == 2
    assert dp.map_label("unknown") == 1


def test_map_label_int():
    assert dp.map_label(5) == 5


def test_get_tokenizer_is_callable():
    # Non testiamo il download, solo che la funzione esiste
    tokenizer = dp.get_tokenizer
    assert callable(tokenizer)
