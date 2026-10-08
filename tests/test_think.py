import engine


def test_parse_think():
    assert engine.parse_think(
        "<think>Etape 1.\nEtape 2.</think>\nReponse : Paris"
    ) == ("Etape 1.\nEtape 2.", "Paris")


def test_parse_think_rejects_incomplete_output():
    assert engine.parse_think("<think>raisonnement</think>") is None
    assert engine.parse_think("Reponse : sans raisonnement") is None
    assert engine.parse_think("<think></think>\nReponse : vide") is None
