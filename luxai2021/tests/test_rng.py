from luxai2021.env.rng.rng import SeededRandom, get_n_values


def test_seeded_random_matches_seedrandom_values():
    rng = SeededRandom(123456789)

    assert [rng.random() for _ in range(5)] == [
        0.3631819401404689,
        0.7995896811459317,
        0.694804040387595,
        0.5666659071511181,
        0.4576113950066108,
    ]


def test_get_n_values_keeps_materialized_list_api():
    rng = SeededRandom(0)

    assert get_n_values(0, 10) == [rng.random() for _ in range(10)]
