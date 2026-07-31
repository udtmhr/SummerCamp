# ruff: noqa: ANN001, ANN201, ARG005, PLR2004, S101, SLF001

import torch

import main
from luxai2021.imitation.agent import FirstPlaceAgent


def test_first_place_agent_selects_team_and_converts_xy_axes():
    output = {"worker": torch.zeros(1, 2, 19, 32, 32)}
    output["worker"][0, 1, 3, 5, 7] = 11

    selected = FirstPlaceAgent._select_team_output(output, team=1)

    assert selected["worker"].shape == (1, 19, 32, 32)
    assert selected["worker"][0, 3, 7, 5] == 11


def test_one_on_one_cli_selects_first_place_agent(monkeypatch):
    expected = object()
    monkeypatch.setattr(main, "FirstPlaceAgent", lambda *args, **kwargs: expected)

    actual = main.create_agent("teacher.pt", "first-place", "cpu", "auto")

    assert actual is expected
