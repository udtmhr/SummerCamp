# Lux AI Challenge 2021 rules for reward evolution

Source: https://www.lux-ai.org/specs-2021#Background

- Lux Season 1 is a fully observed, two-player game. A normal game lasts 360 turns: nine cycles of 30 day turns followed by 10 night turns.
- The primary win condition is the number of CityTiles at the end of the game. Units are the tiebreaker; equality in both is a draw. A game can end early when one team has neither Units nor CityTiles.
- Maps are symmetric square grids of size 12, 16, 24, or 32. Each team starts with one CityTile and one Worker.
- Wood, Coal, and Uranium are worth 1, 10, and 40 fuel per unit. Coal collection requires 50 research and Uranium requires 200 research.
- Workers carry 100 resources and Carts carry 2000. A Worker with 100 total resources can consume its cargo to build a zero-fuel CityTile.
- During every night turn a City consumes light upkeep. A CityTile contributes `23 - 5 * adjacent_friendly_city_tiles` upkeep. If a connected City cannot pay one night turn, the whole City is destroyed.
- A Worker or Cart outside a friendly CityTile consumes 4 or 10 fuel per night turn and is destroyed if it cannot pay. Units on friendly CityTiles do not burn cargo for light.
- Actions are validated from the state at the start of the turn. Resolution order is CityTile actions, Unit actions, roads, resource collection, deposits to Cities, night consumption, Wood regrowth, and cooldown updates.
- Building CityTiles directly increases the primary score, but also creates future fuel liability. Connected expansion can reduce upkeep while isolated expansion can create fragile Cities.
- CityTiles, Units, Research, and collected resources are proxy signals. They must not override terminal win/loss or hide late-night City extinction, fuel-delivery failure, or poor final CityTile count.

Implications for proposals:

- Preserve terminal outcome as the authoritative objective.
- Treat the official first-place Teacher as the strongest evaluation anchor; an improvement against a distilled base must not compensate for a regression against the Teacher.
- Prefer bounded, phase-aware potential shaping that explains how it improves final CityTiles, last-night survival, or fuel delivery.
- Interpret raw CityTile losses together with peak/final CityTiles and fuel margins; a larger expanding City can lose more tiles in absolute terms and still win.
