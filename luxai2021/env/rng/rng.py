"""Lux AI's seeded random number generator.

The original implementation spawned Node.js and eagerly transferred one
million comma-separated floats for every generated map.  This is a direct
Python port of the bundled seedrandom/ARC4 implementation, so values can be
produced on demand without a subprocess or a large temporary list.
"""

_WIDTH = 256
_MASK = _WIDTH - 1
_CHUNKS = 6
_DIGITS = _WIDTH ** _CHUNKS
_START_DENOM = 2 ** 52
_SIGNIFICANCE = 2 * _START_DENOM


class _ARC4:
    def __init__(self, key):
        self.i = 0
        self.j = 0
        self.S = list(range(_WIDTH))

        j = 0
        for i in range(_WIDTH):
            value = self.S[i]
            j = _MASK & (j + key[i % len(key)] + value)
            self.S[i], self.S[j] = self.S[j], value

        # seedrandom discards the first RC4 block.
        self.generate(_WIDTH)

    def generate(self, count):
        result = 0
        i = self.i
        j = self.j
        state = self.S

        for _ in range(count):
            i = _MASK & (i + 1)
            value = state[i]
            j = _MASK & (j + value)
            state[i], state[j] = state[j], value
            result = result * _WIDTH + state[_MASK & (state[i] + state[j])]

        self.i = i
        self.j = j
        return result


class SeededRandom:
    """Generate the same doubles as seedrandom with Lux's prefixed seed."""

    def __init__(self, seed):
        key = []
        smear = 0
        for index, char in enumerate(f"gen_{seed}"):
            key_index = _MASK & index
            previous = key[key_index] if key_index < len(key) else 0
            smear ^= 19 * previous
            value = _MASK & (smear + ord(char))
            if key_index == len(key):
                key.append(value)
            else:
                key[key_index] = value

        self._arc4 = _ARC4(key)

    def random(self):
        numerator = self._arc4.generate(_CHUNKS)
        denominator = _DIGITS
        extra = 0

        while numerator < _START_DENOM:
            numerator = (numerator + extra) * _WIDTH
            denominator *= _WIDTH
            extra = self._arc4.generate(1)

        while numerator >= _SIGNIFICANCE:
            numerator /= 2
            denominator /= 2
            extra >>= 1

        return (numerator + extra) / denominator


def get_n_values(seed, N=100):
    """
    Generate the same random numbers as the Lux AI CLI for ``seed``.

    Kept for compatibility with callers that need a materialized list.
    """
    rng = SeededRandom(seed)
    return [rng.random() for _ in range(N)]
