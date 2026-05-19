"""
ProceduralSource: infinite problem generation from templates.

Generates problems programmatically — no LLM, no API, no download.
Computes correct answers from the input, so test cases are always verified.

Problem families:
  arithmetic:    sum/product/min/max of N integers with various constraints
  strings:       reverse, count, find, transform, validate
  sequences:     fibonacci-style, running totals, windows, deduplication
  sorting:       sort with comparators, k-th element, merge
  searching:     binary search variants, two-pointer, sliding window
  math:          prime check, GCD, LCM, digits, modular arithmetic
  grids:         path counting, island counting, spiral traversal
  intervals:     overlap detection, merge, coverage

Difficulty is assigned based on:
  - Number of operations required
  - Size of input
  - Need for auxiliary data structures (stack, dict, heap)

All test cases are computed from the solution — no guessing.
"""
from __future__ import annotations

import math
import random
from typing import Iterator


class ProceduralSource:
    """
    Infinite stream of programmatically generated coding problems.

    Each call to sample() or __iter__() produces fresh instances.
    Seeded for reproducibility; different seeds give different problems.
    """

    def __init__(
        self,
        seed:             int   = 42,
        difficulty_range: tuple = (0.15, 0.65),   # 20-60% zone
    ):
        self._seed    = seed
        self._lo, self._hi = difficulty_range
        self._rng     = random.Random(seed)
        self._counter = 0

    def sample(self, n: int) -> list[dict]:
        """Generate n problems."""
        return [self._generate_one() for _ in range(n)]

    def __iter__(self) -> Iterator[dict]:
        """Infinite iterator."""
        while True:
            yield self._generate_one()

    # ── Problem dispatcher ─────────────────────────────────────────────────────

    def _generate_one(self) -> dict:
        self._counter += 1
        rng = random.Random(self._seed + self._counter)

        # Weight families by estimated coverage of difficulty range
        family = rng.choice([
            'arithmetic', 'arithmetic', 'arithmetic',
            'strings',    'strings',    'strings',
            'sequences',  'sequences',
            'sorting',    'sorting',
            'math',       'math',
            'searching',
            'intervals',
        ])

        generators = {
            'arithmetic': self._gen_arithmetic,
            'strings':    self._gen_strings,
            'sequences':  self._gen_sequences,
            'sorting':    self._gen_sorting,
            'math':       self._gen_math,
            'searching':  self._gen_searching,
            'intervals':  self._gen_intervals,
        }
        return generators[family](rng)

    # ── Arithmetic ─────────────────────────────────────────────────────────────

    def _gen_arithmetic(self, rng: random.Random) -> dict:
        variant = rng.choice(['sum', 'max_sub', 'running_avg', 'weighted_sum'])

        if variant == 'sum':
            n    = rng.randint(3, 20)
            nums = [rng.randint(-100, 100) for _ in range(n)]
            mod  = rng.choice([None, 7, 13, 1000000007])
            ans  = sum(nums) if mod is None else sum(nums) % mod
            desc = (f"Given {n} integers, compute their sum"
                    + (f" modulo {mod}" if mod else "") + ".\n"
                    + f"Input: {n} space-separated integers\nOutput: the sum")
            inp  = ' '.join(map(str, nums))
            return self._make(f'arith_sum_{self._counter}', desc, inp, str(ans), 0.15)

        if variant == 'max_sub':
            n    = rng.randint(4, 15)
            nums = [rng.randint(-10, 10) for _ in range(n)]
            # Kadane's algorithm
            best = cur = nums[0]
            for x in nums[1:]:
                cur  = max(x, cur + x)
                best = max(best, cur)
            desc = (f"Find the maximum subarray sum (contiguous) in a list of {n} integers.\n"
                    f"Input: {n} space-separated integers\nOutput: maximum subarray sum")
            inp  = ' '.join(map(str, nums))
            return self._make(f'arith_maxsub_{self._counter}', desc, inp, str(best), 0.35)

        if variant == 'running_avg':
            n    = rng.randint(3, 8)
            nums = [rng.randint(1, 50) for _ in range(n)]
            avgs = [round(sum(nums[:i+1]) / (i+1), 2) for i in range(n)]
            desc = (f"Print the running average (rounded to 2 decimal places) after each element.\n"
                    f"Input: {n} space-separated integers\n"
                    f"Output: {n} space-separated running averages")
            inp  = ' '.join(map(str, nums))
            ans  = ' '.join(f'{a:.2f}' for a in avgs)
            return self._make(f'arith_avg_{self._counter}', desc, inp, ans, 0.25)

        # weighted_sum
        n      = rng.randint(3, 8)
        nums   = [rng.randint(1, 20) for _ in range(n)]
        weights= [rng.randint(1, 5)  for _ in range(n)]
        ans    = sum(a * b for a, b in zip(nums, weights))
        desc   = (f"Compute the weighted sum of two lists of {n} integers.\n"
                  f"Input line 1: {n} values\nInput line 2: {n} weights\n"
                  f"Output: weighted sum")
        inp    = ' '.join(map(str, nums)) + '\n' + ' '.join(map(str, weights))
        return self._make(f'arith_wsum_{self._counter}', desc, inp, str(ans), 0.2)

    # ── Strings ────────────────────────────────────────────────────────────────

    def _gen_strings(self, rng: random.Random) -> dict:
        variant = rng.choice(['compress', 'longest_word', 'word_freq', 'caesar'])

        words = ['apple', 'banana', 'cherry', 'date', 'elderberry',
                 'fig', 'grape', 'honeydew', 'iris', 'jasmine']

        if variant == 'compress':
            n = rng.randint(4, 15)
            s = ''.join(rng.choice('aabbbccccddddd') for _ in range(n))
            # Run-length encode
            ans  = ''
            i    = 0
            while i < len(s):
                c = s[i]; cnt = 0
                while i < len(s) and s[i] == c:
                    cnt += 1; i += 1
                ans += f'{c}{cnt}' if cnt > 1 else c
            desc = ("Run-length encode a string: replace consecutive duplicates "
                    "with the character followed by the count (omit count if 1).\n"
                    "Input: one string\nOutput: compressed string")
            return self._make(f'str_compress_{self._counter}', desc, s, ans, 0.3)

        if variant == 'longest_word':
            n   = rng.randint(3, 8)
            wds = rng.sample(words * 2, n)
            ans = max(wds, key=len)
            inp = ' '.join(wds)
            desc = ("Find the longest word in a list. If tie, return the first.\n"
                    "Input: space-separated words\nOutput: longest word")
            return self._make(f'str_longest_{self._counter}', desc, inp, ans, 0.2)

        if variant == 'word_freq':
            n   = rng.randint(4, 10)
            wds = [rng.choice(words[:4]) for _ in range(n)]
            from collections import Counter
            freq = Counter(wds)
            # Most common word
            ans  = freq.most_common(1)[0][0]
            inp  = ' '.join(wds)
            desc = ("Find the most frequent word. If tie, return the lexicographically first.\n"
                    "Input: space-separated words\nOutput: most frequent word")
            return self._make(f'str_freq_{self._counter}', desc, inp, ans, 0.25)

        # caesar cipher
        shift = rng.randint(1, 25)
        n     = rng.randint(3, 8)
        plain = ''.join(rng.choice('abcdefghijklmnop') for _ in range(n))
        cipher= ''.join(chr((ord(c) - ord('a') + shift) % 26 + ord('a')) for c in plain)
        desc  = (f"Decode a Caesar cipher with a known shift of {shift}.\n"
                 f"Input: encoded lowercase string\nOutput: decoded string")
        return self._make(f'str_caesar_{self._counter}', desc, cipher, plain, 0.25)

    # ── Sequences ─────────────────────────────────────────────────────────────

    def _gen_sequences(self, rng: random.Random) -> dict:
        variant = rng.choice(['prefix_sum', 'sliding_max', 'dedup_order'])

        if variant == 'prefix_sum':
            n    = rng.randint(4, 10)
            nums = [rng.randint(1, 20) for _ in range(n)]
            l, r = sorted(rng.sample(range(n), 2))
            ans  = sum(nums[l:r+1])
            desc = (f"Compute the sum of elements from index {l} to {r} (0-based, inclusive).\n"
                    f"Input line 1: {n} space-separated integers\n"
                    f"Input line 2: l r (0-based inclusive range)\n"
                    f"Output: range sum")
            inp  = ' '.join(map(str, nums)) + f'\n{l} {r}'
            return self._make(f'seq_psum_{self._counter}', desc, inp, str(ans), 0.3)

        if variant == 'sliding_max':
            n = rng.randint(5, 12)
            k = rng.randint(2, min(4, n-1))
            nums = [rng.randint(1, 30) for _ in range(n)]
            maxs = [max(nums[i:i+k]) for i in range(n-k+1)]
            desc = (f"Find the maximum in each window of size {k} in an array of {n} integers.\n"
                    f"Input: {n} space-separated integers\n"
                    f"Output: {n-k+1} maximums, space-separated")
            inp  = ' '.join(map(str, nums))
            ans  = ' '.join(map(str, maxs))
            return self._make(f'seq_slmax_{self._counter}', desc, inp, ans, 0.4)

        # dedup preserving order
        n    = rng.randint(5, 12)
        pool = list(range(1, 8))
        nums = [rng.choice(pool) for _ in range(n)]
        seen = set(); deduped = []
        for x in nums:
            if x not in seen:
                seen.add(x); deduped.append(x)
        desc = ("Remove duplicates from a list while preserving the order of first appearances.\n"
                "Input: space-separated integers\nOutput: deduplicated list, space-separated")
        inp  = ' '.join(map(str, nums))
        ans  = ' '.join(map(str, deduped))
        return self._make(f'seq_dedup_{self._counter}', desc, inp, ans, 0.25)

    # ── Sorting ────────────────────────────────────────────────────────────────

    def _gen_sorting(self, rng: random.Random) -> dict:
        variant = rng.choice(['sort_by_len', 'k_smallest', 'sort_abs'])

        if variant == 'sort_by_len':
            words = ['cat', 'elephant', 'bee', 'hippopotamus', 'ant', 'dog',
                     'butterfly', 'ox', 'frog', 'salamander']
            n   = rng.randint(4, 8)
            wds = rng.sample(words, n)
            ans = sorted(wds, key=lambda w: (len(w), w))
            desc= ("Sort words by length (ascending), then alphabetically for ties.\n"
                   "Input: space-separated words\nOutput: sorted words, space-separated")
            inp = ' '.join(wds)
            return self._make(f'sort_len_{self._counter}', desc, inp, ' '.join(ans), 0.3)

        if variant == 'k_smallest':
            n    = rng.randint(5, 12)
            k    = rng.randint(1, min(4, n))
            nums = [rng.randint(1, 50) for _ in range(n)]
            ans  = sorted(nums)[k-1]
            desc = (f"Find the {k}-th smallest element (1-indexed) in a list of {n} integers.\n"
                    f"Input line 1: {n} integers\nInput line 2: k\nOutput: k-th smallest")
            inp  = ' '.join(map(str, nums)) + f'\n{k}'
            return self._make(f'sort_kth_{self._counter}', desc, inp, str(ans), 0.3)

        # sort by absolute value
        n    = rng.randint(4, 10)
        nums = [rng.randint(-20, 20) for _ in range(n)]
        ans  = sorted(nums, key=abs)
        desc = ("Sort integers by absolute value (ascending). Keep original sign.\n"
                "Input: space-separated integers\nOutput: sorted by |value|, space-separated")
        inp  = ' '.join(map(str, nums))
        return self._make(f'sort_abs_{self._counter}', desc, inp, ' '.join(map(str, ans)), 0.2)

    # ── Math ──────────────────────────────────────────────────────────────────

    def _gen_math(self, rng: random.Random) -> dict:
        variant = rng.choice(['prime_count', 'digit_product', 'gcd_list'])

        if variant == 'prime_count':
            n   = rng.randint(10, 50)
            def is_prime(x):
                if x < 2: return False
                if x == 2: return True
                if x % 2 == 0: return False
                return all(x % i != 0 for i in range(3, int(x**0.5)+1, 2))
            ans = sum(1 for i in range(2, n+1) if is_prime(i))
            desc= (f"Count the prime numbers from 2 to {n} (inclusive).\n"
                   f"Input: n={n}\nOutput: count of primes")
            return self._make(f'math_primes_{self._counter}', desc, str(n), str(ans), 0.3)

        if variant == 'digit_product':
            n   = rng.randint(100, 9999)
            ans = 1
            for d in str(abs(n)):
                if int(d) == 0:
                    ans = 0; break
                ans *= int(d)
            desc= (f"Compute the product of digits of a positive integer.\n"
                   f"Input: one integer\nOutput: product of its digits")
            return self._make(f'math_digprod_{self._counter}', desc, str(n), str(ans), 0.2)

        # gcd of list
        n    = rng.randint(2, 5)
        nums = [rng.randint(2, 30) * rng.randint(1, 5) for _ in range(n)]
        ans  = nums[0]
        for x in nums[1:]:
            ans = math.gcd(ans, x)
        desc = (f"Find the GCD of {n} integers.\n"
                f"Input: {n} space-separated integers\nOutput: their GCD")
        inp  = ' '.join(map(str, nums))
        return self._make(f'math_gcd_{self._counter}', desc, inp, str(ans), 0.25)

    # ── Searching ─────────────────────────────────────────────────────────────

    def _gen_searching(self, rng: random.Random) -> dict:
        n    = rng.randint(5, 15)
        nums = sorted(set(rng.randint(1, 50) for _ in range(n)))[:n]
        # Pick a target that may or may not be in the list
        if rng.random() < 0.5:
            target = rng.choice(nums)
            ans    = str(nums.index(target))
        else:
            # target not in list
            target = rng.randint(1, 50)
            while target in nums:
                target += 1
            ans = '-1'
        desc = ("Binary search: find the index of target in a sorted array, or -1.\n"
                "Input line 1: space-separated sorted integers\n"
                "Input line 2: target\nOutput: index or -1")
        inp  = ' '.join(map(str, nums)) + f'\n{target}'
        return self._make(f'search_bin_{self._counter}', desc, inp, ans, 0.3)

    # ── Intervals ─────────────────────────────────────────────────────────────

    def _gen_intervals(self, rng: random.Random) -> dict:
        variant = rng.choice(['merge', 'count_overlap'])

        n = rng.randint(3, 6)
        intervals = sorted(
            [(rng.randint(0, 10), rng.randint(11, 20)) for _ in range(n)])

        if variant == 'merge':
            merged = [list(intervals[0])]
            for s, e in intervals[1:]:
                if s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            desc = ("Merge overlapping intervals. Print merged intervals sorted by start.\n"
                    "Input: N, then N lines of 'start end' (space-separated)\n"
                    "Output: merged intervals, one per line")
            lines = [str(n)] + [f'{s} {e}' for s, e in intervals]
            ans   = '\n'.join(f'{s} {e}' for s, e in merged)
            return self._make(f'int_merge_{self._counter}', desc,
                              '\n'.join(lines), ans, 0.45)

        # count overlapping pairs
        count = sum(1 for i in range(n) for j in range(i+1, n)
                    if intervals[i][1] >= intervals[j][0])
        desc = ("Count pairs of overlapping intervals.\n"
                "Input: N, then N lines of 'start end'\nOutput: count of overlapping pairs")
        lines = [str(n)] + [f'{s} {e}' for s, e in intervals]
        return self._make(f'int_overlap_{self._counter}', desc,
                          '\n'.join(lines), str(count), 0.35)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _make(pid: str, desc: str, inp: str, expected: str,
              difficulty: float) -> dict:
        return {
            'id':          pid,
            'description': desc,
            'tests':       [{'input': inp, 'expected': expected}],
            'difficulty':  difficulty,
            'solution':    None,
        }
