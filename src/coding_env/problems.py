"""
Coding problem loader.

Built-in: 20 hand-crafted problems covering easy→medium difficulty.
These cover the 20-60% pass-rate zone optimal for RL training (RLEF finding).

Optional: HumanEval (164 problems) via pip install datasets.

Problem format:
    {
        'id':          str,
        'description': str,       # problem statement shown to brain
        'tests':       list[dict], # [{'input': str, 'expected': str}]
        'difficulty':  float,      # 0.0 (easy) to 1.0 (hard)
        'solution':    str | None  # reference solution for CE supervision
    }
"""
from __future__ import annotations

import random
from typing import Iterator


# ── Built-in problems ─────────────────────────────────────────────────────────

_BUILTIN_PROBLEMS = [
    {
        'id': 'fizzbuzz',
        'description': (
            'Write a function fizzbuzz(n) that returns a list of strings for 1..n. '
            'Replace multiples of 3 with "Fizz", multiples of 5 with "Buzz", '
            'multiples of both with "FizzBuzz", otherwise the number as a string.\n'
            'Print the result as space-separated values.\n'
            'Input: one integer n\nOutput: space-separated fizzbuzz values'
        ),
        'tests': [
            {'input': '15', 'expected': '1 2 Fizz 4 Buzz Fizz 7 8 Fizz Buzz 11 Fizz 13 14 FizzBuzz'},
            {'input': '5',  'expected': '1 2 Fizz 4 Buzz'},
            {'input': '3',  'expected': '1 2 Fizz'},
        ],
        'difficulty': 0.1,
        'solution': 'n=int(input())\nprint(*["FizzBuzz"if i%15==0 else"Fizz"if i%3==0 else"Buzz"if i%5==0 else str(i) for i in range(1,n+1)])',
    },
    {
        'id': 'reverse_string',
        'description': 'Read a string and print it reversed.\nInput: one string\nOutput: reversed string',
        'tests': [
            {'input': 'hello', 'expected': 'olleh'},
            {'input': 'abcde', 'expected': 'edcba'},
            {'input': 'a',     'expected': 'a'},
        ],
        'difficulty': 0.05,
        'solution': 'print(input()[::-1])',
    },
    {
        'id': 'sum_digits',
        'description': 'Read an integer and print the sum of its digits.\nInput: one integer\nOutput: sum of digits',
        'tests': [
            {'input': '123',  'expected': '6'},
            {'input': '9999', 'expected': '36'},
            {'input': '10',   'expected': '1'},
        ],
        'difficulty': 0.1,
        'solution': 'print(sum(int(d) for d in input()))',
    },
    {
        'id': 'is_palindrome',
        'description': 'Read a string. Print "YES" if it is a palindrome, "NO" otherwise.\nInput: one string\nOutput: YES or NO',
        'tests': [
            {'input': 'racecar', 'expected': 'YES'},
            {'input': 'hello',   'expected': 'NO'},
            {'input': 'a',       'expected': 'YES'},
            {'input': 'abba',    'expected': 'YES'},
        ],
        'difficulty': 0.15,
        'solution': 's=input();print("YES"if s==s[::-1]else"NO")',
    },
    {
        'id': 'count_vowels',
        'description': 'Read a string and print the number of vowels (a,e,i,o,u, case-insensitive).\nInput: one string\nOutput: count',
        'tests': [
            {'input': 'Hello World', 'expected': '3'},
            {'input': 'AEIOU',       'expected': '5'},
            {'input': 'rhythm',      'expected': '0'},
        ],
        'difficulty': 0.15,
        'solution': 'print(sum(c.lower()in"aeiou"for c in input()))',
    },
    {
        'id': 'fibonacci',
        'description': 'Print the first n Fibonacci numbers (0-indexed: 0,1,1,2,3,5,...).\nInput: n\nOutput: space-separated fibonacci numbers',
        'tests': [
            {'input': '7',  'expected': '0 1 1 2 3 5 8'},
            {'input': '1',  'expected': '0'},
            {'input': '10', 'expected': '0 1 1 2 3 5 8 13 21 34'},
        ],
        'difficulty': 0.2,
        'solution': 'n=int(input());a,b=0,1;r=[]\nfor _ in range(n):r.append(a);a,b=b,a+b\nprint(*r)',
    },
    {
        'id': 'two_sum',
        'description': (
            'Given a list of integers and a target, find two indices whose values sum to target.\n'
            'Print the two indices (0-based), smaller first. Guaranteed unique solution.\n'
            'Input line 1: space-separated integers\nInput line 2: target\nOutput: two indices'
        ),
        'tests': [
            {'input': '2 7 11 15\n9',  'expected': '0 1'},
            {'input': '3 2 4\n6',       'expected': '1 2'},
            {'input': '1 5 3 8 2\n10',  'expected': '1 3'},
        ],
        'difficulty': 0.3,
        'solution': 'nums=list(map(int,input().split()));t=int(input());seen={}\nfor i,n in enumerate(nums):\n if t-n in seen:print(seen[t-n],i);break\n seen[n]=i',
    },
    {
        'id': 'anagram',
        'description': 'Read two strings. Print "YES" if they are anagrams, "NO" otherwise.\nInput: two lines\nOutput: YES or NO',
        'tests': [
            {'input': 'listen\nsilent', 'expected': 'YES'},
            {'input': 'hello\nworld',   'expected': 'NO'},
            {'input': 'rat\ntar',       'expected': 'YES'},
        ],
        'difficulty': 0.2,
        'solution': 'a=input();b=input();print("YES"if sorted(a)==sorted(b)else"NO")',
    },
    {
        'id': 'roman_to_int',
        'description': (
            'Convert a Roman numeral string to an integer.\n'
            'I=1,V=5,X=10,L=50,C=100,D=500,M=1000. Subtraction rule applies.\n'
            'Input: Roman numeral string\nOutput: integer'
        ),
        'tests': [
            {'input': 'III',    'expected': '3'},
            {'input': 'IV',     'expected': '4'},
            {'input': 'IX',     'expected': '9'},
            {'input': 'LVIII',  'expected': '58'},
            {'input': 'MCMXCIV','expected': '1994'},
        ],
        'difficulty': 0.35,
        'solution': (
            's=input();v={"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}\n'
            'r=0\nfor i in range(len(s)):r+=(-v[s[i]] if i+1<len(s) and v[s[i]]<v[s[i+1]] else v[s[i]])\nprint(r)'
        ),
    },
    {
        'id': 'longest_common_prefix',
        'description': 'Find the longest common prefix of space-separated strings.\nPrint empty string if none.\nInput: space-separated strings\nOutput: common prefix',
        'tests': [
            {'input': 'flower flow flight', 'expected': 'fl'},
            {'input': 'dog racecar car',    'expected': ''},
            {'input': 'abc abc abc',        'expected': 'abc'},
        ],
        'difficulty': 0.3,
        'solution': 'ws=input().split();p=ws[0]\nfor w in ws[1:]:\n while not w.startswith(p):p=p[:-1]\nprint(p)',
    },
    {
        'id': 'valid_brackets',
        'description': (
            'Check if a string of brackets is valid. Brackets: ()[]{}.\n'
            'Print YES if valid, NO otherwise.\n'
            'Input: one string\nOutput: YES or NO'
        ),
        'tests': [
            {'input': '()',      'expected': 'YES'},
            {'input': '()[]{}', 'expected': 'YES'},
            {'input': '(]',     'expected': 'NO'},
            {'input': '([)]',   'expected': 'NO'},
            {'input': '{[]}',   'expected': 'YES'},
        ],
        'difficulty': 0.35,
        'solution': (
            's=input();st=[]\nm={")":" (","}":" {","]":"["}\n'
            'for c in s:\n if c in "([{":st.append(c)\n elif not st or st[-1]!=m[c]:print("NO");exit()\n else:st.pop()\n'
            'print("YES"if not st else"NO")'
        ),
    },
    {
        'id': 'merge_sorted',
        'description': (
            'Merge two sorted integer arrays and print the result.\n'
            'Input line 1: space-separated integers (sorted)\n'
            'Input line 2: space-separated integers (sorted)\n'
            'Output: merged sorted integers, space-separated'
        ),
        'tests': [
            {'input': '1 3 5\n2 4 6',   'expected': '1 2 3 4 5 6'},
            {'input': '1 2 3\n4 5 6',   'expected': '1 2 3 4 5 6'},
            {'input': '1\n2',           'expected': '1 2'},
        ],
        'difficulty': 0.25,
        'solution': 'a=list(map(int,input().split()));b=list(map(int,input().split()))\nprint(*sorted(a+b))',
    },
    {
        'id': 'rotate_matrix',
        'description': (
            'Rotate an NxN matrix 90 degrees clockwise. Print each row space-separated.\n'
            'Input: N, then N rows of N space-separated integers\n'
            'Output: rotated matrix rows'
        ),
        'tests': [
            {'input': '2\n1 2\n3 4',           'expected': '3 1\n4 2'},
            {'input': '3\n1 2 3\n4 5 6\n7 8 9','expected': '7 4 1\n8 5 2\n9 6 3'},
        ],
        'difficulty': 0.4,
        'solution': (
            'n=int(input());m=[list(map(int,input().split())) for _ in range(n)]\n'
            'r=[[m[n-1-j][i] for j in range(n)] for i in range(n)]\n'
            'for row in r:print(*row)'
        ),
    },
    {
        'id': 'group_anagrams',
        'description': (
            'Group anagrams together from a list of words.\n'
            'Print each group sorted alphabetically, one per line, groups sorted by first element.\n'
            'Input: space-separated words\nOutput: groups, one per line'
        ),
        'tests': [
            {'input': 'eat tea tan ate nat bat',
             'expected': 'ate eat tea\nbat\nnat tan'},
            {'input': 'a', 'expected': 'a'},
        ],
        'difficulty': 0.4,
        'solution': (
            'from collections import defaultdict\n'
            'ws=input().split();d=defaultdict(list)\n'
            'for w in ws:d[tuple(sorted(w))].append(w)\n'
            'for g in sorted(sorted(v) for v in d.values()):print(*sorted(g))'
        ),
    },
    {
        'id': 'binary_search',
        'description': (
            'Binary search: find the index of target in a sorted array, or -1.\n'
            'Input line 1: space-separated sorted integers\n'
            'Input line 2: target\nOutput: index or -1'
        ),
        'tests': [
            {'input': '1 3 5 7 9\n5',  'expected': '2'},
            {'input': '1 3 5 7 9\n4',  'expected': '-1'},
            {'input': '1\n1',          'expected': '0'},
        ],
        'difficulty': 0.3,
        'solution': (
            'a=list(map(int,input().split()));t=int(input());l,r=0,len(a)-1\n'
            'while l<=r:\n m=(l+r)//2\n if a[m]==t:print(m);exit()\n elif a[m]<t:l=m+1\n else:r=m-1\n'
            'print(-1)'
        ),
    },
    {
        'id': 'lru_cache',
        'description': (
            'Implement an LRU cache with capacity k.\n'
            'Operations: "SET key val" and "GET key" (print -1 if not found).\n'
            'Input: capacity, then N operations\nOutput: GET results, one per line'
        ),
        'tests': [
            {'input': '2\n5\nSET 1 1\nSET 2 2\nGET 1\nSET 3 3\nGET 2',
             'expected': '1\n-1'},
        ],
        'difficulty': 0.55,
        'solution': (
            'from collections import OrderedDict\n'
            'k=int(input());n=int(input());c=OrderedDict()\n'
            'for _ in range(n):\n'
            ' op=input().split()\n'
            ' if op[0]=="GET":\n'
            '  key=int(op[1])\n'
            '  if key in c:c.move_to_end(key);print(c[key])\n'
            '  else:print(-1)\n'
            ' else:\n'
            '  key,val=int(op[1]),int(op[2])\n'
            '  if key in c:c.move_to_end(key)\n'
            '  c[key]=val\n'
            '  if len(c)>k:c.popitem(last=False)'
        ),
    },
    {
        'id': 'coin_change',
        'description': (
            'Minimum coins to make amount using given denominations.\n'
            'Print -1 if impossible.\n'
            'Input line 1: space-separated coin denominations\n'
            'Input line 2: target amount\nOutput: minimum coins'
        ),
        'tests': [
            {'input': '1 5 6 9\n11', 'expected': '2'},
            {'input': '2\n3',        'expected': '-1'},
            {'input': '1 2 5\n11',   'expected': '3'},
        ],
        'difficulty': 0.5,
        'solution': (
            'coins=list(map(int,input().split()));amt=int(input())\n'
            'dp=[float("inf")]*(amt+1);dp[0]=0\n'
            'for i in range(1,amt+1):\n for c in coins:\n  if c<=i:dp[i]=min(dp[i],dp[i-c]+1)\n'
            'print(-1 if dp[amt]==float("inf") else dp[amt])'
        ),
    },
    {
        'id': 'word_ladder',
        'description': (
            'Find shortest word ladder length from start to end (change one letter at a time).\n'
            'All words same length. Print 0 if no path.\n'
            'Input line 1: start word\nInput line 2: end word\n'
            'Input line 3: space-separated word list\nOutput: ladder length'
        ),
        'tests': [
            {'input': 'hit\ncog\nhot dot dog lot log cog', 'expected': '4'},
            {'input': 'hit\ncog\nhot dot',                  'expected': '0'},
        ],
        'difficulty': 0.6,
        'solution': (
            'from collections import deque\n'
            'start=input();end=input();words=set(input().split())\n'
            'if end not in words:print(0);exit()\n'
            'q=deque([(start,1)]);visited={start}\n'
            'while q:\n word,steps=q.popleft()\n'
            ' for i in range(len(word)):\n'
            '  for c in "abcdefghijklmnopqrstuvwxyz":\n'
            '   nw=word[:i]+c+word[i+1:]\n'
            '   if nw==end:print(steps+1);exit()\n'
            '   if nw in words and nw not in visited:visited.add(nw);q.append((nw,steps+1))\n'
            'print(0)'
        ),
    },
    {
        'id': 'matrix_chain',
        'description': (
            'Minimum scalar multiplications for matrix chain.\n'
            'Input: N, then N+1 dimensions (d0 d1 ... dN means matrices are d0xd1, d1xd2, etc.)\n'
            'Output: minimum multiplications'
        ),
        'tests': [
            {'input': '3\n40 20 30 10', 'expected': '26000'},
            {'input': '2\n10 30 5',      'expected': '1500'},
        ],
        'difficulty': 0.65,
        'solution': (
            'n=int(input());dims=list(map(int,input().split()))\n'
            'dp=[[0]*n for _ in range(n)]\n'
            'for length in range(2,n+1):\n'
            ' for i in range(n-length+1):\n'
            '  j=i+length-1;dp[i][j]=float("inf")\n'
            '  for k in range(i,j):dp[i][j]=min(dp[i][j],dp[i][k]+dp[k+1][j]+dims[i]*dims[k+1]*dims[j+1])\n'
            'print(dp[0][n-1])'
        ),
    },
    {
        'id': 'regex_match',
        'description': (
            'Implement regex matching with . (any char) and * (zero or more of prev).\n'
            'Input line 1: string\nInput line 2: pattern\nOutput: True or False'
        ),
        'tests': [
            {'input': 'aa\na*',   'expected': 'True'},
            {'input': 'ab\n.*',   'expected': 'True'},
            {'input': 'aab\nc*a*b', 'expected': 'True'},
            {'input': 'aa\na',    'expected': 'False'},
        ],
        'difficulty': 0.7,
        'solution': (
            's=input();p=input()\n'
            'def m(i,j):\n'
            ' if j==len(p):return i==len(s)\n'
            ' f=i<len(s) and p[j] in(s[i],".")\n'
            ' if j+1<len(p) and p[j+1]=="*":return m(i,j+2) or (f and m(i+1,j))\n'
            ' return f and m(i+1,j+1)\n'
            'print(m(0,0))'
        ),
    },
    {
        'id': 'sudoku_solver',
        'description': (
            'Solve a 9x9 sudoku. Empty cells are 0.\n'
            'Input: 9 rows of 9 space-separated integers\n'
            'Output: solved grid, same format'
        ),
        'tests': [
            {
                'input': (
                    '5 3 0 0 7 0 0 0 0\n'
                    '6 0 0 1 9 5 0 0 0\n'
                    '0 9 8 0 0 0 0 6 0\n'
                    '8 0 0 0 6 0 0 0 3\n'
                    '4 0 0 8 0 3 0 0 1\n'
                    '7 0 0 0 2 0 0 0 6\n'
                    '0 6 0 0 0 0 2 8 0\n'
                    '0 0 0 4 1 9 0 0 5\n'
                    '0 0 0 0 8 0 0 7 9'
                ),
                'expected': (
                    '5 3 4 6 7 8 9 1 2\n'
                    '6 7 2 1 9 5 3 4 8\n'
                    '1 9 8 3 4 2 5 6 7\n'
                    '8 5 9 7 6 1 4 2 3\n'
                    '4 2 6 8 5 3 7 9 1\n'
                    '7 1 3 9 2 4 8 5 6\n'
                    '9 6 1 5 3 7 2 8 4\n'
                    '2 8 7 4 1 9 6 3 5\n'
                    '3 4 5 2 8 6 1 7 9'
                ),
            }
        ],
        'difficulty': 0.8,
        'solution': None,  # intentionally no solution — brain must figure it out
    },
]


def builtin_problems(
    difficulty_range: tuple[float, float] = (0.0, 1.0),
    shuffle: bool = True,
    seed:    int  = 42,
) -> list[dict]:
    """Return built-in problems filtered by difficulty range."""
    lo, hi = difficulty_range
    filtered = [p for p in _BUILTIN_PROBLEMS
                if lo <= p['difficulty'] <= hi]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(filtered)
    return filtered


def training_problems(seed: int = 42) -> list[dict]:
    """20-60% difficulty zone — optimal for RL training."""
    return builtin_problems(difficulty_range=(0.2, 0.65), seed=seed)


def eval_problems(seed: int = 99) -> list[dict]:
    """Held-out evaluation set — harder problems."""
    return builtin_problems(difficulty_range=(0.5, 1.0), shuffle=False)


def stream_problems(
    problems: list[dict],
    repeat: bool = True,
    seed:   int  = 42,
) -> Iterator[dict]:
    """Infinite (or single-pass) iterator over problems."""
    rng = random.Random(seed)
    while True:
        shuffled = list(problems)
        rng.shuffle(shuffled)
        yield from shuffled
        if not repeat:
            break
