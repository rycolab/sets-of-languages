import math

from pytest import approx

from recognizers.calibration.ngram_lm import (
    get_ngrams,
    SpecialSymbol,
    NGramLanguageModel
)

b = SpecialSymbol.BOS
e = SpecialSymbol.EOS

def test_get_ngrams() -> None:
    assert list(get_ngrams('abcdef', 4)) == [
        (b, b, b, 'a'),
        (b, b, 'a', 'b'),
        (b, 'a', 'b', 'c'),
        ('a', 'b', 'c', 'd'),
        ('b', 'c', 'd', 'e'),
        ('c', 'd', 'e', 'f'),
        ('d', 'e', 'f', e)
    ]

def test_ngram_lm() -> None:
    lm = NGramLanguageModel(n=2, vocabulary_size=2, examples=[
        'a',
        'aa',
        'ab'
    ])
    assert math.exp(lm.conditional_log_probability((b,), 'a', k=0)) == 1
    assert math.exp(lm.conditional_log_probability((b,), 'b', k=0)) == 0
    assert math.exp(lm.conditional_log_probability((b,), e, k=0)) == 0
    assert math.exp(lm.conditional_log_probability(('a',), 'a', k=0)) == approx(1/4)
    assert math.exp(lm.conditional_log_probability(('a',), 'b', k=0)) == approx(1/4)
    assert math.exp(lm.conditional_log_probability(('a',), e, k=0)) == approx(2/4)
    assert math.exp(lm.conditional_log_probability(('b',), 'a', k=0)) == 0
    assert math.exp(lm.conditional_log_probability(('b',), 'b', k=0)) == 0
    assert math.exp(lm.conditional_log_probability(('b',), e, k=0)) == 1
    assert math.exp(lm.conditional_log_probability((b,), 'a', k=1)) == approx(4/6)
    assert math.exp(lm.conditional_log_probability((b,), 'b', k=1)) == approx(1/6)
    assert math.exp(lm.conditional_log_probability((b,), e, k=1)) == approx(1/6)
    assert math.exp(lm.conditional_log_probability(('a',), 'a', k=1)) == approx(2/7)
    assert math.exp(lm.conditional_log_probability(('a',), 'b', k=1)) == approx(2/7)
    assert math.exp(lm.conditional_log_probability(('a',), e, k=1)) == approx(3/7)
    assert math.exp(lm.conditional_log_probability(('b',), 'a', k=1)) == approx(1/4)
    assert math.exp(lm.conditional_log_probability(('b',), 'b', k=1)) == approx(1/4)
    assert math.exp(lm.conditional_log_probability(('b',), e, k=1)) == approx(2/4)
