# Regular expression matching whitespace:
import re
from unidecode import unidecode
from unicodedata import normalize
from xpinyin import Pinyin
from .numbers import normalize_numbers
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]

_cht_norm = [(re.compile(r'[%s]' % x[0]), x[1]) for x in [
    ('。．；', '.'),
    ('，、', ', '),
    ('？', '?'),
    ('！', '!'),
    ('─‧', '-'),
    ('…', '...'),
    ('《》「」『』〈〉（）', "'"),
    ('：︰', ':'),
    ('　', ' ')
]]

_pinyin = Pinyin()


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


def korean_cleaners(text):
    '''Pipeline for Korean text, including collapses whitespace.'''
    text = collapse_whitespace(text)
    text = normalize('NFKD', text)
    return text


def chinese_cleaners(text):
    '''Pipeline for Chinese text, including collapses whitespace.'''
    for regex, replacement in _cht_norm:
        text = re.sub(regex, replacement, text)
    text = collapse_whitespace(text)
    text = text.strip()
    text = _pinyin.get_pinyin(text, splitter='', tone_marks='numbers')
    return text
