#!/usr/bin/env python3
# Standard Library
# Standard Library
# Standard Library
import logging
import re
from array import array
from copy import deepcopy
from functools import reduce
from os import getenv, listdir
from os.path import isfile, join
from random import choices
from typing import Any, Dict, Iterator, List, Match, Optional, Set, Tuple, Union

TOKENIZE_REGEX = re.compile(r"(?P<token>(?P<punc>[,;])|(?P<sentend>\.(\.\.)?|\?(\?\?)?|!(!!)?)|(?P<word>([A-Za-z]?[a-z]+|[IAO])(-[a-z]+)*('[a-z]{0,4})?)|(?P<dash>--?)|(?P<qt>[\"']))", re.A | re.S)

SENT_ENDINGS: Set[str] = {'.', '?', '!', '...', '???', '!!!'}
SENT_ENDINGS_LIST: List[str] = list(SENT_ENDINGS)
SENT_ENDINGS_WEIGHTS: List[float] = \
    [0.55, 0.2, 0.1, 0.05, 0.05, 0.05]

DONT_REPEAT_TAGS: Set[str] = {'sentend', 'punc', 'qt', 'dash'}

# set loging level using $ LOG=info ./corpus.py
logging.basicConfig(format='%(levelname)s %(message)s')
log = logging.getLogger('Corpus Logger')
log.setLevel(eval(f'logging.{getenv("LOG", "debug").upper()}', locals(), globals()))


def tokenize(txt: str) -> Iterator[Match]:
    global TOKENIZE_REGEX
    return TOKENIZE_REGEX.finditer(txt)


def make_index(xs: List[Any]) -> Tuple[List[Any], Dict[Any, int]]:
    """Make index and reverse index.
    """
    d: Dict[Any, int] = dict()
    a: List[Any] = []
    idx = 0
    for x in xs:
        if d.get(x, None) is None:
            d[x] = idx
            a.append(x)
            idx += 1
    return a, d


def prettify(tokens: Union[Tuple[str, ...], List[str]]) -> str:
    """Transform list of words to legible text suitable for human consumption.
    """
    global SENT_ENDINGS
    global SENT_ENDINGS_LIST
    global SENT_ENDINGS_WEIGHTS
    global TOKENIZE_REGEX
    ms: List[Match] = list(TOKENIZE_REGEX.finditer(' '.join(tokens)))
    chunks = [ms[0].group(0)]
    inside_qt = False

    for i, m in enumerate(ms[1:], 1):
        prev = ms[i - 1]

        if m.group('qt') is not None:
            if inside_qt:
                inside_qt = False
            else:
                inside_qt = True
                if prev.group('word') is not None or prev.group('sentend') is not None:
                    chunks.append(' ')
                    chunks.append(m.group(0))
                    continue
                # 2x " in a row
                elif prev.group('qt') is not None:
                    inside_qt = False
                    chunks.pop()
                    continue

        if not inside_qt and prev.group('qt') is not None and m.group('word') is not None:
            chunks.append(' ')
            chunks.append(m.group(0))
        elif prev.group('word') is not None and m.group('word') is not None:
            chunks.append(' ')
            chunks.append(m.group(0))
        elif prev.group('dash') is not None and m.group('word') is not None:
            chunks.append(' ')
            chunks.append(m.group(0))
        elif prev.group('punc') is not None and m.group('word') is not None:
            chunks.append(' ')
            chunks.append(m.group(0))
        elif prev.group('sentend') is not None and m.group('word') is not None:
            chunks.append(' ')
            chunks.append(m.group(0)[0].upper() + m.group(0)[1:])
        else:
            chunks.append(m.group(0))

    if chunks[-1] not in SENT_ENDINGS:
        chunks.append(choices(population=SENT_ENDINGS_LIST, weights=SENT_ENDINGS_WEIGHTS)[0])

    if chunks[0][0].islower():
        chunks[0] = chunks[0][0].upper() + chunks[0][1:]

    return ''.join(chunks)


class Corpus:
    def __init__(self, text: str, name: str = ''):
        self.text = text
        self.name = name

        self._dummy_index: List[int] = None

        # map and rev-map
        self._word_to_idx: Dict[str, int] = None
        self._idx_to_word: List[str] = None
        # array index is the word numer
        self._word_ps: array = None
        # iterate over all words as word numers
        self._word_idxs: array = None

        self._nword_ps: List[Dict[Tuple[int, ...], Dict[int, float]]] = \
            [None for i in range(30)]

        log.warn('make sure to call `.index()` to initialise')

    @staticmethod
    def files() -> List[str]:
        """List of files in the corpus.
        """
        return [f for f in listdir('./data') if isfile(join('./data', f))]

    @staticmethod
    def from_file(name: str, chunk_size=10, delta=2, max_n_errors=10):
        """Create Corpus from a single file name.

        NOTE it will look for it in the corpus ie. ./data dir).
        """
        if not name.endswith('.txt'):
            return Corpus.from_file(f'{name}.txt')
        with open(f'./data/{name}', encoding='ascii', errors='ignore') as f:
            nerrors = 0
            chunks = []
            step = chunk_size
            try:
                result = f.read(step)
                while result:
                    chunks.append(result)
                    step *= delta
                    result = f.read(step)
            except UnicodeDecodeError as e:
                nerrors += 1
                step = chunk_size
                log.warn(str(e))
                if nerrors >= max_n_errors:
                    log.warn(f'failed to read file {name}')
            return Corpus(''.join(chunks), name)

    @staticmethod
    def from_all_files(max_n_files=100, delta=2, chunk_size=10, max_n_errors=10):
        texts = []
        files = Corpus.files()[:max_n_files]
        log.info(f'reading from {len(files)} files in ./data')
        for fname in files:
            log.debug(f'reading from {fname}')
            with open(join('./data', fname), encoding='ascii', errors='ignore') as f:
                chunks = []
                step = chunk_size
                nerrors = 0
                try:
                    result = f.read(step)
                    while result:
                        chunks.append(result)
                        step *= delta
                        result = f.read(step)
                except UnicodeDecodeError as e:
                    nerrors += 1
                    step = chunk_size
                    log.warn(str(e))
                    if nerrors >= max_n_errors:
                        log.warn(f'failed to read file {fname}')
                        break
                texts.append(''.join(chunks))
        txt = '\n\n'.join(texts)
        log.info(f'done reading from all {len(files)} files (read {len(txt)} bytes)')
        return Corpus(txt, f'{len(texts)} texts from ./data')

    def index(self) -> None:
        """Initialise the Corpus.

        Associate a natural number with each word (INDEX)
        and with each such natural number a word (REV-INDEX).
        """
        if self._idx_to_word is not None:
            log.debug('cache HIT, index already generated')
            return

        global DONT_REPEAT_TAGS

        words: List[str] = []

        log.info('tokenizing')

        prev_tag: str = 'word'

        # pre-process
        nremoved = 0

        for m in tokenize(self.text):
            tag: str = None
            # extract tag
            for g in m.groupdict():
                if g != 'token' and m.group(g) is not None:
                    tag = g
                    break

            # skip
            if prev_tag == tag:
                if tag in DONT_REPEAT_TAGS:
                    nremoved += 1
                    continue

            # else add full match
            words.append(m.group('token'))
            prev_tag = tag

        log.info(f'created {len(words)} tokens, ratio of nchars to ntokens = {len(self.text) / len(words):4.2f}')
        log.info(f'pre-processing removed {nremoved} tokens (corpus smaller by {nremoved / (nremoved + len(words)):6.4f})')

        idx_to_word, word_to_idx = make_index(words)
        self._idx_to_word = idx_to_word
        self._word_to_idx = word_to_idx
        # homogenous u32 array
        self._word_idxs = array('L', [0 for i in range(len(words))])

        ptr = 0
        for w in words:
            self._word_idxs[ptr] = self._word_to_idx[w]
            ptr += 1

        del self.text  # dealloc

    def nword_ps(self, n=2, min_freq=2, min_p=0.0001, min_total=2) -> Dict[Tuple[int, ...], Dict[int, float]]:
        """Compute probabilites for every n-gram and the word after.
        """
        if n < 2:
            raise Exception(f'n MUST be >= 2')

        elif self._nword_ps[n] is not None:
            log.debug(f'cache HIT for {n}-word probabilities')
            return self._nword_ps[n]

        log.debug(f'cache MISS for {n}-word probabilities, generating')

        def nword_counts(self) -> Dict[Tuple[int, ...], Dict[int, int]]:
            """Get counts of every n-tuple of words.
            """
            d: Dict[Tuple[int, ...], Dict[int, int]] = dict()

            # look ahead
            for ptr in range(len(self._word_idxs) - n - 1):
                record: Tuple[int, ...] = tuple(self._word_idxs[ptr:ptr + n])
                next_w: int = self._word_idxs[ptr + n]
                maybe_dict = d.get(record, None)
                if maybe_dict is None:
                    d[record] = {next_w: 1}
                else:
                    maybe_dict[next_w] = maybe_dict.get(next_w, 0) + 1

            return d

        ps: Dict[Tuple[int, ...], Dict[int, float]] = nword_counts(self)

        npruned_dicts = 0
        log.info(f'prunning {n}-gram probability dict (len = {len(ps)} dicts)')

        for word_seq in list(ps.keys()):

            if len(ps[word_seq]) < min_total:
                del ps[word_seq]
                npruned_dicts += 1
                continue

            # prune using ABSOLUTE frequencies to save memory
            counts: Dict[int, int] = {widx: count for widx, count in ps[word_seq].items() if count >= min_freq}

            if len(counts) < min_total:
                del ps[word_seq]
                npruned_dicts += 1
                continue

            total: int = sum(counts.values())

            probs: Dict[int, float] = counts

            del counts

            for k in probs:
                probs[k] /= total

            # prune using RELATIVE frequencies to save memory
            probs = {widx: p for widx, p in probs.items() if p >= min_p}

            if len(probs) < min_total:
                del ps[word_seq]
                npruned_dicts += 1
            else:
                ps[word_seq] = probs

        self._nword_ps[n] = ps

        log.info(f'pruned {npruned_dicts} dicts (reduction by {1 - (len(ps) / (len(ps) + npruned_dicts)):6.4f}, {len(ps)} entries left)')

        return ps

    def word_ps(self) -> array:
        if self._word_ps is not None:
            return self._word_ps

        def word_counts(self) -> array:
            counts = array('L', [0 for _ in range(len(self._idx_to_word))])
            for w_idx in self._word_idxs:
                counts[w_idx] += 1
            return counts

        counts = word_counts(self)
        total: int = sum(counts)
        ps = array('f', [0 for _ in range(len(self._idx_to_word))])

        for w_idx in self._word_idxs:
            ps[w_idx] = counts[w_idx] / total

        self._word_ps = ps
        return ps

    @property
    def rand_word_idx(self) -> int:
        if self._dummy_index is None:
            self._dummy_index = list(range(len(self.word_ps())))
        return choices(
            population=self._dummy_index,
            weights=self._word_ps,
            k=1)[0]

    def markov(self, s='Once upon a time there was a', n=2, max_len=100) -> List[str]:
        words: List[str] = [m.group('token') for m in tokenize(s)]
        word_idxs = [self._word_to_idx.get(w, None) for w in words]
        usefulness = [0 for _ in range(n + 1)]
        for i in range(len(word_idxs)):
            if word_idxs[i] is None:
                log.warn(f'unrecognised word {words[i]} (unseen in corpus), replacing with a rand word')
                word_idxs[i] = self.rand_word_idx

        del words

        while len(word_idxs) < max_len:
            found = False

            for i in range(n, 1, -1):  # upper is exclusive so iterate up to n..2

                slice: Tuple[int, ...] = tuple(word_idxs[-i:])
                words: List[str] = []

                for idx, w in zip(map(str, slice), map(lambda idx: self._idx_to_word[idx], slice)):
                    words.append(w)

                words_s = ' '.join(words)

                nword_ps_dict = self.nword_ps(n=i)
                maybe: Optional[Dict[int, float]] \
                        = nword_ps_dict.get(slice, None)

                if maybe is not None:
                    usefulness[n] += 1
                    found = True
                    # needs indexing
                    idxs = list(maybe.keys())
                    ps = maybe.values()
                    rand_widx: int = choices(population=idxs, weights=ps, k=1)[0]

                    candidate_width: int = \
                        reduce(max, (len(cand) for cand in map(lambda idx: self._idx_to_word[idx], idxs)))

                    log.debug(f'{words_s} {" ".rjust(candidate_width)} (last {i} words)')

                    for cand, prob in zip(idxs, ps):
                        to_log = f'{words_s} {self._idx_to_word[cand].rjust(candidate_width)} [{prob:6.4f}]'
                        if cand == rand_widx:
                            to_log += ' CHOSEN'
                        log.info(to_log)

                    word_idxs.append(rand_widx)
                    break

            if not found:
                usefulness[0] += 1
                rand_widx = self.rand_word_idx
                log.debug(f'appending rand word {self._idx_to_word[rand_widx]} [FAILED TO FIND]')
                word_idxs.append(rand_widx)

        total_usefulness: int = sum(usefulness)
        for i in range(len(usefulness)):
            log.debug(f'relative usefullness of {i} lookbehind = {usefulness[i] / total_usefulness}, from {usefulness[i]} instances')

        return [self._idx_to_word[idx] for idx in word_idxs]

    def __repr__(self) -> str:
        return f'Corpus {self.name}' + (' UINDEXED' if self._idx_to_word is None else '')


if __name__ == '__main__':
    import cmd
    log.info('running as a script')
    d = Corpus.from_all_files()
    d.index()

    n = 5
    max_len = 200

    class REPL(cmd.Cmd):
        intro = 'Welcome to Markov Story Generator'
        prompt = 'Begin you story ... => '
        file = None
        ruler = '-'

        def do_story(self, arg):
            words = d.markov(s=arg, n=n, max_len=max_len)
            print(prettify(words))

        def do_set_lookbehind(self, arg):
            global n
            n = int(arg)
            print(f'lookbehind set to {n}')

        def do_set_len(self, arg):
            global max_len
            max_len = int(arg)
            print(f'len set to {max_len}')

        def do_set_logging_lvl(self, arg):
            log.setLevel({'critical': 50, 'error': 40, 'warning': 30, 'info': 20, 'debug': 10, 'noset': 0}[arg.lower()])
            print(f'logging set to {arg}')

        def do_exit(self, arg):
            from sys import exit
            exit(0)

    REPL().cmdloop()

# vim:sw=4:ts=8:expandtab:foldmethod=indent:nu:hlsearch:
