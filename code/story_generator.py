#!/usr/bin/env python3
"""
This application implement a Markov Chain (MC) which is used to generate
stories from intial strings provided by users.

It has simple but interactive command line interface. Once the model is
built, stories are generated instantaneously in response to users' input.

You can set options that the MC runs by supplying flags such as -l <INT>.
Standard UNIX conventions are used and you can see all flags with:

    $ python3 ./story_generator.py -h

To see what is happening under the hood, you can lower the logging level.
Various metrics are displayed on 'debug'. You can do that by running it
with the -v <debug|info|warning> flag. E.g.:

    $ python3 ./story_generator.py -v debug

I am making use of Python's standard library but all the remaining code
is mine.

Norbert Logiewa 2019
"""
# Standard Library
import logging
import re
from string import capwords
from collections import deque
from array import array
from functools import reduce
from os import listdir
from os.path import isfile, join, dirname
from random import choices
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Match,
    Optional,
    Sequence,
    Tuple,
)

MAX_FILES = 100

# more efficient than NLTK tokenize
TOKENIZE_REGEX = re.compile(r"(?P<token>(?P<punc>[,;])|(?P<sentend>\.(\.\.)?|\?(\?\?)?|!(!!)?)|(?P<word>([A-Za-z]?[a-z]+|[IAO])(-[a-z]+)*('[a-z]{0,4})?)|(?P<dash>--?)|(?P<qt>[\"']))", re.A | re.S)

# for lookup
SENT_ENDINGS = {'.', '?', '!', '...', '???', '!!!'}

# for indexing
SENT_ENDINGS_LIST = list(SENT_ENDINGS)

# fullstop most common
SENT_ENDINGS_WEIGHTS = [0.55, 0.2, 0.1, 0.05, 0.05, 0.05]

# set loging level using ./story_generator.py -v debug | info | warning
logging.basicConfig(format='[%(levelname)s] %(message)s')
log = logging.getLogger('StoryGenerator Logger')


def tokenize(txt: str) -> Iterator[Match]:
    """Generate regular expression matches from text.
    """
    global TOKENIZE_REGEX
    return TOKENIZE_REGEX.finditer(txt)


def make_index(xs: Iterable[Any]) -> Tuple[List[Any], Dict[Any, int]]:
    """Make index and reverse index.

    Types:
        index: List[T]
        rev:   Dict[T, int]
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


def prettify(tokens: Sequence[str]) -> str:
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
                if prev.group('word') is not None or \
                        prev.group('sentend') is not None:
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
            # if prev.group(0)[0].isupper() and len(prev.group(0)) >= 2 and any((m.group(0) == prev.group(0).lower() for m in ms)):
                # chunks[-1] = chunks[-1][0].lower() + chunks[-1][1:]
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
        chunks.append(choices(
            population=SENT_ENDINGS_LIST,
            weights=SENT_ENDINGS_WEIGHTS)[0])

    if chunks[0][0].islower():
        chunks[0] = chunks[0][0].upper() + chunks[0][1:]

    return ''.join(chunks)


class StoryGenerator:
    def __init__(self, text: str, n: int, min_freq: int, min_p: float, min_entries: int, name: str = '', **kwargs):
        """Constructs a StoryGenerator.

        `text`:
            is the corpus (you will need to read a file and supply
            the string,

        `n`:
            is the amount of lookbehind (how many previous states the
            Markov Chain will keep track of). Raising it above 7-8
            is pointless.

        `min_freq`:
            is the minumum frequency of words. Raising it causes
            above 1 causes drastic pruning.

        `min_p`:
            is the minumum probability of words. Raising it causes
            pruning.

        `min_entries`:
            is the minumum number of entries in each
            dict. Raising it above 1 causes drastic pruning.

        `name`:
            you can give the StoryGenerator a name (this is used
            for printing)
        """
        self.name = name

        self._ngram_ps: List[
                Optional[Dict[Tuple[int, ...], Dict[int, float]]]] = \
            [None for i in range(30)]

        self.n = n
        self.min_freq = min_freq
        self.min_p = min_p
        self.min_entries = min_entries

        words: deque = deque()

        prev_tag = 'word'
        nremoved = 0
        norepeat = {'sentend', 'punc', 'qt', 'dash'}

        log.info('tokenizing & pre-processing')
        for m in tokenize(text):
            tag: Optional[str] = None
            # extract tag
            for g in m.groupdict():
                if g != 'token' and m.group(g) is not None:
                    tag = g
                    break
            # skip
            if prev_tag == tag:
                if tag in norepeat:
                    nremoved += 1
                    continue

            # else add full match
            words.append(m.group('token'))
            prev_tag = tag

        log.info(f'created {len(words)} tokens, ratio of nchars to ntokens = {len(text) / len(words):4.2f}')
        log.info(f'pre-processing removed {nremoved} tokens (corpus smaller by {nremoved / (nremoved + len(words)):6.4f})')

        # index and reverse index
        idx_to_word, word_to_idx = make_index(words)
        self.idx_to_word = idx_to_word
        self.word_to_idx = word_to_idx

        # homogenous u32 array
        self.word_idxs = array('L', (self.word_to_idx[w] for w in words))

    @staticmethod
    def files() -> List[str]:
        """List files in the corpus.
        """
        dir: str = join(dirname(__file__), 'data')
        return [f for f in listdir(dir) if isfile(join(dir, f))]

    @staticmethod
    def from_file(
            name: str, n: int, min_freq: int, min_p: float, min_entries: int):
        """Create StoryGenerator from a single file name.

        NOTE it will look for it in the ./data dir).
        """
        if not name.endswith('.txt'):
            return StoryGenerator.from_file(f'{name}.txt')

        chunk_size = 10
        delta = 2
        max_n_errors = 10

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
            return StoryGenerator(''.join(chunks), name, **kwargs)

    @staticmethod
    def from_all_files(
            n: int, min_freq: int, min_p: float, min_entries: int, **kwargs) -> object:
        max_n_errors = 10
        chunk_size = 10
        delta = 2
        dir: str = join(dirname(__file__), 'data')
        texts = []
        files: List[str] = StoryGenerator.files()[:MAX_FILES]
        log.info(f'reading from {len(files)} files in ./data')
        for fname in files:
            log.debug(f'reading from {fname}')
            with open(
                    join(dir, fname), encoding='ascii', errors='ignore') as f:
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
        return StoryGenerator(
            txt,
            n,
            min_freq,
            min_p,
            min_entries,
            f'{len(texts)} texts from ./data',
        )

    def ngram_ps(self, n: int) -> Dict[Tuple[int, ...], Dict[int, float]]:
        """Compute conditional probabilites for words occurring after
        every n-gram.
        """
        if n < 1:
            raise Exception(f'n MUST be >= 1')

        elif self._ngram_ps[n] is not None:
            # log.debug(f'cache HIT for {n}-gram probabilities')
            return self._ngram_ps[n]

        log.debug(f'[cache MISS] generating {n}-gram probabilities, min_freq = {self.min_freq}, min_p = {self.min_p}, min_entries = {self.min_entries}')

        def ngram_counts(self) -> Dict[Tuple[int, ...], Dict[int, int]]:
            """Get counts of every n-gram of words.
            """
            d: Dict[Tuple[int, ...], Dict[int, int]] = dict()

            # look ahead
            for ptr in range(len(self.word_idxs) - n - 1):
                record: Tuple[int, ...] = tuple(self.word_idxs[ptr:ptr + n])
                next_w: int = self.word_idxs[ptr + n]
                maybe_dict = d.get(record, None)
                if maybe_dict is None:
                    d[record] = {next_w: 1}
                else:
                    maybe_dict[next_w] = maybe_dict.get(next_w, 0) + 1

            return d

        ps: Dict[Tuple[int, ...], Dict[int, float]] = ngram_counts(self)

        if self.min_freq == 1 and self.min_entries == 1 and self.min_p == 1E-4:
            log.debug(f'not pruning {n}-gram probabilities')
            for word_seq in ps:
                counts: Dict[int, int] = ps[word_seq]
                total: int = sum(counts.values())
                for k in counts:
                    counts[k] /= total
                ps[word_seq] = counts
        else:
            log.info(f'pruning {n}-gram probability dict (len = {len(ps)} dicts)')
            npruned_dicts = 0
            for word_seq in list(ps.keys()):
                if len(ps[word_seq]) < self.min_entries:
                    del ps[word_seq]
                    npruned_dicts += 1
                    continue

                # prune using ABSOLUTE frequencies to save memory
                counts: Dict[int, int] = {widx: count for widx, count in ps[word_seq].items() if count >= self.min_freq}

                if len(counts) < self.min_entries:
                    del ps[word_seq]
                    npruned_dicts += 1
                    continue

                total: int = sum(counts.values())

                probs: Dict[int, float] = counts

                del counts

                for k in probs:
                    probs[k] /= total

                # prune using RELATIVE frequencies to save memory
                probs = {widx: p for widx, p in probs.items() if p >= self.min_p}

                if len(probs) < self.min_entries:
                    del ps[word_seq]
                    npruned_dicts += 1
                else:
                    ps[word_seq] = probs

            log.info(f'pruned {npruned_dicts} dicts (reduction by {1 - (len(ps) / (len(ps) + npruned_dicts)):6.4f}, {len(ps)} entries left)')

        self._ngram_ps[n] = ps
        return ps

    @property
    def word_ps(self) -> array:
        """Gets absolute (not-conditional) probabilities of words.

        This is used as a fallback mechanism by self.rand_word_idx to
        decide which word to propose, when all n-grams have been tried
        and none has been found.
        """
        if hasattr(self, '_word_ps'):
            return self._word_ps

        def word_counts(self) -> array:
            counts = array('L', [0 for _ in range(len(self.idx_to_word))])
            for w_idx in self.word_idxs:
                counts[w_idx] += 1
            return counts

        counts = word_counts(self)
        total: int = sum(counts)
        ps = array('f', [0 for _ in range(len(self.idx_to_word))])

        for w_idx in self.word_idxs:
            ps[w_idx] = counts[w_idx] / total

        setattr(self, '_word_ps', ps)
        return ps

    @property
    def rand_word_idx(self) -> int:
        """Chooses a random word from the indexed words.

        This is used as a fallback mechanism when all n-grams have been
        tried and none has been found.
        """
        # to avoid re-allocating on every call to self.rand_word_idx
        if not hasattr(self, '_dummy_index'):
            setattr(self, '_dummy_index', list(range(len(self.word_ps))))
        return choices(
            population=self._dummy_index,
            weights=self.word_ps,
            k=1)[0]

    def markov(self, s: str, max_len: int) -> str:
        """Generate stories using Markov Chains.

        Stories are generated from string `s` and are of length `max_len`.

        This is the main function exposed by StoryGenerator. It should
        be called by the API consumer.
        """
        log.info(f'generating story from "{s}"')

        def get_word_idxs(self) -> List[int]:
            words: List[str] = [m.group('token') for m in tokenize(s)]
            word_idxs: List[int] = \
                    [self.word_to_idx.get(w, None) for w in words]
            for i in range(len(word_idxs)):
                if word_idxs[i] is None:
                    log.warn(f'unrecognised word {words[i]} (unseen in corpus), replacing with a rand word')
                    word_idxs[i] = self.rand_word_idx
            return word_idxs

        word_idxs: List[int] = get_word_idxs(self)
        usefulness: List[int] = [0 for _ in range(self.n + 1)]
        ntrans = 0
        nstuck = 0

        while len(word_idxs) < max_len:
            found = False

            # upper is exclusive so iterate up to n..2
            for i in range(self.n, 0, -1):
                slice: Tuple[int, ...] = tuple(word_idxs[-i:])
                words_s: str = \
                        ' '.join((self.idx_to_word[idx] for idx in slice))

                maybe_ps_dict: Optional[Dict[int, float]] = \
                    self.ngram_ps(n=i).get(slice, None)

                # avoid nesting
                if maybe_ps_dict is None:
                    continue

                usefulness[i] += 1
                found = True
                # needs indexing
                cand_idxs = list(maybe_ps_dict.keys())
                ps = maybe_ps_dict.values()
                selected: int = choices(population=cand_idxs, weights=ps, k=1)[0]

                max_cand_width: int = \
                    reduce(max,
                            map(len,
                                map(lambda idx:
                                    self.idx_to_word[idx], cand_idxs)))

                log.debug(f'{words_s} {" ".rjust(max_cand_width)} (last {i} words)')

                if len(ps) == 1:
                    # novelty metric, count how many times
                    # it chose a word with p = 1.0
                    nstuck += 1
                    log.info(f'selected adjacent word "{self.idx_to_word[selected]}" [STUCK]')

                else:
                    # count proper transitions to different pieces of text
                    # not just with p = 1.0
                    ntrans += 1
                    for cand, prob in zip(cand_idxs, ps):
                        log.info(f'{words_s} {self.idx_to_word[cand].rjust(max_cand_width)} [{prob:6.4f}]{" CHOSEN" if cand == selected else ""}')

                word_idxs.append(selected)

                break

            if not found:
                usefulness[0] += 1
                rand_widx = self.rand_word_idx
                log.debug(f'appending rand word {self.idx_to_word[rand_widx]} [FAILED TO FIND]')
                word_idxs.append(rand_widx)

        # metrics
        total_usefulness: int = sum(usefulness)
        for i in range(len(usefulness)):
            log.debug(f'relative usefullness of {i} lookbehind = {usefulness[i] / total_usefulness} ({usefulness[i]} instances)')

        log.debug(f'got stuck {nstuck} times ({nstuck / (nstuck + ntrans)})')
        log.debug(f'transitioned {ntrans} times ({ntrans / (nstuck + ntrans)})')

        return prettify([self.idx_to_word[idx] for idx in word_idxs])

    def __repr__(self) -> str:
        return f'StoryGenerator {self.name}' + \
                (' UINDEXED' if self.idx_to_word is None else '')


if __name__ == '__main__':
    # command line interface
    import readline
    from argparse import Namespace, ArgumentParser

    parser = ArgumentParser(
            prog=capwords(__file__.replace('.py', '').replace('_', ' ').replace('./', '')),
            description="""Generate stories from initial sequence of words.  When run, you will enter a REPL (Read-Print-Eval Loop). A prompt will appear, and you will be able to begin your story by typing an initial sequence of words.""", epilog="")

    parser.add_argument(
        "-n",
        help="how many previous tokens to consider when calculating conditional probabilities",
        metavar='INT',
        type=int,
        choices=set(range(1, 10 + 1)),
        default=5,
    )

    parser.add_argument(
        "-l",
        "--max_len",
        help="length of text to generate",
        default=200,
        type=int,
        metavar='INT',
    )

    parser.add_argument(
        "-f",
        "--min_freq",
        help="min frequency of entries",
        default=1,
        type=int,
        choices=set(range(1, 5 + 1)),
        metavar='INT',
    )

    parser.add_argument(
        "-p",
        "--min_p",
        help="min probability of entries",
        default=1E-4,
        type=float,
        metavar='FLOAT',
    )

    parser.add_argument(
        "-e",
        "--min_entries",
        help="min entries in each dict (prunning)",
        default=1,
        type=int,
        choices=set(range(1, 5 + 1)),
        metavar='INT',
    )

    parser.add_argument(
        "-v",
        "--verbosity",
        choices={'debug', 'info', 'warning'},
        default='info',
        help="increase output verbosity",
    )

    args: Namespace = parser.parse_args()

    log.setLevel(eval(
        f'logging.{args.verbosity.upper()}',
        locals(),
        globals(),
    ))

    story = StoryGenerator.from_all_files(**vars(args))

    readline.parse_and_bind('tab: complete')
    print('''Welcome to StoryGenerator!

HINT: type "exit" or "quit" or CTRL-C or CTRL-D to quit''')

    while True:
        line = None
        try:
            line = input('Begin your story => ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line in {'stop', 'quit', 'exit'}:
            break
        elif line == '':
            continue
        else:
            print(story.markov(s=line, max_len=args.max_len))

# vim:sw=4:ts=8:expandtab:foldmethod=indent:nu:hlsearch:
