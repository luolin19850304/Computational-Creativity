# Video Script 

Hello. Welcome to a tutorial on how to use my story generator.

## Installation

This application has no dependencies beyond Python. This means that all
you need to have to run it is Python3.6 or newer.  You can either run
it on raptor which has Python3.6

[show output of `python3 --version` on raptor]

or you can download the newest release from <https://www.python.org/downloads/>.

[show how to naviagate to the relevant python version on their website]

In this tutorial I will be running it on raptor. I will also assume you
are in the same directory as the `story_generator.py` file.

[show the `story_generator.py` file in the output of "ls"]

## UNIX CLI

As discussed in the report, there are several extensions that I
introduced.  These extensions allow to modify the behaviour of the
Markov Chain.

For instance, you may change how many tokens the Markov Chain considers
when suggesting the next word. You may also force the Markov Chain to
choose words that are sufficiently common.

The command line interface, which allows you to set these options, follows
standard conventions for command line tools. You can run the program with
"flags" also called "switches" such as `-n` which will set those options.

For example, to make the Markov Chain consider the last 4 words, you
can pass the `-n 4` flag.

[show how to pass the flag along with application name: `python3 ./story_generator.py -n 4`]

This is optional and you may supply 0, 1 or more flags if you wish.

You can refer to the description of all the parameters which is displayed
when you run the application with the `-h` or `--help` flags.

[show output of `python3 ./story_generator.py --help`]

## REPL

When you run the application, you will immediately see logging indicating
that initialisation is being performed.

[run the application and show logging]

The default logging threshold is set to debug. There are several
stages the Markov Chains needs to go through before it can generate
stories. Logging will show you what is being done at which step.

[highlight "reading from 26 files in ./data", "reading from caroll-alice.txt" and everything including and below "done reading fron all 26 files"]

This hopefully makes the actions of the application and the decision
making process more transparent. If you find the amount of logging
annoying, you can re-run the application with the `-v info` flag which
which will cause it to log less.

[show how you can run the application with the `-l info` flag]

Once the corpus of files is read, the text is tokenized and tokens are indexed,
you will be able to start typing. You should be able to see the prompt

[highlight the "Begin your story => " prompt]

which indicates that the application is ready.

The story generator is designed in such a way that the creative process
is interactive and the system considers user input when generating text.

[type "Amy was in a great" and press enter]

Conditional probabilities are calculated lazily so they are computed when
the first request to create a story is made.

[highlight "[cache MISS]"]

Every subsequent story is generated instantaneously.

[press the up arrow and press enter]

As indicated in the report, there are also metrics that help evaluate
the behaviour of the Markov Chain and help to tweak the parameters.

If the system fails to find a combination of `n` words in it's corpus it will
temporarily shrink the "lookbehind" to `n - 1` until it reaches 0 at which
point a random word is proposed.

One of the metrics is "usefulness of lookbehind". You can see how often words
were suggested by considering different amounts of previous words.

[highlight the "relative usefullness" logging information]

A lookbehind of 0

[highlight the "relative usefullness of 0 lookbehind" logging information]

means the word was generated completely randomly without taking
into account previous words.

If `--min_entries` is set to 2, and the `--lookbehind` is set to a high value
such as 7, you may find that the Markov Chain is not using this large
lookbehind at all.

Another metric is the amount of transitions between different chunks of text.
When reading the generated stories, it becomes clear that some sentences are
taken word-for-word from a single piece of text.

[show good quality piece of text that is obviously from the same book]

To see how often the source of text is changed, i.e. how often the Markov
Chain transitions, you can consult this metric.

[higlight "relative usefullness"]

As with "usefulness of lookbehind" there are raw and relative
frequencies.

[higlight the probabilites and frequencies]

The system will make more frequent transitions between texts if lookbehind
is shorter or if `--min_entries` is set to 2 or if `--min_freq` is set
to 2.

Lastly, if you wish to observe at what point the Markov Chain decided to
suggest a word based on the context, you can scroll up and trace the logs.

[scroll up to the first transition]

When suggesting the next word, the system prints the candidates along
with their probabilities

[higlight the whole thing]

and indicates which was chosen.

If the logs say that the Markov Chain is "stuck" it means that it is using the
same source of text.

To exit the application you can type "exit", "quit" or you may press `CTRL-C` or
`CTRL-D`.

[show exiting with "exit" and "CTRL-C"]
