# Humor

## Weaknesses of the Article

- I found it very difficult to process the article just because the area, for me,
  is not that interesting. It took me a couple of days to read it and there
  isn't anything obviously wrong with it. It's just that the results the author
  discusses are very uninteresting. The research seems to have only explored
  language-related humour and this is largely what the paper discusses so the
  title seems a bit misleading because it's not really about humour, it's about
  humour in language. It seems to me that there is a lot of untapped potential
  for generating humour with images by, for example.
- So a major disadvantage of the article is that humor is discussed in a
  very narrow context.
- The results are a bit disappointing because none of the systems seem to be intelligent i.e.
  they cannot be said to understand why the thing they created is amusing,
  there is also no intention behind those systems, they are programs that use
  some fixed template to generate e.g. puns, "punning riddles",
- The humour created by those systems is very crude and often not funny at all
  and is some cases it's not comprehensible.
- We don't really know how often those systems produce something of value.
  It could be once every 10 runs for all we know.
  If the puns are not amusing or are incomprehensible most of the
  time and they still needs to be verified by a human then the system is not useful
  at all.
- Another potential problem is that the systems seem to rely on recombination
  of data that has been fed into them in advance.
  It's questionable if that counts as "creative".
- The research the author mentions (Mihalcea and Strapparava (2006)) was not
  very successful in creating an accurate classifier suggesting it is difficult
  for a computational system to recognise humor.
  It performed well on some data (0.77 \& 0.96 accuracy) but not so well on
  other (0.54 accuracy)...
  Furthermore, the authors of the research used Naive Bayes classifier which is
  probably not the best choice of an algorithm.
  Natural language is context sensitive meaning the things you say before (context)
  affect the meaning of information that follows.
  If, for example, you treat each word as a feature, Naive Bayes classifier would
  assume that each word is a separate (isolated) entity and it doesn't affect the
  meaning of other words which might have been the cause of low accuracy.
  But the researchers also used other classification algorithms.
- As the author mentioned, there is no formal theory of humor.
  We all have an intuitive idea what makes something funny e.g. when
  something is unexpected or out of place or very surprising, but we do not
  have a good formal definition which captures all forms of humor.
  So the implication is that we cannot create a metric to evaluate how funny something is
  and use it as, for example, the loss function in neural networks.
- However, I did a bit of research it's not the case that we don't have theories of humour,
  after some Googling I was able to find:
  - O'Shannon model of humor
  - Benign violation theory
  - Misattribution theory
  - Ontic-Epistemic Theory of Humor
  - Computational-Neural Theory of Humor
  - Script-based Semantic Theory of Humor
  - General Theory of Verbal Humor
  - Incongruous juxtaposition theory
- There is also the issue that humor is a very subjective thing.
  Some thing are not seen as funny by some people and some things are only
  amusing if you have the right information.
  E.g. only computer scientists will find jokes about programming languages amusing.
  It seems to me that for the system to generate humour more sophisticated
  than puns, it must be fed some knowledge about the audience which is
  not mentioned in the article.
- The article mentions that the task of creating more sophisticated humour
  is a very difficult one, even for humans.
  Not everyone would make a great comedian and I'm sure you all have heard people
  fail at humour.
  What we want to do here is something most humans fail at.
  And arguably we fail, because we cannot decompose the process of creating humour
  into some sequence of instructions to carry out.
  This could be because of lack of understanding of humour.
  But since we cannot do that ourselves, how would we program a computer to do it?
  This seems to rule out traditional methods such as writing an algorithm to do it.
- In conclusion, research into computational humor seems to be very lacking BUT
  the article is 10 years old so it's possible that things have improved since
  the article was written.

<!--
vim:sw=2:ts=4:expandtab:
-->
