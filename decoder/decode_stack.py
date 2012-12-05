#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple

def start(n):
    for i in range(len(n)):
        if n[i] == '0':
            yield i

def end(n, i): #i = index to start search for 1                
    for j in range(i, len(n)+1):
        if j >= len(n):
            return j
        elif n[j] == '1':
            return j
          
def high_bits(bit_map): #for determining which stack/bucket the hypothesis should go in 
  """returns number of bits in bit vector (given as a long that are high"""
  binary = bin(bit_map)[2:] #convert to binary, cut off "0b"
  sum = 0
  for i in range(len(binary)): #if length = 10, doesn't include 10 (0-9)
    if binary[i] == '1':
      sum += 1
  return sum

def num_trans(n): #takes in bit vector
  sum = 0
  for i in range(len(n)): #if length = 10, doesn't include 10 (0-9)
    if n[i] == '1':
      sum += 1
  return sum

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]

sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
  # The following code implements a monotone decoding
  # algorithm (one that doesn't permute the target phrases).
  # Hence all hypotheses in stacks[i] represent translations of 
  # the first i words of the input sentence. You should generalize
  # this so that they can represent translations of *any* i words.
  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, state")
  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, '0'*len(f)) #initial state: bit vector of len(f) of all zeros 
  stacks = [{} for _ in f] + [{}]
  stacks[0][lm.begin()] = initial_hypothesis
  for i, stack in enumerate(stacks[:-1]):
    for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
      for j in start(h.state):
        for k in range(j+1, end(h.state, j+1)+1):
          if f[j:k] in tm:
            for phrase in tm[f[j:k]]:
              logprob = h.logprob + phrase.logprob
              lm_state = h.lm_state
              for word in phrase.english.split():
                (lm_state, word_logprob) = lm.score(lm_state, word)
                logprob += word_logprob
              new_hypothesis = hypothesis(logprob, lm_state, h, phrase, h.state[:j] + (k-j)*'1' + h.state[k:])
              l = num_trans(new_hypothesis.state)
              state = new_hypothesis.state
              logprob += lm.end(lm_state) if l == len(f) else 0.0 #correct? 
              if state not in stacks[l] or stacks[l][state].logprob < logprob: # second case is recombination
                stacks[l][state] = new_hypothesis 
  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
  def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
  print extract_english(winner)

  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))

