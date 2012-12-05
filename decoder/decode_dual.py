#!/usr/bin/env python
import optparse
import sys
import models
import pdb 
from collections import namedtuple

def update_y(h,n):
  if h is None:
    return [0]*n
  else:
    return [sum(pair) for pair in zip(  phrase_bits(h,n), update_y(h.predecessor, n))    ]
                
def phrase_bits(h, n): #returns a bit vector for words translated in french phrase
  b = [0]*n
  for i in range(h.start, h.end):
    b[i] += 1
  return b

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
  u = [0 for _ in f]              #Lagrangian                                                                                                 
  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, start, end, num_trans") 
  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, 0, 0, 0) 
  stacks = [{} for _ in f] + [{}]
  stacks[0][lm.begin()] = initial_hypothesis
  num_words = len(f)
  t = 1 #iteration 
  T=100 #total number iterations; may not necessarily converge if not high enough, or at all if too loose?  
  
  for t in range(1,T+1):
    sys.stdout.write('t: ' + str(t) + '\n')
    #end_new - start_new <= num_words - num_words_translated
    for i, stack in enumerate(stacks[:-1]):
      for h in sorted(stack.itervalues(),key=lambda h: -h.logprob): # no pruning
        start = h.start
        end = h.end 
        lm_state = h.lm_state
        start_indices = [xs for xs in xrange(len(f)) if xs < start or xs >= end] #get valid span 
        for j in start_indices:
          end_indices = [ys for ys in xrange(j+1, len(f)+1) if ys <= start or xs > end]
          for k in end_indices:
            if f[j:k] in tm: 
              for phrase in tm[f[j:k]]:
              #log prob now the Lagrangian?
                logprob = h.logprob + phrase.logprob + sum(u[j:k]) #add u's into phrase score g 
                lm_state = h.lm_state
                start2 = None #need to declare?
                end2 = None
                if k == start - 1:
                  start2 = j #need to define variable up top? 
                  end2 = end
                elif j == end + 1:
                  start2 = start
                  end2 = k
                else:
                  start2 = j
                  end2 = k
                for word in phrase.english.split():
                  (lm_state, word_logprob) = lm.score(lm_state, word)
                  logprob += word_logprob
                logprob += lm.end(lm_state) if j == len(f) else 0.0
                num_trans2 = h.num_trans+k-j
                if num_trans2 <= len(f): #if translated N or fewer words from source
                  new_hypothesis = hypothesis(logprob, lm_state, h, phrase, start2, end2, num_trans2)
                  #sys.stdout.write('num_trans2: ' + str(num_trans2) + '\n')
                  #sys.stdout.write('len(stacks): ' + str(len(stacks)) + '\n')
                  if lm_state not in stacks[num_trans2] or stacks[num_trans2][lm_state].logprob < logprob: # second case is recombination
                    stacks[num_trans2][lm_state] = new_hypothesis #replace? or no replacement? many hypotheses can have same lm_state? 
    winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
   
    def get_counts(h): #don't need to initialize y? 
        return [0]*len(h) if h.predecessor is None else [i+j for i, j in zip(h, h)] #h.predecessor? 
    def extract_english(h): 
      return "" if h.predecessor is None else "%s%s%s%s " % (extract_english(h.predecessor), h.phrase.english, str(h.start), str(h.end))
    
    print(extract_english(winner))
    y = update_y(winner,len(f))

    if all(y_i == 1 for y_i in y):
      #done
      print('done')
    else:
      #pdb.set_trace()
      print('y:\n ')
      print(y)
      print('u: \n')
      print(u)
      print('\n')
      #pdb.set_trace()
      u = [u_old - (float(1)/t)*y_t for u_old, y_t in zip(u, y)]
    
    if opts.verbose:
      def extract_tm_logprob(h):
        return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    
