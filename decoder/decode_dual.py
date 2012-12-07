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
optparser.add_option("-e", "--number-iterations", dest="e", default=10, type="int", help="number of iterations (default=10)")
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
  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, start, end, num_trans, y_i") 
  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, 0, 0, 0, [0 for _ in f]) 
  stacks = [{} for _ in f] + [{}]
  stacks[0][(lm.begin(), 0,0)] = initial_hypothesis
  num_words = len(f)
  
  for e in range(1,opts.e+1):
    #end_new - start_new <= num_words - num_words_translated
    for i, stack in enumerate(stacks[:-1]):
      #for s_i in range()
      for h in stack.itervalues(): 
        start = h.start
        end = h.end 
        lm_state = h.lm_state
        start_indices = [xs for xs in xrange(len(f)) if xs < start or xs >= end] 
        for j in start_indices:
          end_indices = [ys for ys in xrange(j+1, len(f)+1) if ys <= start or xs > end]
          for k in end_indices:
            if f[j:k] in tm and (h.num_trans + k - j) <= len(f): 
              
              for phrase in tm[f[j:k]]: 
                logprob = h.logprob + phrase.logprob - sum(u[j:k])  
                lm_state = h.lm_state
                start2 = None #need to declare?
                end2 = None
                if k == start - 1:
                  start2 = j
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
                y_i = h.y_i[:] 
                for idx in range(j,k):
                  y_i[idx] += 1

                new_hypothesis = hypothesis(logprob, lm_state, h, phrase, start2, end2, num_trans2, y_i)             
                
                if (lm_state, start2, end2) not in stacks[num_trans2] or stacks[num_trans2][(lm_state, start2, end2)].logprob < logprob:
                  stacks[num_trans2][(lm_state, start2, end2)] = new_hypothesis  
    
    winner = min(stacks[-1].itervalues(), key=lambda h: h.logprob)
    
    print('\nITERATION ' + str(e) + ' =============================================\n')
    print(sum(u_temp*y_i_temp for u_temp, y_i_temp in zip(u, winner.y_i)))
    print('\n')
    
    def extract_english(h): 
      return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
    
    print(extract_english(winner))
    y = winner.y_i[:]

    violated = 0
    for idx in range(0, len(f)):
      if y[idx] != 1:
        violated += 1
    print('num constraints violated: ' + str(violated) + '\n')
    
    if all(_ == 1 for _ in y):
      print('done')
      #TODO: print out best sentence
    else:
      print('\n')
      print(y)

      print('\n')
      #print(u)
      print('\n')
      u = [u_old - (float(1)/e)*(y_t-1) for u_old, y_t in zip(u, y)]
      #print(u)
    if opts.verbose:
      def extract_tm_logprob(h):
        return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    
