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

def extract_english(h): 
  return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)

def make_bits(i,j,n):
    bits = ''
    for k in xrange(n):
        if k >= i and k < j:
            bits += '1'
        else:
            bits += '0'
    return bits

def make_bits_constraints(C,n):
    bits = ''
    for k in xrange(n):
      if k in C:
        bits += '1'
      else:
        bits += '0'
    return bits


optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
optparser.add_option("-e", "--number-iterations", dest="num_iter", default=10, type="int", help="number of iterations (default=10)")
optparser.add_option("-d", "--minimum-improvement", dest="d", default=.000002, type="int", help="if difference of dual less than this, dual value has stopped improving. start tightening (default=.002)")
#add G, K (number of constraints to add)

opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)

french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]
sys.stderr.write("Decoding %s...\n" % (opts.input,))
print('Iters\tViolations')
  
def optimize(f,C,u):
  print("==============================optimize()===========================")
  L1 = float("-inf")
  L2 = float("-inf")
  t1 = None
  t2 = None
  n = len(f)
  constraints = make_bits_constraints(C, n) #bitmap of constraints 
  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, start, end, num_trans, y_i, bitmap") 
  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, 0, 0, 0, [0 for _ in f] , '0'*n) 
  stacks = [{} for _ in f] + [{}]
  stacks[0][(lm.begin(), 0,0)] = initial_hypothesis #Q: need to index by more things such as bitmap? 
  iter = 1
  
  while True:
    if iter > opts.num_iter:
      break
    print(u)
    for i, stack in enumerate(stacks[:-1]):
      #for s_i in range()
      for h in stack.itervalues(): 
        start = h.start
        end = h.end 
        lm_state = h.lm_state
        start_indices = [xs for xs in xrange(n) if xs < start or xs >= end] 
        for j in start_indices:
          end_indices = [ys for ys in xrange(j+1, n+1) if ys <= start or xs > end]
          for k in end_indices:
            if f[j:k] in tm and (h.num_trans + k - j) <= n: 
              
              for phrase in tm[f[j:k]]:
                phrase_bitmap = make_bits(j,k,n)
                if int(phrase_bitmap, 2) & int(h.bitmap, 2) & int(constraints,2) == 0:
                  logprob = h.logprob + phrase.logprob + sum(u[j:k]) #updating correctly? 
                  lm_state = h.lm_state
  
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
                  logprob += lm.end(lm_state) if j == n else 0.0
                  num_trans2 = h.num_trans+k-j               
                  y_i = h.y_i[:] 
                  for idx in range(j,k):
                    y_i[idx] += 1
                  new_bitmap = bin(int(h.bitmap, 2) | int(phrase_bitmap,2)) #extend bitmap
                  new_hypothesis = hypothesis(logprob, lm_state, h, phrase, start2, end2, num_trans2, y_i, new_bitmap)             
                
                  if (lm_state, start2, end2) not in stacks[num_trans2] or stacks[num_trans2][(lm_state, start2, end2)].logprob < logprob:
                    stacks[num_trans2][(lm_state, start2, end2)] = new_hypothesis

    print("logprob:")
    print(logprob)
    if logprob > L1:
      if L1 != 0:
        L2 = L1
        t2 = t1
        L1 = logprob
        t1 = iter
      else:
        L1 = logprob
        t1 = iter
    elif logprob > L2:
      L2 = logprob
      t2 = iter
    
    print("L1")
    print(L1)
    print("t1")
    print(t1)
    print("L2")
    print(L2)
    print("t2")
    print(t2)
    
    if t1 != None and t2 != None and t1 != t2:
      if (float(L1)-L2)/(t1-t2) < opts.d:
        print("stopped improving!")
        break
    #dual still improving
    winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
    print(extract_english(winner)) #to see if winner improving; delete later
    
    y = winner.y_i[:]
    print("y is: " )
    print(y)
    if all(_ == 1 for _ in y):
      print(extract_english(winner))
      return extract_english(winner)
      break #go to next sentence?? 
    else:
      #print(str(e) + '\t' + str(violated))
      u = [u_old - (float(1)/(1+iter))*(y_t-1) for u_old, y_t in zip(u, y)]
      #print(u)
    iter +=  1

  count = [0]*n
  K = 10 #change to 10 later;  
  for i in range(1, K):     
    #end_new - start_new <= num_words - num_words_translated
    for i, stack in enumerate(stacks[:-1]):
      #for s_i in range()
      for h in stack.itervalues(): 
        start = h.start
        end = h.end 
        lm_state = h.lm_state
        start_indices = [xs for xs in xrange(n) if xs < start or xs >= end] 
        for j in start_indices:
          end_indices = [ys for ys in xrange(j+1, n+1) if ys <= start or xs > end]
          for k in end_indices:
            if f[j:k] in tm and (h.num_trans + k - j) <= n: 
              
              for phrase in tm[f[j:k]]:
                phrase_bitmap = make_bits(j,k,n)
                if int(phrase_bitmap, 2) & int(h.bitmap, 2) & int(constraints,2) == 0:                
                  logprob = h.logprob + phrase.logprob + sum(u[j:k]) #updating correctly? 
                  lm_state = h.lm_state
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
                  logprob += lm.end(lm_state) if j == n else 0.0
                  num_trans2 = h.num_trans+k-j               
                  y_i = h.y_i[:] 
                  for idx in range(j,k):
                    y_i[idx] += 1
                  
                  new_bitmap = bin(int(h.bitmap, 2) | int(phrase_bitmap,2)) #extend bitmap
                  new_hypothesis = hypothesis(logprob, lm_state, h, phrase, start2, end2, num_trans2, y_i, new_bitmap)             
                
                  if (lm_state, start2, end2) not in stacks[num_trans2] or stacks[num_trans2][(lm_state, start2, end2)].logprob < logprob:
                    stacks[num_trans2][(lm_state, start2, end2)] = new_hypothesis

    #dual still improving
    winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
    
    y = winner.y_i[:]
    print("y is: " )
    print(y)
    if all(_ == 1 for _ in y):
      print(extract_english(winner))
      return extract_english(winner)
      break #go to next sentence?? 
    else:
      u = [u_old - (float(1)/(1+iter))*(y_t-1) for u_old, y_t in zip(u, y)]
      for idx in range(0, n):
        if y[idx] != 1:
          count[idx] += 1
    iter +=  1
  
  print("count: ")
  print(count)
  CI = []
  count2 = {} #dictionary version of count 
  for i in range(len(count)):
      count2[i] = count[i]
  
  #get top 3 most frequently violated (no check for adjacency)              
  while(len(CI) < 3):
      if len(count2) != 0:
          ci = max(count2, key=count2.get)
          del count2[ci]
          if ci not in C:
              CI = CI + [ci]
  if len(CI) == 0: #if CI empty, no more constraints to add; done
    return
  #pdb.set_trace()
  C = C + CI
  return optimize(f, C, u)
  
for f in french:
  C = []  
  u = [0 for _ in f]                                                                                                           
  optimize(f, C, u)
  
  #if opts.verbose:
  #  def extract_tm_logprob(h):
  #    return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)

#for i in range(1,len(french)+1): #need to add 1 because 0-250 inclusive 
#  converged[i] = converged[i] + converged[i-1]
#print('converged:\n')
#print(converged)