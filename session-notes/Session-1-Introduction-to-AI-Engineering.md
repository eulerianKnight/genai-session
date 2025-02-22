# Session 1: Introduction to AI Engineering

## What's Past is Prologue.

### Prediction and Entropy of Printed English C.E. Shannon January,1951




## Transformers

Transformers contains of 3 main part:

- Word Embedding: Word embedding converts words, bits of words and symbols, collectively called Tokens, into numbers.
- Positional Encoding: Helps keep track of word order.
- Attention: It tries to establish relationship among words.

### Encoder

#### Input Embedding

![input-embedding.jpg](../assets/input-embedding.jpg)

#### Positional Encoding

- Each word should carry some information about its position in the sentence.
- The model should treat words that appear close to each other as "close" and words that are distant as "distant".
- We want positional encoding to represent a pattern that can be learned by the model.

![positional-encoding.jpg](../assets/positional-encoding.jpg)

$PE(pos, 2i) = sin\frac{pos}{10000^{\frac{2i}{d_{model}}}}$

$PE(pos, 2i + 1) = cos\frac{pos}{10000^{\frac{2i}{d_{model}}}}$

![positional_encoding.png](../assets/positional_encoding.png)

- Trigonometric functions like **cos** and **sin** naturally represent a pattern that the model can recognize as 
continuous, so relative positions are easier to see for the model.

### Self-Attention

$Attention(Q, K, V) = Softmax(\frac{(QK^T)}{\sqrt(d_k)}V$

![self-attention.jpg](../assets/self-attention.jpg)

- Self-Attention allows the model to relate words to each other.
- Each row in the final Attention matrix not only captures meaning(provided by the embedding) or the position in the 
sentence (provided by the positional encodings) but also each word's interaction with other words.
- Self-Attention is permutation invariant.
- Self-Attention requires no parameters. Up to now the interaction between words has been driven by their embedding and the positional encodings.
- Expectation is that the values along the diagonal to be the highest.


### Multi-Head Attention

$MultiHead(Q, K, V) = Concat(head_1....head_2)W^{O}$
$head_i=Attention(QW_{i}^{Q},KW_{i}^{K},VW_{i}^{V})$

$Where, W_{i}^{Q} \in R^{d_{model}\times{d_{k}}}, W_{i}^{K} \in R^{d_{model}\times{d_{k}}}, W_{i}^{V} \in R^{d_{model}\times{d_{k}}}, W_{i}^{O} \in R^{hd_{v}\times{d_{model}}}$

![multi-head-attention.jpg](../assets/multi-head-attention.jpg)