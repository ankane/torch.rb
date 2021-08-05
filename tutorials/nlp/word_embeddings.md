# Word Embeddings: Encoding Lexical Semantics

Ported from [this tutorial](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html) by Robert Guthrie

Available under the same license ([BSD-3-Clause](LICENSE-nlp-tutorial.txt))

---

Word embeddings are dense vectors of real numbers, one per word in your vocabulary. In NLP, it is almost always the case that your features are words! But how should you represent a word in a computer? You could store its ascii character representation, but that only tells you what the word *is*, it doesn’t say much about what it *means* (you might be able to derive its part of speech from its affixes, or properties from its capitalization, but not much). Even more, in what sense could you combine these representations? We often want dense outputs from our neural networks, where the inputs are :math:`|V|` dimensional, where :math:`V` is our vocabulary, but often the outputs are only a few dimensional (if we are only predicting a handful of labels, for instance). How do we get from a massive dimensional space to a smaller dimensional space?

How about instead of ascii representations, we use a one-hot encoding? That is, we represent the word :math:`w` by

.. math::  \overbrace{\left[ 0, 0, \dots, 1, \dots, 0, 0 \right]}^\text{|V| elements}

where the 1 is in a location unique to :math:`w`. Any other word will have a 1 in some other location, and a 0 everywhere else.

There is an enormous drawback to this representation, besides just how huge it is. It basically treats all words as independent entities with no relation to each other. What we really want is some notion of *similarity* between words. Why? Let’s see an example.

Suppose we are building a language model. Suppose we have seen the sentences

* The mathematician ran to the store.
* The physicist ran to the store.
* The mathematician solved the open problem.

in our training data. Now suppose we get a new sentence never before seen in our training data:

* The physicist solved the open problem.

Our language model might do OK on this sentence, but wouldn’t it be much
better if we could use the following two facts:

* We have seen  mathematician and physicist in the same role in a sentence. Somehow they have a semantic relation.
* We have seen mathematician in the same role  in this new unseen sentence as we are now seeing physicist.

and then infer that physicist is actually a good fit in the new unseen sentence? This is what we mean by a notion of similarity: we mean *semantic similarity*, not simply having similar orthographic representations. It is a technique to combat the sparsity of linguistic data, by connecting the dots between what we have seen and what we haven’t. This example of course relies on a fundamental linguistic assumption: that words appearing in similar contexts are related to each other semantically. This is called the [distributional hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics).

## Getting Dense Word Embeddings

How can we solve this problem? That is, how could we actually encode semantic similarity in words? Maybe we think up some semantic attributes. For example, we see that both mathematicians and physicists can run, so maybe we give these words a high score for the "is able to run" semantic attribute. Think of some other attributes, and imagine what you might score some common words on those attributes.

If each attribute is a dimension, then we might give each word a vector, like this:

.. math::

    q_\text{mathematician} = \left[ \overbrace{2.3}^\text{can run},
   \overbrace{9.4}^\text{likes coffee}, \overbrace{-5.5}^\text{majored in Physics}, \dots \right]

.. math::

    q_\text{physicist} = \left[ \overbrace{2.5}^\text{can run},
   \overbrace{9.1}^\text{likes coffee}, \overbrace{6.4}^\text{majored in Physics}, \dots \right]

Then we can get a measure of similarity between these words by doing:

.. math::  \text{Similarity}(\text{physicist}, \text{mathematician}) = q_\text{physicist} \cdot q_\text{mathematician}

Although it is more common to normalize by the lengths:

.. math::

    \text{Similarity}(\text{physicist}, \text{mathematician}) = \frac{q_\text{physicist} \cdot q_\text{mathematician}}
   {\| q_\text{physicist} \| \| q_\text{mathematician} \|} = \cos (\phi)

Where :math:`\phi` is the angle between the two vectors. That way, extremely similar words (words whose embeddings point in the same direction) will have similarity 1. Extremely dissimilar words should have similarity -1.

You can think of the sparse one-hot vectors from the beginning of this section as a special case of these new vectors we have defined, where each word basically has similarity 0, and we gave each word some unique semantic attribute. These new vectors are *dense*, which is to say their entries are (typically) non-zero.

But these new vectors are a big pain: you could think of thousands of different semantic attributes that might be relevant to determining similarity, and how on earth would you set the values of the different attributes? Central to the idea of deep learning is that the neural network learns representations of the features, rather than requiring the programmer to design them herself. So why not just let the word embeddings be parameters in our model, and then be updated during training? This is exactly what we will do. We will have some *latent semantic attributes* that the network can, in principle, learn. Note that the word embeddings will probably not be interpretable. That is, although with our hand-crafted vectors above we can see that mathematicians and physicists are similar in that they both like coffee, if we allow a neural network to learn the embeddings and see that both mathematicians and physicists have a large value in the second dimension, it is not clear what that means. They are similar in some latent semantic dimension, but this probably has no interpretation to us.

In summary, **word embeddings are a representation of the *semantics* of a word, efficiently encoding semantic information that might be relevant to the task at hand**. You can embed other things too: part of speech tags, parse trees, anything! The idea of feature embeddings is central to the field.


## Word Embeddings in Torch.rb

Before we get to a worked example and an exercise, a few quick notes about how to use embeddings in Torch.rb and in deep learning programming in general. Similar to how we defined a unique index for each word when making one-hot vectors, we also need to define an index for each word when using embeddings. These will be keys into a lookup table. That is, embeddings are stored as a :math:`|V| \times D` matrix, where :math:`D` is the dimensionality of the embeddings, such that the word assigned index :math:`i` has its embedding stored in the :math:`i`'th row of the matrix. In all of my code, the mapping from words to indices is a dictionary named word\_to\_ix.

The module that allows you to use embeddings is Torch::NN::Embedding, which takes two arguments: the vocabulary size, and the dimensionality of the embeddings.

To index into this table, you must use Torch::LongTensor (since the indices are integers, not floats).

```ruby
# Author: Robert Guthrie

require "torch"

Torch.manual_seed(1)
```

```ruby
word_to_ix = {"hello" => 0, "world" => 1}
embeds = Torch::NN::Embedding.new(2, 5) # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = Torch.tensor([word_to_ix["hello"]], dtype: :long)
hello_embed = embeds.call(lookup_tensor)
p hello_embed
```

Out:

```text
tensor([[ 0.6614,  0.2669,  0.0617,  0.6213, -0.4519]], requires_grad: true)
```

## An Example: N-Gram Language Modeling

Recall that in an n-gram language model, given a sequence of words :math:`w`, we want to compute

.. math::  P(w_i | w_{i-1}, w_{i-2}, \dots, w_{i-n+1} )

Where :math:`w_i` is the ith word of the sequence.

In this example, we will compute the loss function on some training examples and update the parameters with backpropagation.

```ruby
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = "When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.".split

# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = (test_sentence.length - 2).times.map { |i| test_sentence[i, 3] }
# print the first 3, just so you can see what they look like
p trigrams.first(3)

vocab = test_sentence.to_set
word_to_ix = vocab.map.with_index.to_h

class NGramLanguageModeler < Torch::NN::Module
  def initialize(vocab_size, embedding_dim, context_size)
    super()
    @embeddings = Torch::NN::Embedding.new(vocab_size, embedding_dim)
    @linear1 = Torch::NN::Linear.new(context_size * embedding_dim, 128)
    @linear2 = Torch::NN::Linear.new(128, vocab_size)
  end

  def forward(inputs)
    embeds = @embeddings.call(inputs).view(1, -1)
    out = Torch::NN::F.relu(@linear1.call(embeds))
    out = @linear2.call(out)
    Torch::NN::F.log_softmax(out, 1)
  end
end

losses = []
loss_function = Torch::NN::NLLLoss.new
model = NGramLanguageModeler.new(vocab.size, EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = Torch::Optim::SGD.new(model.parameters, lr: 0.001)

10.times do |epoch|
  total_loss = 0
  trigrams.each do |trigram|
    context = trigram.first(2)
    target = trigram.last

    # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
    # into integer indices and wrap them in tensors)
    context_idxs = Torch.tensor(context.map { |w| word_to_ix[w] }, dtype: :long)

    # Step 2. Recall that torch *accumulates* gradients. Before passing in a
    # new instance, you need to zero out the gradients from the old
    # instance
    model.zero_grad

    # Step 3. Run the forward pass, getting log probabilities over next
    # words
    log_probs = model.call(context_idxs)

    # Step 4. Compute your loss function. (Again, Torch wants the target
    # word wrapped in a tensor)
    loss = loss_function.call(log_probs, Torch.tensor([word_to_ix[target]], dtype: :long))

    # Step 5. Do the backward pass and update the gradient
    loss.backward
    optimizer.step

    # Get the Ruby number from a 1-element Tensor by calling tensor.item
    total_loss += loss.item
  end

  losses << total_loss
end
p losses  # The loss decreased every iteration over the training data!

# To get the embedding of a particular word, e.g. "beauty"
p model.embeddings.weight[word_to_ix["beauty"]]
```

Out:

```text
[["When", "forty", "winters"], ["forty", "winters", "shall"], ["winters", "shall", "besiege"]]
[519.774599313736, 517.382593870163, 515.0068221092224, 512.6433944702148, 510.2921438217163, 507.95162177085876, 505.62235164642334, 503.3039937019348, 500.9952976703644, 498.6960301399231]
tensor([ 0.8210, -0.1641, -0.8544, -0.7513, -0.0947, -2.3143,  0.5580,  0.5047,
        -0.0697, -1.0458], requires_grad: true)
```

## Exercise: Computing Word Embeddings: Continuous Bag-of-Words

The Continuous Bag-of-Words model (CBOW) is frequently used in NLP deep learning. It is a model that tries to predict words given the context of a few words before and a few words after the target word. This is distinct from language modeling, since CBOW is not sequential and does not have to be probabilistic. Typically, CBOW is used to quickly train word embeddings, and these embeddings are used to initialize the embeddings of some more complicated model. Usually, this is referred to as *pretraining embeddings*. It almost always helps performance a couple of percent.

The CBOW model is as follows. Given a target word :math:`w_i` and an :math:`N` context window on each side, :math:`w_{i-1}, \dots, w_{i-N}` and :math:`w_{i+1}, \dots, w_{i+N}`, referring to all context words collectively as :math:`C`, CBOW tries to minimize

.. math::  -\log p(w_i | C) = -\log \text{Softmax}(A(\sum_{w \in C} q_w) + b)

where :math:`q_w` is the embedding for word :math:`w`.

Implement this model in Torch.rb by filling in the class below. Some tips:

* Think about which parameters you need to define.
* Make sure you know what shape each operation expects. Use .view() if you need to reshape.

```ruby
CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = "We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.".split

# By deriving a set from `raw_text`, we deduplicate the array
vocab = Set.new(raw_text)
vocab_size = vocab.size

word_to_ix = vocab.each.with_index.to_h { |word, i| [word, i] }
data = []
2.upto(raw_text.length - 2) do |i|
  context = [raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2]]
  target = raw_text[i]
  data << [context, target]
end
p data.first(5)

class CBOW < Torch::NN::Module
  def initialize
    super()
  end

  def forward(inputs)
  end
end

# create your model and train.  here are some functions to help you make
# the data ready for use by your module

def make_context_vector(context, word_to_ix)
  idxs = context.map { |w| word_to_ix[w] }
  Torch.tensor(idxs, dtype: :long)
end

make_context_vector(data[0][0], word_to_ix) # example
```
