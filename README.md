java c
Quiz   1
CSCI-567 –   Machine   Learning
Fall   2024
1.      Assume   a   linear   regression   model   parameterized   by   θ   .    Denote   the   linear   model   by   fθ (x(i))   =   θ   Tx(i)   .   Which   of the   following   represents   the   gradient   of the   loss   function   L(θ)   in   gradient   descent   for   linear   regression?


2.    Given   a   linear   classifier   f(x)   = wTx   + b,   which   of   the   following   conditions   must   hold   for   a   dataset   to be   linearly   separable?
a)      3w,b   such   that   y(i)(wTx(i)    + b)   > 0   for   all   i
b)      3w,b   such   that   y(i)(wTx(i)   + b) =   0   for   all   i
c)   丫w,b,   y   (i)(wTx(i)   + b) =   1   for   all   i
d)   丫w,b,   y   (i)(wTx(i)   + b)   < 0   for   all   i
3.   Which   of   the   following   optimization   problems   represents   the   regularized   binary   logistic   regression   objective with   ℓ2-regularization?


4.   Which   of   the   following   is   an   example   of   a   convex   surrogate   loss   function?
a)    Hinge   loss
b)    Squared   loss
c)    Cross-entropy   loss
d)      All   of the   above
5.   In   gradient   descent,   the   update   rule   is   given   by   θ   (t+1)      =   θ   (t)    −   η∇J(θ(t)).    What   condition   must   be true   for   the   algorithm   to   converge?
a)    η   must   be   very   large   to   speed   up   convergence
b)      ∇J(θ(t))   must   be   negative
c)    η   must   change   at   every   step
d)    η   must   be   small   enough   to   ensure   the   loss   function   decreases   at   each   step
6.   Which   of   the   following   expressions   represents   the   gradient   of   the   logistic   regression   loss   function   L(θ),   where   fθ 1+e−θT x/1?

7.   Which   of   the   following   surrogate   loss   functions   is   convex   and   differentiable,   and   used   in   binary   logistic regression?
a)      L(θ) = max(0,   1 −   yfθ (x))
b)      L(θ = log(1 + e−yfθ (x))
c)      L(θ) =   (y −   fθ   (x))2
d)      L(θ =   |y   −   fθ (x)|
8.   What   is   a   surrogate   loss   function   used   for?
a)      To   approximate   the   true   error   function   in   a   more   tractable   form.
b)    To   reduce   the   number   of   features   in   the   model
c)      To   avoid   overfitting   by   adding   a   regularization   term
d)      To   directly   minimize   the   classification   accuracy
9.   Which   of   data   below   is   not   linearly   separable?    (Hint:   consider   visualizing   the   data)
a)    Class   1:    [(-1,0),    (-1,2)],   Class   2:    [(1,2),   (1,0)]
b)    Class   1:    [(-1,0),   (1,0)],   Class   2:    [(0,1),   (0,-1)]
c)    Class   1:    [(-1,0,0),   (-1,2,0)],   Class   2:    [(1,2,0),    (1,0,0)]
d)    Class   1:    [(1,1,2),   (2,1,1),   (1,2,1)],   Class   2:    [(0,-1,-1),   (-2,0,-1),   (-1,-2,0)]
10.    Given   a   dataset   matrix   X   ∈   Rm   ×n      where   m   < n,   and   each   column   of   X   represents   a   data   sample.    Let C   be   the   covariance   matrix   of   the   dataset,   and   assume   the   mean   vector   of   the   samples   µ   ∈   Rm      is   a   zero   vector.   Which   of the   statements   below   is   definitely   correct?
(a)      C   ∈   Rn   ×n
(b)      rank(C) = rank(X)
(c)    C   does   not   have   any   negative   elements.
(d)      C   is   a   positive   definite   matrix.
11.   Which   of   the   followings   about   neural   networks   is   wrong?
a)      The   attention   layer   in   neural   networks   typically   uses   the   ReLU   activation   function.
b)    Convolutional   Neural   Networks   (CNNs)   are   inherently   designed   to   be   rotationally   invariant.
c)    Dropout   techniques   are   employed   to   enhance   the   generalizability   of   the   model.
d)    Transformers   can   also   be   effectively   trained   to   classify   images.
12.   Which   of   the   following   statements   is   wrong?
a)    Overfitting occurs when   a   machine   learning   model   learns to   capture   noise or   random   fluctuations   in   the   training   data,   resulting   in   poor   generalization.
b)      Regularization   techniques   can   help   prevent   overfitting.
c)      Both   ℓ   1    and   ℓ2    regularization   can   help   in   reducing   the   variance   of   a   model   by   penalizing   overly complex   models.
d)    CNNs   do   not   work   well   on   linearly   non-separable   datasets.
13.      Regarding   activation   functions   in   neural   networks,   which   statement   is   true?
a)      The   sigmoid   activation   function   is   preferred   over   ReLU   in   deep   neural   networks   to   avoid   vanishing gradient   problems.
b)      The   tanh   activation   function   outputs   values   in   the   range   [0   , 1].
c)      The   ReLU   activation   function   is   defined   as   f(x) = max(0,   x2   ).
d)      The softmax activation function is commonly used in the output   layer   for   multi-class   classification   problems.
14.      Regarding   convolutional   neural   networks   (CNNs),   which   of   the   following   statements   is   false?
a)      Pooling   layers   in   CNNs   help   to   reduce   the   dimensionality   and   retain   important   features   of   the   input.
b)      The   stride   in   a   convolutional   layer   determines the   step   size   by which the   convolution   filter   moves   across   the   input.
c)   Increasing   the   number   of filters   in   a   convolutional   layer   always   reduces   overfitting.
d)      Padding   is   used   in   convolutional   layers   to   control   the   spatial   size   of the   output   feature   maps.
15.      Consider aneural network with a ReLU   activation   function,   f(x)   =   max(0,   x).   Let   z   =   wtx+bbe   the   linear   transformation   of the   input   x with weight   matrix   W   and   bias   vector   b.    What   is   the   gradient   of   f(z)   with   respect   to   x   when   z   < 0?
a)      w
b)      b
c)      0
d)      diag(z)W
16.   Which   of   the   following   statements   about   the   transformer   architecture   is   false?
a)    Transformers use self-attention mechanisms to process input sequences in parallel, unlike recurrent   neural   networks   (RNNs)   which   process   inputs   sequentially.
b)      The   positional   encoding   in   transformers   allows   the   model   to   understand   the   order   of elements   in   the   input   sequence   without   using   recurrence.
c)    Transformer   models   always   require   both   encoder   and   decoder   components   to   function   effectively for   any   task.
d)      The   multi-head   attention   mechanism   in   transformers   allows   the   model   to   focus   on   different   aspects of the   input   simultaneously.
17.      For   a   neural   network   with   a   hidden   layer   size   of   128,   if   the   ReLU   activation   is   used,   which   of   the following   statements   is   correct?
a)      The   output   of   the   hidden   layer   will   be   bounded   between   0   and   1.
b)      The   gradient   of the   activation   function   can   never   be   zero.
c)    Some   units   may   output   zero   if   the   input   is   negative.
d)      The   ReLU   activation   is   linear   for   all   inputs.
18.   In   the    context   of   stochastic   gradient    descent      (SGD),    which      of   the      following   factors   most    directly influences   convergence   speed?
a)    Number   of   epochs
b)      Learning   rate
c)    Batch   size
d)      Number   of   features
19.   Which   of the   following   is   the   purpose   of multi-head   attention   in   transformers?
a)    To   reduce   the   computational   complexity   of attention   mechanisms.
b)      To   ensure   that   attention   focuses   on   a   single   position   in   the   input.
c)    To   allow   the   model   to   focus   on   different   parts   of the   input   simultaneously.
d)    To   generate   more   robust   positional   encodings.
20.      Consider   the   function   f(x,y) = x2   −y2   .   Which   of   the   following   statements   about   the   function’s   critical points   is   correct?
a)      The   point   (0,   0)   is   a   saddle   point
b)      The   point   (0,   0)   is   a   global   minimum.
c)      The   point   (0,   0)   is   a   local   minimum.
d)      The   function   has   no   critical   points.
21.   Which   of   the   following   is   not   a   true   statement   about   gradient   descent   (GD)   vs.    stochastic   gradient   descent   (SGD)?
a)    Both   provide   unbiased   estimates   of   the   true   gradient   at   each   step.
b)    The   memory   and   compute   requirements   of   a   single   update   step   for   both   methods   scales   linearly with   the   number   of features.
c)    The   memory   and   compute   requirements   of   a   single   update   step   for   both   methods   scales   linearly with   the   number   of data   points.
d)    GD   is   likely   to   converge   in   fewer   updates/iterations   than   SGD,   with   a   properly   selected   learning   rate.


22.   Write   down   a   closed   form   solution   for   the   optimal   parameters   θ   that   minimize   the   loss   function

in   terms   of   the   N   × d   matrix   X   whose   i-th   row   is   xi(T)   and   the   N   × 1   vector   y   whose   i-th   entry   is   yi. You   may   assume   that   any   relevant   matrix   is   invertible.
a)      w*   = 2(XTX)−1X   Ty
b)      w*   = (XTX)−1X   Ty
c)      w*   = (XTX)−1Xy
d)      w*   = (XXT   )   −1X   Ty
23.   Which   statement   is   true?
(a)      Logistic   regression   is   not   a   probabilistic   model.
(b)    Linear   regression   is   best   used   for   classification.
(c)      Logistic   regression   works   well   for   non-linearly   separable   data.
(d)   We   can   use   SGD   to   learn   both   logistic   regression   and   linear   regression   models.
24.    Consider   a   convolution   layer   with   an   input   tensor   of   dimension   8   ×   11   × 3   and   an   output   tensor   of dimension   3   × 4   × 3   tensor.   What   is   the   correct   hyperparameter   configuration   of   this   layer?
(a)    Six   4   × 4   × 4   filters,   the   stride   is   three,   1   zero-padding
(b)    Six   2   × 2   × 3   filters,   the   stri代 写CSCI-567 – Machine Learning Quiz 1 Fall 2024Java
代做程序编程语言de   is   three,   2   zero-padding
(c)    Three   4   × 4   × 3   filters,   the   stride   is   three,   1   zero-padding
(d)    Three   3   × 4   × 3   filters,   the   stride   is   three,   1   zero-padding
25.   Which   statement   is   false?
(a)    CNN   can   be   used   for   multiclass   classification.
(b)    Feedforward   neural   networks   can   be   used   to   model   non-linear   datasets.
(c)    Logistic   regression   is   a   linear   model.
(d)      Transformer   is   a   linear   model.
26.   Which   one   is   an   incorrect   characterization   of   overfitting?
(a)   Increasing   data   size   reduces   overfitting.
(b)      Projecting   the   model   to   a   more   complex   feature   space   will   avoid   overfitting.
(c)    Overfitting   can   be   observed   from   high   train   accuracy   and   low   test   accuracy.
(d)      Regularization   can   help   reduce   overfitting.
27.   Which   one   is   not   an   activation   function?
(a)      ReLU
(b)    GeLU
(c)   Indicator   function
(d)    Sigmoid
28.   Which   one   is   an   incorrect   description   of   a   transformer?
(a)   It   is   often   used   with   position   embeddings.
(b)      A   transformer   block   consists   of a   linear   self-attention   layer   and   a   feedforward   network.
(c)   It   is   end-to-end   differentiable.
(d)   It   has   a   recurrent   layer.
29.   When   training   on   an   imbalanced   dataset   where   the   dataset   contains   more   data   with   the   first   label   than   the   three   other   classes   (four   class   in   total),   which   one   of   the   following   statements   is   true?
(a)      The   model   will   overfit   to   the   first   class.
(b)    The   model   will   overfit   to   the   second   class.
(c)      The   model   will   overfit   to   the   third   class.
(d)      The   model   will   overfit   to   the   fourth   class.
30.    Let   F   =   {f(x) = sign(wTx   +   b)|w   ∈ R2   ,   x   ∈ R2   ,   b   ∈ R} be   the   set   of   binary   classifiers   in   2   dimensional space.      Given   there   are   only   two   possible   options   for   the   data   label   of   each   sample   (y   ∈   {−1, 1}),   if   you   have   a   free   choice   in   selecting   the   training   data,   what   is   the   biggest   possible   number   of   training samples, that the   2 dimensional binary   classifier   can   correctly   classify no   matter   how   the   training   data   is   labeled?
a)    3
b)   4
c)    5
d)      ∞
31.   Which   of   the   following   loss   functions   might   still   incur   a   loss   penalty      (non-zero      loss)    even   though sign(wTx   + b)   and   the   ground   truth   label   have   the   same   value?
a)    Perceptron   loss
b)    Hinge   loss
c)    Logistic   loss
d)      Both   b   and   c
32.   Which   of   the   following   statement   is   true   about   Perceptron?
a)    Perceptron   always   converges   in   a   finite   number   of steps   for   any   dataset.
b)    The   update   rule   is   affected   by   all   samples.
c)      The   choice   for   the   learning   rate   significantly   affects   the   prediction   of weights.
d)    The   update   rule   can   be   performed   on   more   than   one   sample   at   a   time.
33.   What   is   the   derivative   of   the   function   f(x)   = lnσ(wTx+b)   with   respect   to   the   parameter   b   ?    (σ(x)   =   1+e−x/1)

34.      Figure 1,   shows   the   change   in   the   loss   function   of   a   model,   during   the   training   process.    What   is   the primary   reason   for   the   way   the   loss   function   changes?

Figure   1:   The   change   in   the   loss   function   of   a   model,   during   the   training   process.
a)      Bad   initialization
b)      Low   learning   rate
c)      High   learning   rate
d)      High   batch   size
35.      During   the   training   of   a   model   we   notice   a   growing   gap   between   the   training   and   validation   perfor-   mances.   What   would   be   the   best   approach   to   solve   this   problem?
a)   Increase   regularization   strength   and   decrease   training   data
b)   Increase   regularization   strength   and   increase   training   data
c)    Decrease   regularization   strength   and   decrease   training   data
d)    Decrease   regularization   strength   and   increase   training   data
36.    One   layer   of   the   convolution   neural   network   described   below   is   wrong.    If   the   input   data   is   224   × 224   colored   images   and   the   size   of the   flattened   output   is   8700,   which   layer   has   the   wrong   specifications?   (For   each   Conv   layer   the   provided   specifications   are   mask   size,   output   depth,   stride   size,   and   padding size.   Same   is   true   for   Max   Pool   layers   except   there   is   no   output   depth.)
a)    Conv   4x4,   128   (s:2,   p:1)
b)    Max   Pool   2x2   (s:2,   p:0)
c)    Conv   5x5,   256   (s:2,   p:1)
d)    Conv   7x7,   324   (s:1,   p:3)
Convolution         Neural      Network         Layers :
Conv      3x3 ,         96         ( s   :1   ,       p :1)
Conv      4x4 ,         128         ( s   :2   ,       p :1)
Conv      5x5 ,         168         ( s   :1   ,       p :2)
Max      Pool      2x2       ( s :2   ,         p :0)
Conv      3x3 ,         212         ( s   :1   ,       p :1)
Conv      5x5 ,         256         ( s   :2   ,       p :1)
Max      Pool      2x2       ( s :2   ,         p :0)
Conv      7x7 ,         324         ( s   :1   ,       p :3)
Conv      2x2 ,         348         ( s   :1   ,       p :0)
Max      Pool      5x5       ( s :2   ,         p :0)
37.   Which   part   of the   attention   layer   best   represents   the   distribution   of the   relationship   between   tokens?

38.   Which   activation   function   is   defined   as f(x) = ex + e−x/ex − e−x and   outputs   values   in   the   range   
(a)    Sigmoid   function
(b)      ReLU   function
(c)      Hyperbolic   tangent   (tanh)   function
(d)    Softmax   function
39.   In   a   convolutional   neural   network   (CNN),   which   of   the   following   statements   about   the   convolution   operation   is   true?
(a)      The   convolution   operation   reduces   the   spatial   dimensions   of the   input.
(b)    The   convolution   operation   uses   the   same   weights   (filters)   across   different   spatial   positions.
(c)      The   convolution   operation   increases   the   number   of   channels   in   the   input.
(d)    The   convolution   operation   always   uses   a   stride   of   1   and   no   padding.
40.   Which   of   the   following   statements   about   the   attention   mechanism   in   transformers   is   false?
(a)    Attention   allows   the   model   to   focus   on   specific   parts   of   the   input   sequence   when   generating   each part   of the   output   sequence.
(b)    The   attention   scores   are   computed   using   dot   products   between   query   and   key   vectors.
(c)    The   value   vectors   are   used   to   compute   the   attention   scores.
(d)      The   scaled   dot-product   attention   includes   a   scaling   factor   to   prevent   softmax   saturation.
41.   In   the   context   of   multi-class   classification, which   loss   function   is   commonly   used   when   training   a   neural network?
(a)    Mean   squared   error   (MSE)
(b)    Hinge   loss
(c)    Cross-entropy   loss
(d)      Absolute   error   loss
42.    Consider   a   linear   model   with   100 input   features, out   of   which   10 are   highly   informative   about   the   label and   90   are   non-informative   about   the   label.    Assume   that   all   features   have   values   between   −1   and   1.   Which   of the   following   statements   are   true?
(a)   ℓ   1    regularization   will   encourage   most   of   the   non-informative   weights   to   be   exactly   0   .0.
(b)   ℓ   1   regularization   will   encourage   most   of   the   non-informative   weights   to   be   nearly   (but   not   exactly)   0.0.
(c)   ℓ2    regularization   will   encourage   most   of   the   non-informative   weights   to   be   exactly   0   .0.
(d)   ℓ2   regularization   will   encourage   most   of   the   non-informative   weights   to   be   nearly   (but   not   exactly)   0.0.
43.   Which   of   the   following   options   will   decrease   the   generalization   gap   (difference   between   test   error   and   training   error)   of a   machine   learning   model?
(a)    Use   more   data   to   learn   the   model.
(b)      Add   l2    regularization   on   the   parameters   when   learning   the   model.
(c)    Consider   a   more   complex   model   class,   which   is   a   superset   of the   original   function   class.
(d)    Simplify   the   model   by   reducing   its   complexity.
44.   Which   of   the   following   statements   about   supervised   learning   are   true?
(a)    The   test   set   should   not   be   used   to   train   the   model,   but   can   be   used   to   tune   hyperparameters.
(b)    The   generalization   gap    (difference   between   test   and   training   errors)   generally   decreases   as   the   size   of the   training   set   increases.
(c)   We   cannot estimate the risk of   a   predictor   (its   average   error   on   the   data   distribution)   solely   with   the   data   used   to   train   it.
(d)   If training and test data are   drawn   from   different   distributions,   then   low   error   on   the   training   set   may   not guarantee   low error on the   test   set   even   if the   size   of the   training   set   is   sufficiently   large.
45.   In   the   context   of   gradient   descent   optimization,   what   is   the   primary   advantage   of   using   mini-batch gradient   descent   over   batch   gradient   descent?
(a)   It   always   converges   to   the   global   minimum.
(b)   It   reduces   the   variance   of the   parameter   updates,   leading   to   more   stable   convergence.
(c)   It   computes   the   gradient   using   the   entire   training   dataset,   making   it   more   accurate.
(d)   It   requires   less   memory   and   allows   for   faster   computation   per   update.





         
加QQ：99515681  WX：codinghelp  Email: 99515681@qq.com
