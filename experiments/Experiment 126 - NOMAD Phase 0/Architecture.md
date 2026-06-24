Yes. If we design from scratch, the clean target is **not** “a Transformer without backprop.” It is a new kind of system:

[ iterative reasoner + attention-as-associative-memory + external reconstructive memory + local no-backprop learning ]

Let’s call it for now:

[ NOMAD: No-backprop Online Memory-Attention Dynamics ]

The job of NOMAD is to keep the model small, let memory expand externally, and train using **local / recursive / projection / perturbation / alignment rules**, not backprop.

---

# 1. What does an AI need?

At minimum:

```text
1. Representation
   tokens/bytes → vectors

2. Attention / retrieval
   find relevant internal and external information

3. Working memory
   keep current context active

4. Long-term memory
   store documents, examples, traces, user facts, code

5. Reasoning dynamics
   iteratively settle toward a stable answer

6. Learning rule
   update parameters during training

7. Output head
   convert hidden state to next-token distribution

8. Self-correction / confidence
   know when memory conflicts, when state is unstable, when to continue
```

For your constraints, the key rule is:

[ the model should not memorize all knowledge in weights ]

Instead:

[ weights learn how to think/use memory; memory stores experience ]

That means intelligence and memory are decoupled.

---

# 2. Math sources worth stealing from, 1930s–2026

Not all of this was originally “neural net math.” That is the point. A lot of the good stuff came from numerical analysis, adaptive filters, control, and neuroscience.

## 2.1 Kaczmarz projection, 1937

Kaczmarz is a row-action method for solving linear systems. Modern surveys describe it as an iterative method where each update uses one row/equation at a time, which makes each step cheap. ([arXiv][1])

For equation:

    a_i^T * w = b_i

the Kaczmarz update is:

    w <- w + ((b_i - a_i^T * w) / (||a_i||^2 + epsilon)) * a_i

This is underrated for no-backprop because each neuron can be treated as a small equation:

    w_j^T * h = t_j

Then update that neuron locally:

    w_j <- w_j + ((t_j - w_j^T * h) / (||h||^2 + epsilon)) * h

No backward graph. No chain rule. Just “project this neuron closer to its local target.”

This is a huge candidate.

---

## 2.2 Robbins–Monro stochastic approximation, 1951

Robbins and Monro introduced stochastic approximation as a method for finding a root when only noisy measurements are available. ([Project Euclid][2])

Generic form:

    theta_(n+1) = theta_n - a_n * [N(theta_n) - alpha]

For us:

    theta_(n+1) = theta_n + a_n * (local noisy improvement signal)

This is the mathematical ancestor of many online learning rules.

---

## 2.3 Hebb, 1949; Oja and BCM, 1982

Hebb’s 1949 work gave the basic “co-activation strengthens connection” idea. ([MPG.PuRe][3])

Basic Hebb:

    Delta W = eta * y * x^T

But pure Hebb explodes. Oja fixed that with a stabilizing normalization term and showed a simple neuron can extract the principal component of the input stream. ([David Kleinfeld Laboratory][4])

    Delta w = eta * y * (x - y * w)

BCM added a sliding threshold so synapses strengthen or weaken depending on postsynaptic activity relative to a moving threshold. ([Scholarpedia][5])

    Delta w = eta * y * (y - theta) * y_pre

Better notation:

    Delta w = eta * phi(y, theta) * y_pre

where theta tracks recent activity.

Use this for unsupervised feature formation inside the model.

---

## 2.4 LMS / ADALINE and RLS / Kalman, 1960s

Widrow-Hoff LMS appeared in the early ADALINE era and uses a simple local error times input update. ([Information Systems Laboratory][6])

    w <- w + eta * (y - y_hat) * x

Recursive least squares and Kalman filtering are the stronger cousins: maintain uncertainty about weights/states and update recursively as data arrives. Kalman filtering is described as an efficient recursive estimator of internal state from noisy measurements. ([Wikipedia][7])

RLS update:

    K_t = (P_(t-1) * x_t) / (lambda + x_t^T * P_(t-1) * x_t)

    e_t = y_t - W_(t-1) * x_t

    W_t = W_(t-1) + e_t * K_t^T

    P_t = lambda^(-1) * [ P_(t-1) - K_t * x_t^T * P_(t-1) ]

This is extremely interesting for the output head, adapters, and memory gates. It is no-backprop and online.

---

## 2.5 Hopfield energy, 1982

Hopfield networks showed how a recurrent system can act as content-addressable memory with attractor states. ([PNAS][8])

Classic energy idea:

    E(s) = -0.5 * s^T * W * s + b^T * s

The network updates until energy decreases.

This matches your FPRM/fixed-point direction:

    h^(k+1) = F(h^k, x)

    r^k = ||h^(k+1) - h^k||

    halt when r^k < tau

Your current uploaded FPRM script already uses an adaptive fixed-point loop with residual halting, so this direction fits your existing codebase. 

---

## 2.6 Anderson acceleration and Broyden, 1965

Anderson acceleration was created for fixed-point iterations and later became widely used in scientific computing. ([SIAM][9])

Broyden’s method approximates Newton-like behavior with rank-one Jacobian updates instead of recomputing the full Jacobian. ([Department of Mathematics & Statistics][10])

For our architecture, these are not “training rules” first. They are **thinking-loop accelerators**.

Given fixed-point residual:

    f(h) = F(h, x) - h

Anderson uses recent residual history:

    h_(k+1) = Sum_(i=0..m) alpha_i * F(h_(k-i))

with:

    Sum_i alpha_i = 1

chosen to reduce residual.

This can make the model settle faster without backprop.

---

## 2.7 EM algorithm, 1977

The EM algorithm gives a way to learn when some variables are hidden: infer latent assignments, then update parameters. ([JSTOR][11])

Generic:

    q(z) <- p(z | x, theta)

    theta <- argmax_theta E_q(z) [ log p(x, z | theta) ]

For NOMAD, memory retrieval assignments are hidden variables:

    z = which memory chunk/trace caused this answer?

So memory learning can use EM-like assignment:

    q_i = p(c_i | x, y)

then update the memory router locally.

---

## 2.8 Reservoir computing / liquid state machines, 2001–2002

Liquid State Machines and Echo State Networks showed that a large fixed dynamical reservoir can compute useful features, while only a readout is trained. Liquid state machines emphasized real-time computation without stable states and relied on separation and approximation properties. ([PubMed][12])

Echo State Networks similarly use a fixed recurrent reservoir and train a readout, often with linear methods. ([Scholarpedia][13])

This is very compatible with no-backprop:

    h_t = phi(W_r * h_(t-1) + W_x * x_t)

Train only:

    W_o

using LMS/RLS/Kaczmarz.

For NOMAD, we do not want a purely random reservoir, but we can use this principle:

[ make the body partly stable/self-organizing; train cheap local adapters/readouts ]

---

## 2.9 Attention, modern Hopfield, fast weights

Transformers use attention as the core operator. The 2017 paper introduced an architecture based solely on attention mechanisms. ([arXiv][14])

Attention:

    Attn(Q, K, V) = softmax(Q * K^T / sqrt(d)) * V

Modern Hopfield work later showed that transformer attention can be interpreted as the update rule of a continuous Hopfield network. ([arXiv][15])

Linear attention has another useful interpretation: linear Transformers are formally equivalent to fast-weight programmers, where memory is updated by additive outer products. ([arXiv][16])

Fast-weight memory:

    A_t = lambda * A_(t-1) + eta * k_t * v_t^T

Read:

    r_t = A_t^T * q_t

This is perfect for no-backprop because writing to memory is local:

    Delta A_t = eta * k_t * v_t^T

No chain rule needed. Attention becomes dynamic memory, not just a layer.

---

## 2.10 Modern no-backprop alternatives

Feedback alignment showed that fixed random feedback weights can support learning in deep networks instead of exact backpropagated errors. ([arXiv][17])

Direct feedback alignment:

    delta_l = B_l * e

    Delta W_l = eta * delta_l * h_(l-1)^T

Equilibrium propagation uses two phases and nudges the output toward the target, letting error influence hidden states through the same dynamics used for inference. ([Frontiers][18])

Predictive coding networks have also been shown to approximate error backpropagation using local Hebbian-like plasticity. ([PubMed][19])

Hinton’s Forward-Forward algorithm replaces forward/backward passes with two forward passes, one positive and one negative, with layer-local objectives. ([arXiv][20])

Recent NoProp work proposes training without full forward or backward propagation by making layers learn local denoising targets. ([arXiv][21])

So the modern lesson is:

[ use local objectives, local targets, random feedback, and two-phase contrast ]

not global backprop.

---

# 3. New architecture: NOMAD

## 3.1 Main idea

[ NOMAD: No-backprop Online Memory-Attention Dynamics ]

It has five interacting systems:

```text
1. Local feature body
2. Fast attention memory
3. External reconstructive memory
4. Fixed-point reasoning loop
5. No-backprop learning engine
```

Diagram:

```text
tokens/bytes
   ↓
local encoder
   ↓
fast-weight attention memory  ←→ external memory bus
   ↓
fixed-point reasoning core
   ↓
readout head
   ↓
next-token distribution
   ↓
local no-backprop updates
```

---

# 4. Forward pass math

Input sequence:

    x_1, ..., x_T

Embedding:

    e_t = E[x_t] + p_t

Initial state:

    h_t^0 = e_t

---

## 4.1 Local recurrent / state-space body

Use a small gated recurrent state, inspired by state-space/recurrent models rather than a full Transformer. Modern selective SSMs such as Mamba were proposed to get linear-time sequence modeling while addressing content-based reasoning weaknesses in earlier subquadratic models. ([arXiv][22])

For our no-backprop architecture, use a simple input-conditioned recurrence:

    s_t = g_t * s_(t-1) + (1 - g_t) * psi(W_s * e_t + U_s * r_t)

Gate:

    g_t = sigma(W_g * e_t + U_g * h_(t-1))

Hidden:

    h_t = Norm(s_t + e_t + r_t + m_t)

where:

* r_t = fast internal memory read
* m_t = external memory read
* s_t = working state

---

## 4.2 Fast-weight attention memory

Instead of quadratic attention over all tokens, maintain a fast associative memory matrix:

    A_t in R^(d_k x d_v)

Keys, values, queries:

    k_t = K * h_t

    v_t = V * h_t

    q_t = Q * h_t

Write:

    A_t = lambda_A * A_(t-1) + eta_A * phi(k_t) * v_t^T

Read:

    r_t = (A_t^T * phi(q_t)) / (z_t^T * phi(q_t) + epsilon)

with normalizer:

    z_t = lambda_A * z_(t-1) + eta_A * phi(k_t)

This gives attention-like retrieval with local outer-product writes.

Important: this is not trained by backprop during the sequence. It is a fast memory update rule.

---

## 4.3 External reconstructive memory bus

Long-term memory:

    M = M_exact + M_compression + M_semantic + M_trace + M_kNN

Given state h_t and context x_<=t, score chunk c_i:

    R_i^t = lambda_e * R_exact(c_i, x) + lambda_g * R_gzip(c_i, x) + lambda_s * R_sem(c_i, x) + lambda_k * R_kNN(c_i, h_t) + lambda_r * R_trace(c_i, x)

Attention over top chunks:

    a_i^t = softmax_(i in TopK)(R_i^t)

Memory field:

    m_t = Sum_(i in TopK) a_i^t * V_M * c_i

New document insertion:

    M' = M cup Ingest(d)

No parameter update needed for adding knowledge.

---

## 4.4 Compression memory score

For candidate continuation y, compression gain:

    Delta C_0(y) = C(x_tail || y) - C(x_tail)

    Delta C_i(y) = C(c_i || x_tail || y) - C(c_i || x_tail)

    G_i(y | x) = Delta C_0(y) - Delta C_i(y)

If G_i(y | x) > 0, then memory chunk c_i helped predict y.

This keeps the GzipT insight but does not make gzip the whole memory system.

---

# 5. Fixed-point reasoning core

For each position or for the current generation step, run an inner reasoning loop:

    h^(n+1) = F_theta(h^n, e_t, r_t, m_t)

Use damping:

    h^(n+1) = (1 - eta_n) * h^n + eta_n * Phi_theta(h^n, e_t, r_t, m_t)

Residual:

    rho_n = ||h^(n+1) - h^n||

Halt when:

    rho_n < tau

for patience steps.

Optional Anderson acceleration:

    h_(n+1) = Sum_(j=0..m) alpha_j * Phi_theta(h_(n-j))

with:

    Sum_j alpha_j = 1

chosen from recent residuals.

This gives “thinking” without extending sequence length forever.

---

# 6. Output head

Logits:

    z_t = W_o * h_t + b_o + beta * b_M

where b_M is memory-derived token bias.

Probability:

    p_t = softmax(z_t)

Loss for supervised/next-token training:

    L_t = -log p_t[y_t]

But training will not backpropagate through the body.

---

# 7. No-backprop learning engine

This is the important part.

We train different parts with different no-backprop rules.

---

## 7.1 Output head: LMS / RLS

For target one-hot vector y_t and prediction p_t:

    e_t^out = y_t - p_t

Simple LMS:

    Delta W_o = eta_o * e_t^out * h_t^T

Better RLS:

    K_t = (P_(t-1) * h_t) / (lambda + h_t^T * P_(t-1) * h_t)

    W_(o,t) = W_(o,t-1) + e_t^out * K_t^T

    P_t = lambda^(-1) * [ P_(t-1) - K_t * h_t^T * P_(t-1) ]

This is local to the head. No backprop.

---

## 7.2 Body update: Direct feedback alignment

Use output error:

    e_t^out = y_t - p_t

Each layer l gets a fixed random feedback matrix:

    B_l

Layer teaching signal:

    delta_l = B_l * e_t^out

Local update:

    Delta W_l = eta_l * (delta_l * sigma'(a_l)) * h_(l-1)^T

No error is propagated layer-by-layer. Each layer receives a direct teaching signal.

If we use hard activations/ternary weights, replace sigma' with a local surrogate or just a clipped eligibility gate:

    g_l = 1(|a_l| < c)

    Delta W_l = eta_l * (delta_l * g_l) * h_(l-1)^T

---

## 7.3 Body update alternative: Kaczmarz target projection

Instead of gradients, define a local target activation:

    h_l_target = h_l + alpha * B_l * e_t^out

For neuron j:

    w_(l,j)^T * h_(l-1) ≈ h_l_target_j

Update:

    w_(l,j) <- w_(l,j) + eta * ((h_l_target_j - w_(l,j)^T * h_(l-1)) / (||h_(l-1)||^2 + epsilon)) * h_(l-1)

This may be the most elegant no-backprop body update.

Why? Because it says:

> “Given the local input, minimally change this neuron so its output moves toward a target.”

No chain rule. No graph. No backward pass.

This is likely worth making the main Exp127 learning rule.

---

## 7.4 Feature self-organization: Oja / BCM

For unsupervised stabilization:

    y_l = W_l * h_(l-1)

Oja:

    Delta W_l = eta * [ y_l * h_(l-1)^T - (y_l^2) * W_l ]

Use this during pretraining to make internal features non-degenerate.

BCM-like:

    theta_l <- (1 - mu) * theta_l + mu * y_l^2

    Delta W_l = eta * [ y_l * (y_l - theta_l) * h_(l-1)^T - lambda * W_l ]

This gives local strengthening and weakening.

---

## 7.5 Memory router learning: EM-like assignment

Suppose answer y was supported by chunk c_i.

Latent assignment:

    q_i = p(c_i | x, y)

Use retrieval scores:

    q_i = softmax(R_i)

If known support chunk c^+ exists, contrastive update:

    L_rank = -log( exp(R(c^+, x)) / (exp(R(c^+, x)) + Sum_j exp(R(c_j^-, x))) )

But avoid backprop by updating router weights with local perceptron/LMS:

    R_i = w^T * f_i

    w <- w + eta * (q_i_target - q_i) * f_i

where:

    f_i = [ R_exact, R_gzip, R_sem, R_kNN, R_trace ]

This trains the memory mixer without backprop.

---

## 7.6 Gate learning: RLS / bandit update

Memory gate:

    gamma_t = sigma(u^T * s_t)

where:

    s_t = [ H(p_t), max R_i, margin(R), conflict, rho_t ]

If memory helped, reward:

    r = 1

If memory hurt:

    r = -1

Prediction:

    r_hat = u^T * s_t

LMS update:

    u <- u + eta * (r - r_hat) * s_t

This is basically “learn when to listen to the goblin.”

---

## 7.7 Low-dimensional perturbation learning

For small global knobs only:

    theta_g = [ lambda_e, lambda_g, lambda_s, lambda_k, lambda_r, beta, gamma_max ]

Use SPSA:

    g_hat_i = (L(theta_g + c * Delta) - L(theta_g - c * Delta)) / (2 * c * Delta_i)

    theta_g <- theta_g - eta * g_hat

SPSA is attractive because it estimates a gradient using only two loss evaluations regardless of dimension. ([Wikipedia][23])

Use it for tiny hyper-controllers, not big matrices.

---

# 8. Ternary / BitNet-compatible weights

Maintain master weights:

    M_l in R^(d_out x d_in)

Forward weights:

    W_l = Q(M_l)

where:

    Q(M) = alpha * sign(M) * 1(|M| > tau)

No-backprop update applies to master:

    M_l <- M_l + Delta M_l

Then forward pass requantizes:

    W_l = Q(M_l)

This keeps the model deployment cheap while learning remains stable.

---

# 9. Full NOMAD equation

For token t, inner iteration n:

    r_t^n = FastRead(A_t, q_t^n)

    m_t^n = MemoryBus(x_<=t, h_t^n)

    h_t^(n+1) = (1 - eta_n) * h_t^n + eta_n * Phi_theta(h_t^n, e_t, r_t^n, m_t^n)

Final hidden state:

    h_t^* = h_t^N

Logits:

    z_t = W_o * h_t^* + b_o + beta * b_M

Probability:

    p_t = softmax(z_t)

Fast memory write:

    A_t = lambda_A * A_(t-1) + eta_A * phi(k_t) * v_t^T

Long-term memory update:

    M <- M cup Ingest(d)

No-backprop parameter update:

    Delta W_l = eta_l * [local target error] * [local presynaptic activity]^T

The architecture’s learning principle is:

    local error * local activity

with targets coming from DFA, Kaczmarz projection, RLS, contrastive phases, or memory rewards.

---

# 10. Training curriculum

## Phase A: self-organizing features

Use Oja/BCM on raw text streams.

Goal:

    stable local features

No labels needed.

Updates:

    Delta W_l = eta * [ y_l * h_(l-1)^T - y_l^2 * W_l ]

---

## Phase B: next-token readout

Freeze most body weights.

Train W_o with LMS/RLS.

Goal:

    prove reservoir/body has predictive signal

---

## Phase C: body learning with Kaczmarz/DFA

Enable local body updates.

For each layer:

    h_l_target = h_l + alpha * B_l * e_t^out

Kaczmarz update:

    w_(l,j) <- w_(l,j) + eta * ((h_l_target_j - w_(l,j)^T * h_(l-1)) / (||h_(l-1)||^2 + epsilon)) * h_(l-1)

Goal:

    learn representations without backprop

---

## Phase D: memory training

Add external documents and traces.

Train:

```text
router weights
memory gate
memory token bias
trace retriever
```

using local LMS/RLS/ranking updates.

Goal:

    P(y | x, M + Delta M) > P(y | x, M)

with:

    Delta theta_body = 0

---

## Phase E: fixed-point reasoning

Train on tasks requiring iterative settling:

```text
arithmetic traces
code completion
document QA
multi-hop factual recall
contradiction handling
```

Measure:

    rho_n = ||h^(n+1) - h^n||

Require:

    rho_N < tau

and answer correctness.

---

# 11. What is genuinely new here?

The architecture is not one existing method. It combines:

    Kaczmarz local target projection

    RLS/Kalman-style online readout

    Oja/BCM self-organization

    DFA teaching signals

    fast-weight attention

    Hopfield/fixed-point attractor dynamics

    external reconstructive memory

    compression-gain memory scoring

The most novel core is this:

    [ train a small iterative memory-using model by per-neuron projection updates, not backprop ]

That is the spicy part.

---

# 12. First experimental version

I would not build the full beast first.

Build **NOMAD-0**:

```text
Model size:
  10M–30M params

Core:
  recurrent fixed-point body
  fast-weight attention memory
  tied vocab head

Memory:
  exact + compression + semantic chunks

Training:
  head: RLS/LMS
  body: Kaczmarz + DFA targets
  features: Oja/BCM regularizer
  gates: LMS
  global knobs: SPSA
```

Disable initially:

```text
hidden-kNN memory
trace memory
Anderson acceleration
complex semantic retriever
```

Then add one component at a time.

---

# 13. Main hypotheses

## H1: Kaczmarz target projection can train body layers

    Kaczmarz-DFA > fixed random body

on next-token and arithmetic SFT.

## H2: RLS head gives fast learning

    RLS readout > LMS readout

especially in low-data SFT.

## H3: fast-weight memory improves local context

    fast memory on > fast memory off

for copy, code, and syntax tasks.

## H4: external memory improves new knowledge

    new-doc QA_(M + Delta M) > new-doc QA_M

without changing core weights.

## H5: fixed-point loop improves reasoning

    iterative settling > single pass

on arithmetic/code reasoning tasks.

---

# 14. Minimal update algorithm

For each batch:

```text
1. Forward only.
2. Run fixed-point inner loop.
3. Compute output error e = y - p.
4. Update output head with LMS/RLS.
5. Send direct random teaching signal to each layer:
      delta_l = B_l * e
6. Make local target:
      h_l_target = h_l + alpha * delta_l
7. Update each neuron with Kaczmarz projection.
8. Update fast memory by outer product.
9. Update router/gate using local reward or support labels.
10. Requantize ternary weights.
```

No backward graph.

No optimizer state for full model.

No activation tape.

That is the laptop angle.

---

# 15. The architecture in one box

```text
A_t = lambda_A * A_(t-1) + eta_A * phi(k_t) * v_t^T
m_t^n = Sum_(i in TopK) a_i^n * V_M * c_i
h_t^(n+1) = (1 - eta_n) * h_t^n + eta_n * Phi_theta(h_t^n, e_t, A_t, m_t^n)
z_t = W_o * h_t^* + beta * b_M
p_t = softmax(z_t)
e_t = y_t - p_t
W_o <- W_o + eta_o * e_t * (h_t^*)^T
h_l_target = h_l + alpha * B_l * e_t
w_(l,j) <- w_(l,j) + eta * ((h_l_target_j - w_(l,j)^T * h_(l-1)) / (||h_(l-1)||^2 + epsilon)) * h_(l-1)
```

That is the math-first skeleton.

---

# 16. Verdict

Yes: we can design a new architecture from scratch.

The strongest version is not “Transformer but no backprop.”

It is:

[ small fixed-point recurrent model + fast-weight attention + external reconstructive memory + Kaczmarz/RLS/DFA/Oja no-backprop learning ]

The underrated math I would bet on most:

[ Kaczmarz projection for local neuron updates ]

[ RLS/Kalman for readout and gates ]

[ fast-weight outer-product attention ]

[ Hopfield/fixed-point energy settling ]

GzipT remains useful, but as one memory signal. The training heart should probably be **Kaczmarz + DFA targets + RLS head**.

That is the candidate architecture I would build next.

[1]: https://arxiv.org/html/2401.02842v1?utm_source=chatgpt.com "Survey of a class of iterative row-action methods"
[2]: https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586.short?utm_source=chatgpt.com "A Stochastic Approximation Method"
[3]: https://pure.mpg.de/pubman/item/item_2346268_3/component/file_2346267/Hebb_1949_The_Organization_of_Behavior.pdf?utm_source=chatgpt.com "The Organization of Behavior"
[4]: https://neurophysics.ucsd.edu/courses/physics_171/Oja_1982.pdf?utm_source=chatgpt.com "Simplified neuron model as a principal component analyzer"
[5]: https://www.scholarpedia.org/article/BCM_theory?utm_source=chatgpt.com "BCM theory"
[6]: https://www-isl.stanford.edu/~widrow/papers/j199030years.pdf?utm_source=chatgpt.com "30 Years of Adaptive Neural Networks"
[7]: https://en.wikipedia.org/wiki/Kalman_filter?utm_source=chatgpt.com "Kalman filter"
[8]: https://www.pnas.org/doi/10.1073/pnas.79.8.2554?utm_source=chatgpt.com "Neural networks and physical systems with emergent ..."
[9]: https://epubs.siam.org/doi/10.1137/10078356X?utm_source=chatgpt.com "Anderson Acceleration for Fixed-Point Iterations"
[10]: https://www.math.unm.edu/~vageli/courses/Ma576/papers/Broyden_65.pdf?utm_source=chatgpt.com "A Class of Methods for Solving Nonlinear Simultaneous ..."
[11]: https://www.jstor.org/stable/2984875?utm_source=chatgpt.com "Maximum Likelihood from Incomplete Data via the EM ..."
[12]: https://pubmed.ncbi.nlm.nih.gov/12433288/?utm_source=chatgpt.com "Real-time computing without stable states"
[13]: https://www.scholarpedia.org/article/Echo_state_network?utm_source=chatgpt.com "Echo state network"
[14]: https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com "Attention Is All You Need"
[15]: https://arxiv.org/abs/2008.02217?utm_source=chatgpt.com "Hopfield Networks is All You Need"
[16]: https://arxiv.org/abs/2102.11174?utm_source=chatgpt.com "Linear Transformers Are Secretly Fast Weight Programmers"
[17]: https://arxiv.org/abs/1411.0247?utm_source=chatgpt.com "Random feedback weights support learning in deep neural networks"
[18]: https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00024/full?utm_source=chatgpt.com "Equilibrium Propagation: Bridging the Gap between ..."
[19]: https://pubmed.ncbi.nlm.nih.gov/28333583/?utm_source=chatgpt.com "An Approximation of the Error Backpropagation Algorithm ..."
[20]: https://arxiv.org/abs/2212.13345?utm_source=chatgpt.com "The Forward-Forward Algorithm: Some Preliminary Investigations"
[21]: https://arxiv.org/html/2503.24322v2?utm_source=chatgpt.com "NoProp: Training Neural Networks without Full Back ..."
[22]: https://arxiv.org/abs/2312.00752?utm_source=chatgpt.com "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
[23]: https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation?utm_source=chatgpt.com "Simultaneous perturbation stochastic approximation"
