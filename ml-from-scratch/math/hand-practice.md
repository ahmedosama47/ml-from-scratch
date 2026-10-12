# Hand practice: do on paper, then check the answer key at the bottom

Use natural log (ln). Sigmoid g(z) = 1/(1+e^-z). Round to 4 decimals.
Remember: b is never regularized.

## 1. Logistic regression, one gradient descent step
Data (m = 2): x = [0, 2], y = [0, 1]. Start w = 0, b = 0, α = 0.1.
a) f for each example
b) cost J
c) dJ/dw and dJ/db
d) updated w and b
e) new f(2) and new J. Did J go down?

## 2. Logistic regression with L2 regularization
Same data. Start w = 1, b = 0, λ = 1, α = 0.1.
a) f for each example
b) J_reg = J + (λ/2m)·w²
c) dJ_reg/dw (remember + (λ/m)·w) and dJ/db
d) updated w and b

## 3. Logistic regression with scaling
Data (m = 3): x = [100, 200, 300], y = [0, 1, 1]. Start w = 0, b = 0, α = 0.1.
a) z-score the feature (use the population std: divide by m).
b) Gradient step on the scaled x: dJ/dw, dJ/db, new w, new b.
c) Now do ONE gradient step on the unscaled x. What is the new w?
   What is z = w·300 + b afterwards, and what does that do to f?

## 4. Linear regression with L2 regularization
Data (m = 2): x = [1, 2], y = [2, 4]. Start w = 1, b = 0, λ = 1, α = 0.1.
a) predictions f = w·x + b
b) J_reg = (1/2m)·Σ(f-y)² + (λ/2m)·w²
c) dJ_reg/dw and dJ_reg/db
d) updated w and b

## 5. Thinking questions
a) In problem 2, why is dJ/dw bigger with regularization than without?
b) In problem 3, why is the unscaled step a disaster?
c) What does λ do to w over many steps?

---
## Answer key (don't look before trying)

**1.** a) f = [0.5, 0.5]  b) J = 0.6931
c) errors = [0.5, -0.5]; dJ/dw = (1/2)(0.5·0 + (-0.5)·2) = -0.5; dJ/db = 0
d) w = 0.05, b = 0
e) f(2) = g(0.1) = 0.5250; J = (0.6931 + 0.6444)/2 = 0.6688. Yes, J went down.

**2.** a) f = [g(0), g(2)] = [0.5, 0.8808]
b) J = (0.6931 + 0.1269)/2 = 0.4100; J_reg = 0.4100 + 0.25 = 0.6600
c) errors = [0.5, -0.1192]; dJ/dw = (1/2)(-0.1192·2) + (1/2)(1) = -0.1192 + 0.5 = 0.3808
   dJ/db = (1/2)(0.5 - 0.1192) = 0.1904
d) w = 1 - 0.1·0.3808 = 0.9619; b = 0 - 0.1·0.1904 = -0.0190

**3.** a) mean = 200, std = 81.65 → x_scaled = [-1.2247, 0, 1.2247]
b) errors = [0.5, -0.5, -0.5]; dJ/dw = (1/3)(-0.6124 + 0 - 0.6124) = -0.4082
   dJ/db = (1/3)(-0.5) = -0.1667 → w = 0.0408, b = 0.0167
c) unscaled: dJ/dw = (1/3)(50 - 100 - 150) = -66.67 → w = 6.67.
   z = 6.67·300 = 2000, so f is 1 for every example with x > 0, and the loss on
   the y = 0 example explodes. The step overshoots.

**4.** a) f = [1, 2]
b) errors = [-1, -2]; (1/4)(1 + 4) = 1.25; + (1/4)(1) = 0.25 → J_reg = 1.5
c) dJ/dw = (1/2)(-1·1 + -2·2) + (1/2)(1) = -2.5 + 0.5 = -2.0; dJ/db = (1/2)(-3) = -1.5
d) w = 1.2, b = 0.15

**5.** a) The penalty adds + (λ/m)·w, which pulls w toward 0 (here w = 1, so +0.5).
b) Feature values up to 300 make the gradient huge (-66.67) while α stayed 0.1.
   Scaling puts features near [-1, 1] so one α works.
c) Each step multiplies w by (1 - α·λ/m) before the usual update, so it shrinks
   ("weight decay"). λ → ∞ drives all w to 0 (underfit).
