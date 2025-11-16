# 1D Visco-Plasticity: Recurrent Neural Operator & Transformer Models

This project studies a **one-dimensional visco-elastic unit-cell problem**, and implements two neural operator models capable of learning the solution operator for materials with internal variables and memory effects.

---

## 🧠 Problem Description

We consider the following 1D visco-plastic constitutive problem defined on a unit cell. The unit cell contains 3 repeating phases with different stiffness and viscosity values:

![Kinematic relation](https://latex.codecogs.com/png.latex?\epsilon(x,t)=\frac{\partial%20u(x,t)}{\partial%20x}\qquad\text{(Kinematic%20relation)})

![Equilibrium](https://latex.codecogs.com/png.latex?\frac{\partial%20\sigma(x,t)}{\partial%20x}=0\qquad\text{(Equilibrium)})

![Constitutive relation](https://latex.codecogs.com/png.latex?\sigma(x,t)=E(x)\epsilon(x,t)+v(x)\frac{\partial%20u(x,t)}{\partial%20t}\qquad\text{(Constitutive%20relation)})

**Initial conditions:**

![IC](https://latex.codecogs.com/png.latex?u(x,0)=0,\qquad\dot{u}(x,0)=0)

**Boundary conditions:**

![BC](https://latex.codecogs.com/png.latex?u(0,t)=0,\qquad%20u(1,t)=\ell(t))

Here:
- E(x) is the spatially varying **Young’s modulus**
- v(x) is the spatially varying **viscosity**
- The material consists of **three phases**, where E(x) and v(x) are piecewise constant with three repeating values.

This creates a heterogeneous viscoelastic material requiring models that can capture **history-dependent** and **spatially varying** behavior.

---

## ⚙️ Models Implemented

### 1. Recurrent Neural Operator (RNO)

The **Recurrent Neural Operator** learns the evolution of the system using only **one internal variable**, making it highly efficient for time-dependent PDEs with memory.

**Key features:**
- Models viscoelastic response with **1 internal variable**
- Optimized using **Adam**
- Uses a **StepLR scheduler**
- Captures temporal evolution through recursive updates

The RNO shows strong performance even with compact representation, making it suitable for reduced-order models of visco-elastic systems.

---

### 2. Transformer Model

A Transformer-based operator learning model is also implemented to compare against RNO.

**Key features:**
- Attention layers to model long-range dependencies
- Optimized using **AdamW**
- Uses a **LambdaLR scheduler**
- Hyperparameters require further tuning to improve performance (e.g., number of heads, depth, embedding dimension)

While the Transformer captures complex patterns, it currently underperforms compared to the RNO due to sensitivity to hyperparameter choices.

---

## 📊 Summary of Findings

| Model          | Optimizer | LR Scheduler | Internal Variables | Performance |
|----------------|-----------|--------------|--------------------|-------------|
| **RNO**        | Adam      | StepLR       | 1                  | ✔ Stable, good accuracy |
| **Transformer**| AdamW     | LambdaLR     | N/A                | ⚠ Needs tuning |

The RNO provides a more stable and robust baseline for the visco-elastic operator learning task, while the Transformer has potential but requires further model refinement.

---

## 🙏 Acknowledgments

Special thanks to **Dr. Burigede Liu** and **Ms. Rui Wu** for providing the initial code framework as part of Course 4C11 at CUED.
