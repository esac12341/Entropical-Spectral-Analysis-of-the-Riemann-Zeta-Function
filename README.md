# Spectral Interpretation of the Riemann Zeta Function
**Entropy Dynamics and the Berry-Keating Hamiltonian on the Modular Surface**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Preprint-green)](https://arxiv.org)

## Abstract
This repository contains the source code and datasets for the research paper "Spectral Interpretation of the Riemann Zeta Function." We analyze the zeros of the Riemann Zeta function ($\zeta(s)$) up to $t=10,000$ using quantum chaos theory. By treating the zeros as eigenvalues of a Hamiltonian operator, we identify a statistically significant logarithmic decline in Shannon Entropy ($p < 10^{-9}$) and a positive Lyapunov Exponent ($\lambda \approx +0.19$). These findings provide numerical evidence that the Riemann Zeros behave as the spectrum of the Berry-Keating Hamiltonian ($H=xp$) on a hyperbolic manifold, adhering to the uncertainty principle.

## Key Visualizations
* **Entropy Decline:** Verification of structure emerging over long ranges ($t \to 10,000$).
* **Spectrogram Analysis:** Identification of resonant "ringing" modes in the zero distribution.
* **Phase Space:** 3D delay-embedding revealing a low-dimensional chaotic attractor.

## Getting Started

### Prerequisites
* Python 3.8+
* Libraries: `mpmath`, `numpy`, `pandas`, `matplotlib`, `tqdm`

### Installation
```bash
git clone [https://github.com/esac12341/Riemann-Zeta-Entropy.git](https://github.com/esac12341/Riemann-Zeta-Entropy.git)
cd Riemann-Zeta-Entropy
pip install -r requirements.txt
