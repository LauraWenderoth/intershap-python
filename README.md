# intershap

Official Python implementation of InterShap. Still work in progress. First releas planned for April 2026.

## Installation

You can install the package via pip:

```bash
pip install intershap
```

## Usage

```python
import intershap
# Example usage
```

## Features

- Explainability tools for machine learning
- Easy integration with scikit-learn, numpy, and matplotlib

## License

Apache 2.0

## Author

Laura Wenderoth

###

The Shapley value of modality $A$ is defined as:

$$
\phi_A
=
\sum_{S \subseteq N \setminus \{A\}}
\frac{|S|!(M-|S|-1)!}{M!}
\left(
f(S \cup \{A\}) - f(S)
\right)
$$

The Shapley interaction value between modalities \(A\) and \(B\) is:

$$
\phi_{A,B} = \sum_{S \subseteq N \setminus \{A,B\}}
\frac{|S|!(M-|S|-2)!}{2(M-1)!}
\Big( f(S \cup \{A,B\}) + f(S) - f(S \cup \{A\}) - f(S \cup \{B\}) \Big)
$$

The Shapley value of $A$ decomposes as:

$$
\phi_A
=
\phi_{A,A}
+
\sum_{B \neq A} \phi_{A,B}
$$

Therefore, the main effect of $A$ is:

$$
\phi_{A,A}
=
\phi_A
-
\sum_{B \neq A} \phi_{A,B}
$$
