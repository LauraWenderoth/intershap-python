# intershap

Official Python implementation of InterShap.

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

For modality i:

$$
\phi_i = \sum_{S \subseteq N \setminus {i}}
\frac{|S|!(M-|S|-1)!}{M!} \big( f(S \cup {i}) - f(S) \big)
$$

The Shapley interaction value between modalities \(A\) and \(B\) is:

$$
\phi_{A,B} = \sum_{S \subseteq N \setminus \{A,B\}}
\frac{|S|!(M-|S|-2)!}{2(M-1)!}
\Big( f(S \cup \{A,B\}) + f(S) - f(S \cup \{A\}) - f(S \cup \{B\}) \Big)
$$
