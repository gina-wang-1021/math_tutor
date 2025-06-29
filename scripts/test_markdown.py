import streamlit as st

# Set page title
st.title("Markdown and Math Formatting Test")

# Create a markdown string with various formatting elements
markdown_test = r"""
# Math Tutor Response Example

## Basic Formatting
**Bold text** for emphasis
*Italic text* for subtle emphasis

## Lists
### Numbered List
1. First step in solving the equation
2. Second step in the process
3. Final calculation

### Bullet Points
- Important concept: Integration
- Key formula: Derivative
- Application: Area under curve

## Math Equations
### Inline Math
The quadratic formula is $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$ which we use to solve quadratic equations.

### Block Equations
For the integral of a function:

$$\int_{a}^{b} f(x) dx = F(b) - F(a)$$

And for the famous Euler's identity:

$$e^{i\pi} + 1 = 0$$

## Code Block
```python
def calculate_derivative(f, x, h=0.0001):
    return (f(x + h) - f(x)) / h
```

## Table
| Function | Derivative |
|----------|------------|
| $x^n$ | $nx^{n-1}$ |
| $\sin(x)$ | $\cos(x)$ |
| $e^x$ | $e^x$ |

## Blockquote
> Remember: Always check your work by substituting your answer back into the original equation.

## Horizontal Rule
---

## Links
[Learn more about calculus](https://en.wikipedia.org/wiki/Calculus)

## Subscript and Superscript
Water is H₂O and the area of a circle is πr²

## Warning Note
⚠️ **Note:** AI calculations may contain errors. Always verify important results.
"""

# Display the markdown
st.markdown(markdown_test)

# Show raw markdown for reference
with st.expander("View Raw Markdown"):
    st.code(markdown_test)
