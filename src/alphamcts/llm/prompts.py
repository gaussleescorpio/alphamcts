"""LLM prompts, following the paper's Appendix K (Figures 15-18).

The four core prompts are reproduced faithfully; a fifth prompt (refinement suggestion
generation) is described but not printed verbatim in the paper, so it is written here to
match the described behavior (dimension-targeted, few-shot exemplars, node context).

Templates use ``string.Template`` ($placeholders) to avoid brace-escaping in JSON examples.
"""

from __future__ import annotations

from string import Template

# ---------------------------------------------------------------------------
# Figure 15: Alpha Portrait Generation Prompt
# ---------------------------------------------------------------------------

ALPHA_PORTRAIT_PROMPT = Template("""Task Description:
You are a quantitative finance expert specializing in factor-based investing. Please design an alpha factor used in investment strategies according to the following requirements, and then provide the content of the alpha in the required format.

Available Data Fields:
The following data fields are available for use:
```
$available_fields
```

Available Operators:
The following operators are available for use:
```
$available_operators
```

Alpha Requirements:
1. The alpha value should be dimensionless (unitless).
2. The alpha should incorporate at least two distinct operations from the "Available Operators" list to ensure it has sufficient complexity. Avoid creating overly simplistic alphas.
3. All look-back windows and other numerical parameters used in the alpha calculation MUST be represented as named parameters in the pseudo-code. These parameter names MUST follow Python naming conventions (e.g., lookback_period, volatility_window, smoothing_factor).
4. The alpha should have NO MORE than 3 parameters in total.
5. The pseudo-code should represent the alpha calculation step-by-step, using only the "Available Operators" and clearly defined parameters. Each line in the pseudo-code should represent a single operation.
6. Use descriptive variable names in the pseudo-code that clearly indicate the data they represent.
7. When designing alpha expressions, try to avoid including the following sub-expressions:
```
$freq_subtrees
```

Formatting Requirements:
The output must be in JSON format with three key-value pairs:
1. "name": A short, descriptive name for the alpha (following Python variable naming style, e.g., price_volatility_ratio).
2. "description": A concise explanation of the alpha's purpose or what it measures. Avoid overly technical language. Focus on the intuition behind the alpha.
3. "pseudo_code": A list of strings, where each string is a line of simplified pseudo-code representing a single operation in the alpha calculation. Each line should follow the format: variable_name = op_name(input=[input1, input2, ...], param=[param1, param2, ...]), where:
- variable_name is the output variable of the operation.
- op_name is the name of one of the "Available Operators".
- input1, input2, ... are input variables (either from "Available Data Fields" or previously calculated variables, cannot be of a numeric type).
- param1, param2, ... are parameter names defined in the alpha requirements.

The format example is as follows:
{
"name": "volatility_adjusted_momentum",
"description": "......",
"pseudo_code": [......]
}
""")

# ---------------------------------------------------------------------------
# Figure 16: Alpha Formula Generation Prompt
# ---------------------------------------------------------------------------

ALPHA_FORMULA_PROMPT = Template("""Task Description:
Please design a quantitative investment alpha expression according to the following requirements.

Available Data Fields:
- The following data fields are available for use: $available_fields

Available Operators:
- The following operators are available for use:
$available_operators

Alpha Requirements:
$alpha_portrait_prompt

Formatting Requirements:
1. Provide the output in JSON format.
2. The JSON object should contain two fields: "formula", and "arguments".
- "formula": Represents the mathematical expression for calculating the alpha.
- "arguments": Represents the configurable parameters of the alpha.
3. "formula" is a list of dictionaries. Each dictionary represents a single operation and must contain four keys: "name", "param", "input", and "output".
- "name": The operator's name (a string), which MUST be one of the operators provided in the "Available Operators" section.
- "param": A list of strings, representing the parameter names for the operator. These parameter names MUST be used as keys in the "arguments" section.
- "input": A list of strings, representing the input variable names for the operator. These MUST be data fields from the "Available Data Fields" or output variables from previous operations in the "formula", cannot be of a numeric type.
- "output": A string, representing the output variable name for the operator. This output can be used as an input for subsequent operations.
4. "arguments" is a list of dictionaries. Each dictionary represents a set of parameter values for the alpha.
- The keys of each dictionary in "arguments" MUST correspond exactly to the parameter names defined in the "param" lists of the "formula".
- The values in each dictionary in "arguments" are the specific numerical values for the parameters.
5. You may include a maximum of 3 sets of parameters within the "arguments" field.
6. The parameter value that indicates the length of the lookback window (if applicable) must be within the $window_range range.
7. Ensure that the alpha expression is both reasonable and computationally feasible.
8. Parameter names should be descriptive and follow Python naming conventions (e.g., window_size, lag_period, smoothing_factor). Avoid using single characters or numbers as parameter names.
9. Refer to the following example:
{
"formula": [......],
"arguments": [......]
}
""")

# ---------------------------------------------------------------------------
# Figure 17: Alpha Overfitting Risk Assessment Prompt
# ---------------------------------------------------------------------------

OVERFITTING_PROMPT = Template("""Task: Critical Alpha Overfitting Risk Assessment

Critically evaluate the overfitting risk and generalization potential of the provided quantitative investment alpha, based on its expression and refinement history. Your assessment must focus on whether complexity and optimization appear justified or are likely signs of overfitting.

Input:
- Alpha Expression:
```
$alpha_formula
```
- Refinement History:
```
$refinement_history
```

Evaluation Criteria:
1. Justified Rationale vs. Complexity:
Critique: Is the complexity of the alpha expression plausibly justified by an inferred economic rationale, or does it seem arbitrary/excessive, suggesting fitting to noise?
2. Principled Development vs. Data Dredging:
Critique: Does the refinement history indicate hypothesis-driven improvements, or does it suggest excessive optimization and curve-fitting (e.g., frequent, unjustified parameter tweaks)?
3. Transparency vs. Opacity:
Critique: Is the alpha's logic reasonably interpretable despite its complexity, or is it opaque, potentially masking overfitting?

Scoring & Output:
- Assign a single Overfitting Risk Score from 0 to 10.
  - 10 = Very Low Risk (High confidence in generalization)
  - 0 = Very High Risk (Low confidence in generalization)
- Use the full 0-10 range to differentiate risk levels effectively.
- Provide a concise, one-sentence Justification explaining the score, citing the key factors from the criteria.
- Format the output as JSON, like the examples below:

Example JSON Outputs:
{
"reason": "Complexity is justified by a strong rationale; principled refinement history suggests low risk.",
"score": 9
}
{
"reason": "Plausible rationale, but some expression opacity and parameter tuning in history indicate moderate risk.",
"score": 5
}
{
"reason": "High risk inferred from opaque expression lacking clear rationale, supported by history showing excessive tuning.",
"score": 1
}
""")

# ---------------------------------------------------------------------------
# Figure 18: Alpha Refinement Prompt
# ---------------------------------------------------------------------------

ALPHA_REFINEMENT_PROMPT = Template("""Task Description:
There is an alpha factor used in quantitative investment to predict asset price trends. Please improve it according to the following suggestions and provide the improved alpha expression.

Available Data Fields:
The following data fields are available for use: $available_fields

Available Operators:
The following operators are available for use:
$available_operators

Alpha Suggestions:
1. The alpha value should be dimensionless (unitless).
2. All look-back windows and other numerical parameters used in the alpha calculation MUST be represented as named parameters in the pseudo-code. These parameter names MUST follow Python naming conventions (e.g., lookback_period, volatility_window, smoothing_factor).
3. The alpha should have NO MORE than 3 parameters in total.
4. The pseudo-code should represent the alpha calculation step-by-step, using only the "Available Operators" and clearly defined parameters. Each line in the pseudo-code should represent a single operation.
5. Use descriptive variable names in the pseudo-code that clearly indicate the data they represent.
6. When designing alpha expressions, try to avoid including the following sub-expressions: $freq_subtrees

Original alpha expression:
$origin_alpha_formula

Refinement suggestions:
NOTE: The following improvement suggestions do not need to be all adopted; they just need to be considered and reasonable ones selected for adoption.
$refinement_suggestions

Formatting Requirements:
The output must be in JSON format with three key-value pairs:
1. "name": A short, descriptive name for the alpha (following Python variable naming style, e.g., price_volatility_ratio).
2. "description": A concise explanation of the alpha's purpose or what it measures. Avoid overly technical language. Focus on the intuition behind the alpha.
3. "pseudo_code": A list of strings, where each string is a line of simplified pseudo-code representing a single operation in the alpha calculation. Each line should follow the format: variable_name = op_name(input=[input1, input2, ...], param=[param1, param2, ...]), where:
- variable_name is the output variable of the operation.
- op_name is the name of one of the "Available Operators".
- input1, input2, ... are input variables (either from "Available Data Fields" or previously calculated variables, cannot be of a numeric type).
- param1, param2, ... are parameter names defined in the alpha requirements.

The format example is as follows:
{
"name": "volatility_adjusted_momentum",
"description": "......",
"pseudo_code": [......]
}
""")

# ---------------------------------------------------------------------------
# Refinement suggestion generation (Appendix D; wording not printed in the paper)
# ---------------------------------------------------------------------------

SUGGESTION_PROMPT = Template("""Task Description:
You are a quantitative finance expert. There is an alpha factor used to predict asset price trends. Analyze it and propose targeted refinement suggestions to improve one specific evaluation dimension.

Target Dimension: $dimension
- Meaning: $dimension_description

Current Alpha:
- Expression: $alpha_expression
- Description: $alpha_description
- Evaluation scores (0-1, higher is better): $evaluation_scores
- Backtest metrics: $backtest_metrics

Refinement History Context (parent / siblings / children of this alpha in the search tree):
```
$refinement_context
```

Example effective alphas from the repository (for inspiration, do not copy):
```
$examples
```

When proposing suggestions, avoid recommending structures containing these sub-expressions:
```
$freq_subtrees
```

Requirements:
- Propose 1 to 3 concrete, actionable refinement suggestions specifically aimed at improving the target dimension.
- Each suggestion should describe a conceptual change grounded in an investment rationale (not just parameter tweaking).
- Avoid repeating refinements already attempted in the history context.

Format the output as JSON:
{
"suggestions": ["...", "..."]
}
""")

# ---------------------------------------------------------------------------
# Formula correction feedback (used in the IsValid iterative correction loop)
# ---------------------------------------------------------------------------

CORRECTION_SUFFIX = Template("""

IMPORTANT - Correction Required:
Your previous output was invalid and must be fixed.

Previous output:
```
$previous_output
```

Validation error:
```
$error
```

Please regenerate the JSON output, fixing this error while keeping the alpha's intent.
""")
