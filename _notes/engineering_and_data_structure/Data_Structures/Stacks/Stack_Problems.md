# Stack Problems

LeetCode problems and common patterns using stack data structure.

## Valid Parentheses problems

```python
def are_brackets_balanced(input_str):
    brackets = set(["(", ")", "[", "]", "{", "}"])
    bracket_map = {"(": ")", "[": "]",  "{": "}"}
    open_par = set(["(", "[", "{"])
    stack = []

    for character in input_str:
        if character not in brackets:
            # Skipping non-bracket characters
            continue
        if character in open_par:
            stack.append(character)
        elif stack and character == bracket_map[stack[-1]]:
                stack.pop()
        else:
            return False
    return len(stack) == 0
```

## Reverse a String

### Naive Approach

```python
def reverse_string(string):
    return string[::-1]
```

### Efficient Approach
Using a stack, we can reverse elements by leveraging its LIFO property. The strategy is straightforward: push all the characters to a stack and then pop them out. As a result, we get the reversed string. This helps demonstrate a practical application of stack operations.

```python
def reverse_string(string):
    stack = list(string)
    results = ''

    i = 0
    while i < len(stack):
        results = results + stack.pop()
        i = i + 1
    return results
```

## Postfix Expression Evaluation
In simple terms, a postfix expression is an arithmetic expression where operators are placed after their operands. For example, the expression 2 3 + is a simple postfix expression, which equals 5 when evaluated.

### Efficient Approach
We create an empty stack. Then, we iterate over each character operand in the expression. If operand is a number, we push it onto the stack. If operand is an operator, we pop two numbers from the stack, perform the operation, and push the result back onto the stack. After we have processed all characters of the expression, the stack should contain exactly one element, the result of the expression.

```python
def evaluate_postfix(expression):
    stack = []
    for element in expression.split(' '):   
        if element.isdigit():             
            stack.append(int(element))

        else:
            operand2 = stack.pop()
            operand1 = stack.pop()
            
            if element == '+': stack.append(operand1 + operand2)
            elif element == '-': stack.append(operand1 - operand2)
            elif element == '*': stack.append(operand1 * operand2)
            elif element == '/': stack.append(operand1 / operand2)
    
    return stack[0]
```

This file will also contain:
- Monotonic Stack patterns
- Expression evaluation
- DFS using stacks
- Backtracking applications
- Next Greater Element problems