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

## Daily Temperatures - LeetCode #739

### Problem Statement

Given an array of integers `temperatures` representing daily temperatures, return an array `answer` such that `answer[i]` is the number of days you have to wait after the `i`-th day to get a warmer temperature. If there is no future day for which this is possible, keep `answer[i] == 0`.

**Example:**

```text
Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]
```

For each day, find how many days you'll have to wait until the next warmer day:

- Day 0 (73°): Next warmer is day 1 (74°) → wait 1 day
- Day 1 (74°): Next warmer is day 2 (75°) → wait 1 day  
- Day 2 (75°): Next warmer is day 6 (76°) → wait 4 days
- Day 3 (71°): Next warmer is day 5 (72°) → wait 2 days
- And so on...

### Monotonic Stack Solution

This is a classic **monotonic stack** problem. We use a decreasing monotonic stack to efficiently find the next warmer temperature.

```python
def dailyTemperatures(temperatures):
    result = [0] * len(temperatures)
    stack = []  # Store indices, not temperatures
    
    for i, temp in enumerate(temperatures):
        # While stack is not empty AND current temp > temp at stack top
        while stack and temperatures[stack[-1]] < temp:
            prev_index = stack.pop()
            result[prev_index] = i - prev_index  # Days to wait
        
        stack.append(i)  # Always push current index
    
    return result
```

## Daily Temperatures II 

### Problem Statement

Given an array of integers `temperatures` representing daily temperatures, return an array `answer` such that `answer[i]` is the number of days you have to wait after the `i`-th day to get a cooler temperature. If there is no future day for which this is possible, keep `answer[i] == -1`.

**Example:**

```text
Input: temperatures = [30, 60, 90, 120, 60, 30]
Output: [-1, 4, 2, 1, 1, -1]
```

### Monotonic Stack Solution

This is a classic **monotonic stack** problem. We use a decreasing monotonic stack to efficiently find the next warmer temperature.

```python
def days_until_cooler(temps):
    result = [-1] * len(temps)
    stack = []

    for i in range(len(temps) - 1, -1, -1):
        while stack and temps[stack[-1]] >= temps[i]: #stack[-1] should be the index of coolest temperature and compare it with current temp, if the current coolest is higher than current, then pop and substitute it with current temp
            stack.pop()
        result[i] = stack[-1] - i if stack else -1
        stack.append(i)
    
    return result
```

**Time Complexity:** O(n) - each element pushed and popped at most once  
**Space Complexity:** O(n) - for the stack and result array

**Key Insight:** The stack maintains indices of days in decreasing temperature order, allowing us to efficiently find the next warmer day for each temperature.

This file will also contain:

- More Monotonic Stack patterns
- Expression evaluation
- DFS using stacks
- Backtracking applications