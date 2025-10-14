# Stack Implementation

Detailed Python implementation of stacks using different underlying structures.

## Stack Manipulation in Python

```python
def is_empty(stack):
    return len(stack) == 0

# Create an empty stack
stack = []
print(is_empty(stack)) # outputs True as stack is empty

stack.append("D")
print(is_empty(stack)) # outputs False as stack has an element "D"

stack_size = len(stack)
print(f"Size of stack: {stack_size}") # outputs 1 as length of stack is 1
```

## Stack Applications and Problem-solving

```python
# Create a stack to store text changes
# The stack stores all historical versions of editor states, excluding the current state
text_stack = []

# The user inputs text
text_stack.append("Hello, world!")
text_stack.append("Hello, CodeSignal!")

print(text_stack)
# Output: ["Hello, world!", "Hello, CodeSignal!"]

# Check if the stack is empty before performing "undo"
if not is_empty(text_stack):
    # The user performs an "undo" operation
    previous_text = text_stack.pop() # Retrieve the last historic change
    print(f'After "undo", the text is: {previous_text}')
else:
    print("Cannot perform undo operation. There are no historic changes.")

# Output: After "undo", the text is: Hello, CodeSignal!

```