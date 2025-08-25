---
title: Bonus Calculator
---
# Bonus Calculator

- You are given a list of employee dictionaries.
- Each dictionary has at least a `'role'` and a `'salary'` key.
- For every employee whose `'role'` is `'developer'`, add a new key `'bonus'` to their dictionary, with a value equal to 10% of their salary.
- For all other employees, set their `'bonus'` to `0`.
- Return the updated list of employee dictionaries.

```Python
def bonus_calculator(employees):
    for employee in employees:
        bonus = 0
        if employee['role'] == 'developer':
            bonus = employee['salary'] * 0.1
        employee['bonus'] = bonus
    return employees
```

```Python
def salary_increment(employees):
    employees_bonus = copy.deepcopy(employees)
    for i in range(len(employees_bonus)):
        if employees_bonus[i]["role"] == "developer":
            employees_bonus[i]["salary"] = employees_bonus[i]["salary"] * 1.15
    return employees_bonus
```