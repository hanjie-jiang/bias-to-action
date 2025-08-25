---
title: Password Strength Counter
---
# Password Strength Counter

A common measure to ensure robust security is testing the strength of passwords. A 'strong' password is usually defined as one that is long (at least 8 characters) and includes a mix of uppercase characters, lowercase characters, and digits.
### Skeleton of the code

```Python
if len(password) >= 8: 
	continue 
else: 
	return false 
	
dict_check = {"upper": false, "lower": false, "digit": false} 

for character in password: # password is a list, no need for .items()
	if character.isnumeric(): 
		dict_check["digit"] = true 
	elif character.isupper(): 
		dict_check["upper"] = true 
	elif character.islower(): 
		dict_check["lower"] = true 
	else: 
		continue 
		
for _, value in dict_check.items(): 
	if value == false: 
		return false 

return true
```

### Implementation of the code

```Python
def password_strength_counter(password):
    strength = {
        'length': False,
        'digit': False,
        'lowercase': False,
        'uppercase': False,
    }
    if len(password) >= 8:
        strength['length'] = True
    for char in password:
        if char.isdigit():
            strength['digit'] = True
        elif char.islower():
            strength['lowercase'] = True
        elif char.isupper():
            strength['uppercase'] = True
    return strength
```