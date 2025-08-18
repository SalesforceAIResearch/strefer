NOUN_GEN_PROMPT = """Your goal is to extract the generalized noun for each item in a list and return the results as a JSON list.
**Rules**
1. Replace all humans — including children — with "person".
2. Replace all other terms with their appropriate broad category noun.

**Examples:**

**Entities:**: 
```json
["small white dog", "child in a red polka-dotted outfit", "person wearing a light green shirt"]
```
**Nouns:**:
```json
["dog", "person", "person"]
```

**Entities:**: 
```json
["baby elephant", "parrot", "adult", "white church"]
```
**Nouns:**:
```json
["elephant", "bird", "person", "church"]
```


You will use the format in the examples. You will only remain the main noun.
Now it is your turn. 

**Entities:**: 
```json
{entities}
```
**Nouns:**:
"""

