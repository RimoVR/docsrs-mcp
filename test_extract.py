#!/usr/bin/env python3

import sys
import json
from src.docsrs_mcp.ingestion.code_examples import extract_code_examples

test_doc = """
# Example Function

Here's how to use this:

```rust
fn main() {
    println!("Hello, world!");
}
```

And another example:

```python
def hello():
    print("Hello from Python")
```
"""

result = extract_code_examples(test_doc)
print(f"Result: {result}")
if result:
    examples = json.loads(result)
    print(f"Parsed: {json.dumps(examples, indent=2)}")
else:
    print("No examples found")