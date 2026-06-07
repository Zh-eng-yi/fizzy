## Agent Workflow: Strict Test-Driven Development (TDD)

You are operating under a strict TDD protocol. Whenever you are asked to implement a new feature, build a module, or write logic, you MUST follow this exact sequence. Do not combine these steps into a single response.

### 1. Requirement Clarification
Analyze the request for missing edge cases, ambiguous data types, or unhandled states. If anything is unclear, print a list of clarifying questions to the terminal and **STOP**. Wait for the user to reply.

### 2. Test Generation (Red Phase)
Once requirements are clear, write the complete test suite first.
- Include happy paths, exact boundary/threshold conditions, and unhappy paths (nulls, errors).
- **CRITICAL:** After creating the test file, you must **STOP**. Print the following exactly: "Please review the tests. Should I add any missing cases, or are we clear to implement?"
- Do NOT write any implementation code. Do NOT run the tests yet. Wait for terminal input.

### 3. Implementation (Green Phase)
Only after the user explicitly types approval in the terminal, write the implementation code.
- Write ONLY the code required to make the approved tests pass. 
- Do not invent extra features.

### 4. Verification & Refactor
Run the test suite using the project's standard test command. If the tests pass, evaluate your implementation for cleanliness and suggest refactoring if necessary.
