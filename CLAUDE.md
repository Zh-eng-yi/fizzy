## Project Docs

Always read these files at the start of every session to stay current on scope and design:

- `REQUIREMENTS.md` — weekly milestones, feature plan, and cross-cutting policies
- `ARCHITECTURE.md` — component responsibilities, data flow, and provider routing

---

## Agent Workflow: Strict Test-Driven Development (TDD)

You are operating under a strict TDD protocol. Whenever you are asked to implement a new feature, build a module, or write logic, you MUST follow this exact sequence. Do not combine these steps into a single response.

### 1. Requirement Clarification
Analyze the request for missing edge cases, ambiguous data types, or unhandled states. If anything is unclear, print a list of clarifying questions to the terminal and **STOP**. Wait for the user to reply.

### 2. Design & Planning
Once requirements are clear, outline how this feature fits into the system.
- Propose the high-level approach, the specific files to be modified, and the exact interfaces (class names, function signatures, data flows).
- Ensure the plan adheres to the boundaries established in `ARCHITECTURE.md`.
- **STOP**. Print: "Please review this architectural approach. Should we adjust the design before I write the tests?" Wait for terminal input.

### 3. Test Generation (Red Phase)
Once the plan is approved, write the complete test suite first.
- Include happy paths, exact boundary/threshold conditions, and unhappy paths (nulls, errors).
- **CRITICAL:** After creating the test file, you must **STOP**. Print the following exactly: "Please review the tests. Should I add any missing cases, or are we clear to implement?"
- Do NOT write any implementation code. Do NOT run the tests yet. Wait for terminal input.

### 4. Implementation (Green Phase)
Only after the user explicitly types approval in the terminal, write the implementation code.
- Write ONLY the code required to make the approved tests pass. 
- Do not invent extra features.

### 5. Verification & Refactor
Run the test suite using the project's standard test command. If the tests pass, evaluate your implementation for cleanliness and suggest refactoring if necessary.

---

## Post-Implementation Protocols

### Architecture Sync
When a task is fully complete and tests are green, evaluate if the implementation introduced new components, changed data flows, or altered dependencies. If so, you MUST update `ARCHITECTURE.md` to reflect the new system state before declaring the task finished.

### Requirements Integrity
Never update `REQUIREMENTS.md` with technical implementation details. Keep it focused purely on user behavior, business rules, component responsibilities, and acceptance criteria.
