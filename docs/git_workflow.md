# Git Feature Branch Workflow: A Best Practices Guide

As a team, adopting a consistent Git workflow is crucial for maintaining a clean, manageable, and collaborative codebase. This document outlines the standard operating procedure for feature development, from branch creation to merging and cleanup. Following these steps will reduce merge conflicts, improve code quality, and make our project history a valuable asset.

---

### Step 1 — Branch Naming Convention

A clear and consistent branch naming strategy is the first step to an organized repository. It allows team members to immediately understand the purpose of a branch just by looking at its name.

-   **Convention**: All feature branches must follow the `feat/{feature_name}` pattern.
-   **Structure**:
    -   `feat/`: A prefix indicating the branch is for a new feature.
    -   `{feature_name}`: A short, descriptive, kebab-case name for the feature.

-   **Good Examples**:
    -   `feat/user-login-api`
    -   `feat/shopping-cart-discount-logic`
    -   `feat/add-dark-mode-toggle`

-   **Why it Matters**:
    -   **Clarity**: Instantly tells everyone what the branch is for.
    -   **Automation**: Allows for automated tools (like CI/CD) to trigger specific workflows based on branch names.
    -   **Hygiene**: Keeps the repository's branch list clean and easy to navigate.

### Step 2 — Checkout a New Branch

Always start your work from the most up-to-date version of the `main` branch. This minimizes the chances of complex merge conflicts later.

1.  **Switch to the `main` branch**:
    ```bash
    git checkout main
    ```

2.  **Pull the latest changes from the remote**:
    ```bash
    git pull origin main
    ```

3.  **Create and switch to your new feature branch**:
    ```bash
    git checkout -b feat/your-feature-name
    ```

### Step 3 — Commit Your Changes

Clean, atomic commits are the foundation of a healthy project history. Each commit should represent a single, logical change.

-   **Best Practices**:
    -   **Atomic Commits**: Keep commits small and focused. One commit should do one thing well (e.g., "add user validation," not "add validation and update UI styles").
    -   **Descriptive Messages**: Follow the **Conventional Commits** specification. This format makes the commit history readable and allows for automated changelog generation.

-   **Commit Message Format**: `<type>: <description>`
    -   `feat`: A new feature for the user.
    -   `fix`: A bug fix for the user.
    -   `docs`: Documentation-only changes.
    -   `style`: Code style changes (formatting, semicolons, etc.).
    -   `refactor`: A code change that neither fixes a bug nor adds a feature.
    -   `test`: Adding missing tests or correcting existing tests.
    -   `chore`: Changes to the build process or auxiliary tools.

-   **Example Commit Messages**:
    ```bash
    git commit -m "feat: implement user authentication endpoint"
    git commit -m "fix: correct password validation regex"
    git commit -m "docs: update API documentation for the login route"
    ```

-   **Why it Matters**:
    -   **Easier Debugging**: Small commits make it simple to use tools like `git bisect` to find exactly where a bug was introduced.
    -   **Clearer Reviews**: Reviewers can understand the purpose of each change more easily.

### Step 4 — Submit a Pull Request (PR)

A Pull Request (PR) is a formal request to merge your changes into the `main` branch. It is the central hub for code review and discussion.

1.  **Push your feature branch to the remote repository**:
    ```bash
    git push origin feat/your-feature-name
    ```

2.  **Open a Pull Request**:
    -   Navigate to the repository in your Git provider (GitHub, GitLab, etc.).
    -   You will usually see a prompt to create a PR from your recently pushed branch.

-   **Writing a Good PR Description**:
    -   **Summary**: Briefly explain what the PR accomplishes.
    -   **Motivation**: Why is this change necessary? What problem does it solve?
    -   **Changes**: Detail the key changes made.
    -   **Testing**: Explain how you tested your changes.
    -   **Screenshots/Logs**: Include visuals or logs if the changes affect the UI or produce specific output.

### Step 5 — Request a Code Review

Code review is a critical quality gate. It helps catch bugs, enforce coding standards, and share knowledge across the team.

-   **Process**:
    -   Assign at least one team member as a reviewer on your PR.
    -   Reviewers should provide constructive, respectful feedback.
    -   The author should address all comments, pushing new commits to the same branch to update the PR.

-   **Reviewer's Role**:
    -   Check for bugs and logical errors.
    -   Ensure the code adheres to team style guides.
    -   Validate that the changes meet the feature's requirements.

### Step 6 — Merge to Main

Once a PR has been approved and has passed all automated checks, it can be merged.

-   **Prerequisites**:
    -   At least one reviewer has approved the changes.
    -   All CI/CD checks if applicable (e.g., tests, linting, builds) are passing.

-   **Merge Strategy**:
    -   Use **Squash and Merge**. This combines all of the feature branch's commits into a single, clean commit on the `main` branch. The commit message should be the PR title and description.

-   **Why it Matters**:
    -   **Quality Control**: Ensures that no un-reviewed or broken code reaches the `main` branch.
    -   **Clean History**: "Squash and Merge" keeps the `main` branch history linear and easy to read, with each commit corresponding to a single, complete feature or fix.

### Step 7 — Delete the Feature Branch

To keep the repository tidy, delete your feature branch both locally and on the remote after it has been successfully merged.

1.  **Delete the remote branch**:
    -   This is often done automatically by the Git provider when you merge the PR. If not, run:
    ```bash
    git push origin --delete feat/your-feature-name
    ```

2.  **Delete the local branch**:
    -   First, switch back to the `main` branch and pull the latest changes (which now include your merged feature).
    ```bash
    git checkout main
    git pull origin main
    ```
    -   Then, delete the local feature branch.
    ```bash
    git branch -d feat/your-feature-name
    ```

### Step 8 — Recap of Best Practices

This workflow is designed to maximize quality and collaboration while minimizing friction.

-   **Be Consistent**: Stick to the naming and commit conventions.
-   **Review Thoroughly**: Treat code reviews as a top priority.
-   **Document as You Go**: Update documentation (like READMEs or API docs) in the same PR as your code changes.
-   **Communicate**: Keep your team informed about your progress and any blockers.

By internalizing these practices, we build a more robust, maintainable, and professional engineering culture.