# Project Specification: Git Plumbing (Init & Hash-Object)

## 1. Objective
Create a Python script `pygit.py` that implements two fundamental Git "plumbing" commands: `init` and `hash-object`.
The goal is to create a repository that is **binary-compatible** with official Git.

## 2. Functional Requirements
The script must support two subcommands:

### Subcommand A: `init`
* Usage: `python pygit.py init`
* **Action:** Creates a `.git` directory with the following structure:
    * `.git/objects` (directory)
    * `.git/refs` (directory)
    * `.git/HEAD` (file containing text: `ref: refs/heads/master\n`)

### Subcommand B: `hash-object`
* Usage: `python pygit.py hash-object <filename>`
* **Action:**
    1.  Reads the file content.
    2.  Calculates the **SHA-1 hash** of the content *including the Git header*.
    3.  Compresses the content using **zlib**.
    4.  Writes the compressed data to `.git/objects/aa/bbbb...` (where `aa` is the first 2 chars of the hash, and `bbbb...` is the rest).
    5.  Prints the SHA-1 hash to `stdout`.

## 3. The Core Challenge (The Git Object Format)
You cannot just hash the file content. Git blobs have a specific header format:
`blob <content_length>\0<content>`

* **Step 1:** Construct the payload: `b"blob " + str(len(content)).encode() + b"\0" + content`
* **Step 2:** SHA-1 hash *that* payload (this is the Object ID).
* **Step 3:** zlib compress *that* payload.
* **Step 4:** Save it.

## 4. Acceptance Criteria
1.  Run `python pygit.py init`.
2.  Create a file `hello.txt` containing `test content`.
3.  Run `python pygit.py hash-object hello.txt`. Output should be the hash.
4.  **Verification:** Run the *real* `git fsck` in that directory. If your script worked, Git will recognize the object and not complain about corruption.
    * *Alternative:* Run `git cat-file -p <hash>` using real Git. It should print `test content`.

## 5. Research Instructions
1.  Verify the exact header format for a Git "blob".
2.  Research `zlib.compress` vs `zlib.decompress`.
3.  Understand how the SHA-1 hash is split into directory name (2 chars) and filename (38 chars).
