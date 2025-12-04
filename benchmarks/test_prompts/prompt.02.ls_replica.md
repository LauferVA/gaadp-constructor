# Project Specification: Cleanroom Re-implementation of `ls -l`

## 1. Objective
Write a Python script named `pyls.py` that replicates the functionality of the Unix command `ls -l` for a given directory. The output format must match the standard POSIX output exactly.

## 2. Constraints & Rules
* **NO Subprocesses:** You may not use `subprocess`, `os.system`, or any method to call the native `ls` command. You must build the output using Python's `os`, `stat`, `pwd`, `grp`, and `datetime` libraries.
* **Target OS:** Linux/macOS (Unix-like environment).
* **Formatting:** Columns must align correctly (padding based on the longest item in the column).

## 3. Functional Requirements
* **Arguments:** Accepts one optional argument: the path to the directory. Defaults to `.` (current directory).
* **Output Columns:** The script must print the following columns in order:
    1.  **File Permissions:** 10 characters (e.g., `drwxr-xr-x` or `-rw-r--r--`).
    2.  **Hard Links:** Integer count.
    3.  **Owner Name:** String (lookup user ID).
    4.  **Group Name:** String (lookup group ID).
    5.  **File Size:** In bytes.
    6.  **Modification Date:** Format `Mmm dd HH:MM` (e.g., `Nov 20 14:30`). Note: If the file is >6 months old, standard `ls` often changes the time to the year. For this task, keep it simple: `Mmm dd HH:MM`.
    7.  **Filename:** Name of the file/directory.

## 4. The Core Challenge (Research Phase Focus)
The agent must correctly implement the **Permission Bitmasking** logic.
* Input: `st_mode` integer from `os.stat()`.
* Output: `rwx` string representation.
* Logic: It must correctly identify directory (`d`), file (`-`), and permissions for User, Group, and Others.

## 5. Acceptance Criteria
**Command:**
```bash
python pyls.py ./test_folder
```

**Expected Output (Exact alignment required):**
```
drwxr-xr-x  3 vincent  staff    96 Dec 03 10:00 .
drwxr-xr-x  5 vincent  staff   160 Dec 03 09:55 ..
-rw-r--r--  1 vincent  staff  2048 Dec 02 14:00 data.txt
-rwxr-x---  1 vincent  staff   500 Nov 20 08:30 script.sh
```

## 6. Research Instructions
Before coding, investigate:
1. How to retrieve the strictly formatted permission string from `os.stat(path).st_mode` using bitwise operators (`&`) and the `stat` module constants.
2. How to convert UID/GID integers to string names using the `pwd` and `grp` modules.
