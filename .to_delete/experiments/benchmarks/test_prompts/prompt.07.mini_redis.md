# Project Specification: Mini-Redis (Key-Value Store)

## 1. Objective
Build a lightweight, in-memory key-value store named `kv_server.py` that listens on a TCP port and handles concurrent clients.

## 2. Protocol
The server must accept text-based commands terminated by newline `\n`.
* `SET key value` -> Responds `OK`
* `GET key` -> Responds with `value` or `(nil)` if not found.
* `DEL key` -> Responds `1` (if deleted) or `0` (if not exists).

## 3. Requirements
* **Concurrency:** Use `threading` or `socketserver.ThreadingTCPServer` to handle multiple clients simultaneously. One blocked client must not stop others.
* **Storage:** A global Python dictionary protected by a `threading.Lock` to prevent race conditions.
* **Persistence:** Implement a `SAVE` command that dumps the dict to `dump.json` on disk.

## 4. Acceptance Criteria
* Launch server.
* Client A connects and runs `SET foo bar`.
* Client B connects and runs `GET foo`. Response must be `bar`.
* Client A runs `SAVE`. File `dump.json` appears.

## 5. Research Instructions
1.  Review Python's `socketserver` module for easy threading implementation.
2.  Identify where to place the `Lock` (around reads? writes? both?).
