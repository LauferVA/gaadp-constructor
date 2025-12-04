# Project Specification: Async Round-Robin Load Balancer

## 1. Objective
Create a lightweight TCP load balancer named `balancer.py` using Python's `asyncio` library. It must sit between a client and 3 backend "mock" servers, distributing requests in a strict Round-Robin fashion.

## 2. Architecture
* **Frontend:** Listens on port `8000`.
* **Backends:** Forward traffic to ports `9001`, `9002`, `9003` (assumed to be running on localhost).
* **Protocol:** Raw TCP. The balancer blindly forwards bytes from Client -> Backend and Backend -> Client.

## 3. Constraints (Strict)
* **No External Frameworks:** You cannot use `FastAPI`, `Flask`, `Django`, or `Nginx`.
* **Pure Asyncio:** You must use `asyncio.start_server` and `asyncio.open_connection`.
* **Concurrency:** The balancer must handle multiple concurrent clients. If Client A is slow, Client B must not be blocked.

## 4. Functional Logic
1.  **Startup:** The script initiates. It initializes a counter/pointer for Round Robin (0, 1, 2).
2.  **Connection:** When a client connects to `8000`:
    * Select the next backend port (e.g., `9001`).
    * Open a connection to that backend.
    * **Pipe Data:** Asynchronously read from Client -> write to Backend, AND read from Backend -> write to Client.
    * **Cleanup:** When either side closes the connection, close the other side.
3.  **Rotation:** The next client gets `9002`, then `9003`, then back to `9001`.

## 5. Edge Cases (The "Gotchas")
* **Backend Down:** If a backend port (`9001`) refuses connection, the balancer should log the error and **immediately try the next backend** in the list without crashing or hanging the client.
* **Partial Reads:** TCP streams arrive in chunks. You cannot assume one `read()` equals one "message." You must pipe the stream continuously until EOF.

## 6. Acceptance Criteria
* **Test:**
    1.  Start 3 dummy servers (netcat or python simple server) on 9001-9003.
    2.  Start `balancer.py` on 8000.
    3.  Run 4 concurrent requests to port 8000.
* **Expected Result:**
    * Request 1 -> hits 9001
    * Request 2 -> hits 9002
    * Request 3 -> hits 9003
    * Request 4 -> hits 9001
* **Failure Condition:** If the script uses `time.sleep()` or blocking socket calls.

## 7. Research Phase Instructions
1.  **Streaming Data:** Research `StreamReader.read()` vs `StreamReader.readexactly()`. For a load balancer that doesn't inspect packets, which is appropriate? (Hint: just `read()` and forward chunks).
2.  **Task Management:** How do you run the "Client->Backend" and "Backend->Client" pipes simultaneously? Look up `asyncio.gather` vs `asyncio.create_task`.
