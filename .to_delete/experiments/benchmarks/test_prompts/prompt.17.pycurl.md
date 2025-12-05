# Project Specification: "PyCurl" (Raw HTTP Client)

## 1. Objective
Create a Python script `pycurl.py` that makes an HTTP GET request to a specified URL and prints the response body.
**Crucially**, you must implement the HTTP protocol manually over raw TCP sockets.

## 2. Constraints
* **Forbidden Libraries:** `requests`, `urllib`, `http.client`.
* **Allowed Libraries:** `socket`, `ssl` (for HTTPS), `sys`.
* **Protocol:** HTTP/1.1.

## 3. Functional Requirements
* **Input:** A URL argument (e.g., `python pycurl.py https://example.com/api`).
* **Parsing:** The script must parse the URL to extract:
    * Scheme (`http` vs `https`) -> Determines port (80 vs 443) and whether to wrap in SSL.
    * Host (`example.com`) -> Needed for the socket connection AND the `Host:` header.
    * Path (`/api`) -> Needed for the GET line.
* **The Request:** Construct the raw HTTP request string.
    * Must include `GET <path> HTTP/1.1`.
    * Must include `Host: <hostname>`.
    * Must end headers with a double CRLF (`\r\n\r\n`).
* **The Response:**
    * Read the response from the socket.
    * Separate Headers from Body (split by the first `\r\n\r\n`).
    * Print **only** the Body to stdout.

## 4. Edge Cases
* **HTTPS:** If the scheme is `https`, you must wrap the socket using `ssl.create_default_context().wrap_socket(...)`.
* **Redirects:** Ignore them. (Implementation of `301/302` handling is out of scope for this version; just print the 301 body).
* **Chunked Encoding:** (Bonus) If the server sends `Transfer-Encoding: chunked`, standard reading fails. *For this task, assume standard `Content-Length` or read until connection close.*

## 5. Acceptance Criteria
1.  `python pycurl.py http://httpbin.org/get` -> Prints the JSON response.
2.  `python pycurl.py https://example.com` -> Prints the HTML (handling SSL correctly).

## 6. Research Instructions
1.  Look up the exact syntax for a minimal HTTP/1.1 Request.
2.  Understand why `Host` header is required (Virtual Hosting).
3.  Research `ssl.wrap_socket` vs `ssl.SSLContext.wrap_socket` (modern usage).
