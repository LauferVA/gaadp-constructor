# Project Specification: Dead Link Crawler

## 1. Objective
Create a multi-threaded web crawler `link_checker.py` that starts at a given URL, traverses all internal links, and reports any broken links (404/500 errors).

## 2. Core Logic (The Graph)
The web is a directed graph. You must perform a **Breadth-First Search (BFS)** or DFS.
* **Scope:** Only follow links that belong to the same *domain* (e.g., if starting at `example.com`, do not crawl `facebook.com`, just check if it returns 200 OK).

## 3. Requirements
* **Input:** Start URL (e.g., `https://crawler-test.com`).
* **Concurrency:** Use `concurrent.futures.ThreadPoolExecutor` to check multiple links at once.
* **State Management:**
    * `visited`: A Set of URLs to ensure we don't check the same page twice.
    * `queue`: URLs to be crawled.
* **Parsing:** Use `BeautifulSoup` (allowed here) to extract `<a href="...">` tags.

## 4. Output
* A report printed to stdout:
    ```text
    Scanning https://example.com...
    [OK] https://example.com/about
    [404] https://example.com/broken-page (Linked from /contact)
    ...
    Found 1 broken links.
    ```

## 5. Constraints
* **Politeness:** Limit to max 10 threads.
* **Recursion:** Do not go deeper than 3 clicks from the start page (Depth limit) OR limit total pages to 50.

## 6. Acceptance Criteria
* Must correctly identify a 404 link.
* Must NOT get stuck in a loop between Page A and Page B.
* Must handle relative URLs (`/about`) by resolving them against the base URL.

## 7. Research Instructions
1.  Research `urllib.parse.urljoin` for resolving relative links.
2.  How to normalize URLs? (`example.com/` and `example.com` are the same).
