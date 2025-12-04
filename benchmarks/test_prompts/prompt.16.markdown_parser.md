# Project Specification: "PyMark" Parser

## 1. Objective
Write a Python function `parse_markdown(md_text: str) -> str` that converts a specific subset of Markdown into HTML.

## 2. Supported Syntax
1.  **Headers:** Lines starting with `# ` to `###### ` -> `<h1>` to `<h6>`.
2.  **Bold:** `**text**` -> `<strong>text</strong>`.
3.  **Italic:** `*text*` -> `<em>text</em>`.
4.  **Unordered Lists:** Lines starting with `* ` (asterisk space) -> `<ul><li>...</li></ul>`.
    * *Note:* You must detect the start/end of the list to wrap the whole group in `<ul>`.
5.  **Links:** `[text](url)` -> `<a href="url">text</a>`.

## 3. Constraints
* **No Libraries:** Do not use `markdown`, `mistune`, or any external package. Use standard `re` (RegEx).
* **Order of Operations:** You must determine the correct order to apply rules. (e.g., if you process italics before lists, `* Item` might become `<em> Item</em>` incorrectly).

## 4. Edge Cases
* **The "Star" Conflict:**
    * `*italic*` (No space after star)
    * `* List Item` (Space after star)
* **Inline vs Block:** Headers and Lists are "Block" elements (affect the whole line). Bold/Italic/Links are "Inline" (affect part of a line).
* **Mixed:** `**bold *italic* bold**` -> `<strong>bold <em>italic</em> bold</strong>`. (Optional bonus, but basic parser should at least not crash).

## 5. Acceptance Criteria
**Input:**
```markdown
# My Title
* List 1
* List 2

This is **bold** and this is *italic*.
Click [here](http://google.com).
```

**Expected Output:**
```html
<h1>My Title</h1>
<ul>
<li>List 1</li>
<li>List 2</li>
</ul>
<p>This is <strong>bold</strong> and this is <em>italic</em>.</p>
<p>Click <a href="http://google.com">here</a>.</p>
```

(Note: Wrapping plain text in `<p>` is required).

## 6. Research Instructions
1. Research the difference between "Block parsing" and "Inline parsing". Which should happen first?
2. Devise a regex strategy to distinguish `*` (list) from `*` (italic).
