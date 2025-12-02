# GRAPH VISUALIZATION TEST
# Tests that the generated artifacts can be visualized meaningfully.
# Expected: Multiple interconnected nodes with clear relationships.

PROJECT: Graph Visualization Test
TARGET FILE(S): utils/graph_demo.py

## CONTEXT
This test creates a more complex module to generate a rich graph structure
that demonstrates the visualization capabilities. The code should create
multiple classes with relationships that will be visible in graph exports.

## OBJECTIVES
1. Create a module with a class hierarchy (3+ levels)
2. Include composition relationships (class A has-a class B)
3. Include method calls between classes
4. Include a factory pattern or similar structural pattern
5. The module should demonstrate clear architectural relationships

## CONSTRAINTS
- Keep to standard library only
- Use meaningful class and method names
- Include docstrings for documentation
- Code must be runnable (include a main block)

## EXPECTED OUTPUT
A module demonstrating patterns like:
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self): pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    def area(self):
        return self.width * self.height

class Square(Rectangle):
    def __init__(self, side):
        super().__init__(side, side)

class ShapeFactory:
    @staticmethod
    def create(shape_type, *args):
        shapes = {"rectangle": Rectangle, "square": Square}
        return shapes[shape_type](*args)

class Canvas:
    def __init__(self):
        self.shapes = []
    def add_shape(self, shape):
        self.shapes.append(shape)
    def total_area(self):
        return sum(s.area() for s in self.shapes)
```

## ACCEPTANCE CRITERIA
- [ ] Code generates valid Python
- [ ] CPG builder extracts class hierarchy
- [ ] Visualization shows inheritance relationships
- [ ] Visualization shows composition/reference relationships
- [ ] HTML export is viewable in browser

## ANTI-PATTERNS
- Do NOT use complex metaclasses
- Do NOT use multiple inheritance (keep it simple)
- Do NOT create circular dependencies
