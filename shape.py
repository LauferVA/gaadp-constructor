from abc import ABC, abstractmethod

class Shape(ABC):
    """Abstract base class representing a geometric shape."""

    @abstractmethod
    def area(self):
        """Calculate and return the area of the shape.

        This method must be implemented by all subclasses.

        Returns:
            float: The area of the shape.
        """
        pass
