from abc import ABC, abstractmethod
from typing import Tuple, Optional

class GameObject(ABC):
    """
    Abstract base class representing a fundamental game object.
    
    Attributes:
        position (Tuple[float, float]): The x, y coordinates of the object in game space.
        velocity (Tuple[float, float]): The x, y velocity vector of the object.
        active (bool): Indicates whether the game object is currently active in the game.
    """
    
    def __init__(
        self, 
        position: Tuple[float, float] = (0.0, 0.0), 
        velocity: Tuple[float, float] = (0.0, 0.0)
    ):
        """
        Initialize a game object with position and velocity.
        
        Args:
            position (Tuple[float, float], optional): Initial position. Defaults to (0.0, 0.0).
            velocity (Tuple[float, float], optional): Initial velocity. Defaults to (0.0, 0.0).
        """
        self._position = list(position)  # Use list for mutability
        self._velocity = list(velocity)  # Use list for mutability
        self._active = True
    
    @property
    def position(self) -> Tuple[float, float]:
        """
        Get the current position of the game object.
        
        Returns:
            Tuple[float, float]: Current x, y coordinates.
        """
        return tuple(self._position)
    
    @position.setter
    def position(self, new_position: Tuple[float, float]):
        """
        Set the position of the game object.
        
        Args:
            new_position (Tuple[float, float]): New x, y coordinates.
        """
        self._position = list(new_position)
    
    @property
    def velocity(self) -> Tuple[float, float]:
        """
        Get the current velocity of the game object.
        
        Returns:
            Tuple[float, float]: Current x, y velocity vector.
        """
        return tuple(self._velocity)
    
    @velocity.setter
    def velocity(self, new_velocity: Tuple[float, float]):
        """
        Set the velocity of the game object.
        
        Args:
            new_velocity (Tuple[float, float]): New x, y velocity vector.
        """
        self._velocity = list(new_velocity)
    
    @property
    def active(self) -> bool:
        """
        Check if the game object is active.
        
        Returns:
            bool: True if the object is active, False otherwise.
        """
        return self._active
    
    def activate(self):
        """
        Activate the game object.
        """
        self._active = True
    
    def deactivate(self):
        """
        Deactivate the game object.
        """
        self._active = False
    
    @abstractmethod
    def update(self, delta_time: float):
        """
        Update the game object's state.
        
        This method should be implemented by subclasses to define 
        specific update behavior.
        
        Args:
            delta_time (float): Time elapsed since the last update.
        """
        pass
    
    @abstractmethod
    def render(self, renderer):
        """
        Render the game object.
        
        This method should be implemented by subclasses to define 
        specific rendering behavior.
        
        Args:
            renderer: The rendering context or system.
        """
        pass