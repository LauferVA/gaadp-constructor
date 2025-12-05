import unittest

class Animal:
    def __init__(self, name):
        self.name = name
    
    def make_sound(self):
        return "Generic animal sound"

class Mammal(Animal):
    def __init__(self, name, fur_color):
        super().__init__(name)
        self.fur_color = fur_color
    
    def make_sound(self):
        return f"{self.name} makes a mammalian sound"
    
    def groom(self):
        return f"Grooming {self.name}'s {self.fur_color} fur"

class Dog(Mammal):
    def __init__(self, name, fur_color, breed):
        super().__init__(name, fur_color)
        self.breed = breed
    
    def make_sound(self):
        return "Woof!"
    
    def fetch(self):
        return f"{self.name} the {self.breed} is fetching"

class TestAnimalInheritance(unittest.TestCase):
    def test_base_animal(self):
        animal = Animal("Generic")
        self.assertEqual(animal.name, "Generic")
        self.assertEqual(animal.make_sound(), "Generic animal sound")
    
    def test_mammal_inheritance(self):
        mammal = Mammal("Lion", "golden")
        self.assertEqual(mammal.name, "Lion")
        self.assertEqual(mammal.fur_color, "golden")
        self.assertEqual(mammal.make_sound(), "Lion makes a mammalian sound")
        self.assertEqual(mammal.groom(), "Grooming Lion's golden fur")
    
    def test_dog_inheritance(self):
        dog = Dog("Buddy", "brown", "Labrador")
        self.assertEqual(dog.name, "Buddy")
        self.assertEqual(dog.fur_color, "brown")
        self.assertEqual(dog.breed, "Labrador")
        self.assertEqual(dog.make_sound(), "Woof!")
        self.assertEqual(dog.groom(), "Grooming Buddy's brown fur")
        self.assertEqual(dog.fetch(), "Buddy the Labrador is fetching")

if __name__ == '__main__':
    unittest.main()