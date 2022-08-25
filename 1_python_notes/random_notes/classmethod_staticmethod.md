---
title: Python Class method vs Static method in Python
---

### Python Class method (@classmethod) vs Static method (@staticmethod ) in Python

*Aug 25, 2022*

**References:**

- [tutorialspoint: class-method-vs-static-method-in-python](https://www.tutorialspoint.com/class-method-vs-static-method-in-python)
- [studytonight: python-static-methods-vs-class-method](https://www.studytonight.com/python-howtos/python-static-methods-vs-class-method)
- [pynative: python-class-method-vs-static-method-vs-instance-method](https://pynative.com/python-class-method-vs-static-method-vs-instance-method/)

**Notes:**

- both @staticmethod and @classmethod are bound to the Class.
- difference between @staticmethod and @classmethod

|  Class Method |  Static Method |
| :------------ | :------------ |
|  The class method takes cls (class) as first argument. |  The static method does not take any specific parameter. |
| @classmethod decorator is used.  |  @staticmethod decorator is used. |
| Class method can access and modify the class state.  | Static Method cannot access or modify the class state.  |
|Class methods are used for factory methods.|Static methods are used to do some utility tasks.|
| The class method takes the class as parameter to know about the state of that class.  |  Static methods do not know about class state. These methods are used to do some utility tasks by taking some parameters. |
|It can modify class-specific details.|It contains totally self-contained code.|










