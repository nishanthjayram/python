"""
    Implements functionality for expression trees and differentiation for the purpose of simulating
    autogradient.

    (Developed as part of a course on advanced Python programming.)

    Author  :   Nishanth Jayram (https://github.com/njayram44)
    Date    :   January 6, 2021
"""
import math

class Expr(object):
    """Implementation of expression trees"""

    def __init__(self, *args):
        """Initializes an expression node, with an accompanying list of child
        expressions."""
        self.children = args
        self.value = None # The value of the expression. 
        self.values = None # The values of the child expressions.
        self.gradient = 0 # The value of the gradient. 

    def op(self):
        """This operator will be implemented in subclasses. It should compute
        self.value from self.values, thus implementing the operation at the
        expression node."""
        raise NotImplementedError()
    
    def compute(self):
        """Computes the value of the expression."""
        # Since the attribute is initialized to None by default, we must
        # reinitialize it here to an empty list.
        self.values = []

        # Loops through the children and either a) appends the computed values
        # of those children if they are expressions, or b) appends the child
        # as is, since they must now be numerical values.
        for child in self.children:
            if isinstance(child, Expr): self.values.append(child.compute())
            else: self.values.append(child)
        
        # Now that all the child values have been added to the self.values list,
        # we call op on our parent expression to compute the final value, stored
        # in self.value, and return it.
        self.op()
        return self.value
    
    def compute_gradient(self, de_loss_over_de_e=1):
        """Computes the gradient."""
        # We append dL/de to the gradient of the expression.
        self.gradient += de_loss_over_de_e

        # This will acquire a list of derivatives dL/d(x_i).
        de_e_over_de_children = self.derivate()

        # Loops through the children and, if it is an expression, computes the
        # gradient by passing in the argument dL/de * de/d(x_i)
        for i in range(len(self.children)):
            child = self.children[i]
            if isinstance(child, Expr):
                child.compute_gradient(de_loss_over_de_e * de_e_over_de_children[i])

    def __repr__(self):
        """Returns string representation of expression."""
        return ("%s:%r %r (g: %r)" % (
            self.__class__.__name__, self.children, self.value, self.gradient))
        
    # Expression constructors
    def __add__(self, other):
        return Plus(self, other)

    def __radd__(self, other):
        return Plus(self, other)

    def __sub__(self, other):
        return Minus(self, other)

    def __rsub__(self, other):
        return Minus(other, self)

    def __mul__(self, other):
        return Multiply(self, other)

    def __rmul__(self, other):
        return Multiply(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def __pow__(self, other):
        return Power(self, other)

    def __rpow__(self, other):
        return Power(other, self)

    def __neg__(self):
        return Negative(self)
    
    def derivate(self):
        """This method computes the derivative of the operator at the expression
        node.  It needs to be implemented in derived classes, such as Plus, 
        Multiply, etc."""
        raise NotImplementedError()
    
    def zero_gradient(self):
        """Sets the gradient to 0, recursively for this expression
        and all its children."""
        self.gradient = 0
        for e in self.children:
            if isinstance(e, Expr):
                e.zero_gradient()

class V(Expr):
    """Implementation of variables"""

    def assign(self, v):
        """Assigns a value to the variable.  Used to fit a model, so we
        can assign the various input values to the variable."""
        self.children = [v]

    def op(self):
        self.value = self.values[0]

    def __repr__(self):
        return "Variable: " + str(self.children[0])
    
    def derivate(self):
        return [1.] # This is not really used.
        
class Plus(Expr):
    """Addition operator"""
    def op(self):
        self.value = self.values[0] + self.values[1]
    
    def derivate(self):
        return [1., 1.]

class Multiply(Expr):
    """Multiplication operator"""

    def op(self):
        self.value = self.values[0] * self.values[1]
    
    def derivate(self):
        return [self.values[1], self.values[0]]

class Minus(Expr):
    """Subtraction operator"""

    def op(self):
        self.value = self.values[0] - self.values[1]
    def derivate(self):
        return [1., -1.]

class Divide(Expr):
    """Division operator"""

    def op(self):
        self.value = self.values[0] / self.values[1]

    def derivate(self):
        return [1. / self.values[1],
                (-1.) * (self.values[0] / (self.values[1] ** 2.))]
    
class Power(Expr):
    """Exponentiation operator"""
    
    def op(self):
        self.value = self.values[0] ** self.values[1]

    def derivate(self):
        return [self.values[1] * (self.values[0] ** (self.values[1] - 1.)),
                (self.values[0] ** self.values[1]) * math.log(self.values[0])]

class Negative(Expr):
    """Negation operator"""

    def op(self):
        self.value = (-1.) * self.values[0]

    def derivate(self):
        return [-1]