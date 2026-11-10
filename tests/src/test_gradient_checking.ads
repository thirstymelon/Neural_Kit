with Neural_Kit.Types; use Neural_Kit.Types;

package Test_Gradient_Checking is

   --  Type for a scalar-to-scalar differentiable function
   type Scalar_Func is not null access function (X : Scalar) return Scalar;

   --  Computes the central finite-difference numerical derivative:
   --  f'(x) ~ (f(x + eps) - f(x - eps)) / (2 * eps)
   function Numerical_Derivative
     (F   : Scalar_Func;
      X   : Scalar;
      Eps : Scalar := 1.0e-4) return Scalar;

   procedure Run;

end Test_Gradient_Checking;

