with Neural_Kit.Math; use Neural_Kit.Math;
with Test_Framework;  use Test_Framework;

package body Test_Gradient_Checking is

   function Numerical_Derivative
     (F   : Scalar_Func;
      X   : Scalar;
      Eps : Scalar := 1.0e-4) return Scalar
   is
   begin
      return (F (X + Eps) - F (X - Eps)) / (2.0 * Eps);
   end Numerical_Derivative;

   procedure Run is
      Test_Points : constant array (1 .. 5) of Scalar :=
        (-2.0, -0.5, 0.0, 0.5, 2.0);
   begin
      Start_Suite ("Numerical Gradient Checking");

      --  Sigmoid Derivative Verification
      for I in Test_Points'Range loop
         declare
            X          : constant Scalar := Test_Points (I);
            Sig_Out    : constant Scalar := Sigmoid (X);
            Analytical : constant Scalar :=
              Sigmoid_Derivative_From_Output (Sig_Out);
            Numerical  : constant Scalar :=
              Numerical_Derivative (Sigmoid'Access, X);
         begin
            Assert_Almost_Equal
              (Analytical,
               Numerical,
               Tolerance => 1.0e-3,
               Message   => "Sigmoid derivative at X=" & Scalar'Image (X));
         end;
      end loop;

      --  Tanh Derivative Verification
      for I in Test_Points'Range loop
         declare
            X          : constant Scalar := Test_Points (I);
            Tanh_Out   : constant Scalar := Tanh (X);
            Analytical : constant Scalar :=
              Tanh_Derivative_From_Output (Tanh_Out);
            Numerical  : constant Scalar :=
              Numerical_Derivative (Tanh'Access, X);
         begin
            Assert_Almost_Equal
              (Analytical,
               Numerical,
               Tolerance => 1.0e-3,
               Message   => "Tanh derivative at X=" & Scalar'Image (X));
         end;
      end loop;

      --  ReLU Derivative Verification for non-zero points
      declare
         X_Pos : constant Scalar := 1.5;
         ReLU_Out : constant Scalar := ReLU (X_Pos);
         Analytical : constant Scalar :=
           ReLU_Derivative_From_Output (ReLU_Out);
         Numerical : constant Scalar :=
           Numerical_Derivative (ReLU'Access, X_Pos);
      begin
         Assert_Almost_Equal
           (Analytical,
            Numerical,
            Tolerance => 1.0e-3,
            Message   => "ReLU derivative at positive X");
      end;

      End_Suite;
   end Run;

end Test_Gradient_Checking;
