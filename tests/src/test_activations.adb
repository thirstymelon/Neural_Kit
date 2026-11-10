with Neural_Kit.Math;  use Neural_Kit.Math;
with Neural_Kit.Types; use Neural_Kit.Types;
with Test_Framework;   use Test_Framework;

package body Test_Activations is

   procedure Run is
   begin
      Start_Suite ("Activation Functions");

      --  Sigmoid
      Assert_Almost_Equal (Sigmoid (0.0), 0.5, Message => "Sigmoid(0) = 0.5");
      Assert_Almost_Equal
        (Sigmoid (100.0), 1.0, Message => "Sigmoid(100) = 1.0 (no overflow)");
      Assert_Almost_Equal
        (Sigmoid (-100.0), 0.0, Message => "Sigmoid(-100) = 0.0 (no overflow)");

      --  Sigmoid Derivative
      Assert_Almost_Equal
        (Sigmoid_Derivative_From_Output (0.5),
         0.25,
         Message => "Sigmoid_Derivative(0.5) = 0.25");

      --  ReLU
      Assert_Almost_Equal (ReLU (-10.0), 0.0, Message => "ReLU(-10) = 0.0");
      Assert_Almost_Equal (ReLU (0.0), 0.0, Message => "ReLU(0) = 0.0");
      Assert_Almost_Equal (ReLU (4.5), 4.5, Message => "ReLU(4.5) = 4.5");

      --  ReLU Derivative
      Assert_Almost_Equal
        (ReLU_Derivative_From_Output (0.0),
         0.0,
         Message => "ReLU_Derivative(0) = 0.0");
      Assert_Almost_Equal
        (ReLU_Derivative_From_Output (3.0),
         1.0,
         Message => "ReLU_Derivative(3) = 1.0");

      --  Tanh
      Assert_Almost_Equal (Tanh (0.0), 0.0, Message => "Tanh(0) = 0.0");
      Assert_Almost_Equal
        (Tanh_Derivative_From_Output (0.0),
         1.0,
         Message => "Tanh_Derivative(0) = 1.0");
      Assert_Almost_Equal
        (Tanh_Derivative_From_Output (1.0),
         0.0,
         Message => "Tanh_Derivative(1) = 0.0");

      --  Softmax Normalization & Stability
      declare
         Input : constant Vector (1 .. 3) := (1.0, 2.0, 3.0);
         Sm    : constant Vector := Softmax (Input);
         Total : Scalar := 0.0;
      begin
         for I in Sm'Range loop
            Total := Total + Sm (I);
            Assert_True
              (Sm (I) > 0.0, Message => "Softmax component must be positive");
         end loop;
         Assert_Almost_Equal
           (Total, 1.0, Tolerance => 1.0e-4, Message => "Softmax sum = 1.0");
         Assert_True
           (Sm (3) > Sm (2) and then Sm (2) > Sm (1),
            Message => "Softmax preserves order");
      end;

      --  Softmax with Extreme Values (avoiding exp overflow)
      declare
         Extreme_Input : constant Vector (1 .. 3) :=
           (1000.0, 1001.0, 1002.0);
         Sm            : constant Vector := Softmax (Extreme_Input);
         Total         : Scalar := 0.0;
      begin
         for I in Sm'Range loop
            Total := Total + Sm (I);
         end loop;
         Assert_Almost_Equal
           (Total,
            1.0,
            Tolerance => 1.0e-4,
            Message => "Softmax with 1000+ extreme values sums to 1.0");
      end;

      End_Suite;
   end Run;

end Test_Activations;

