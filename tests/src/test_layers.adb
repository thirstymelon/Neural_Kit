with Ada.Assertions;
with Neural_Kit.Layer;  use Neural_Kit.Layer;
with Neural_Kit.Math;   use Neural_Kit.Math;
with Neural_Kit.Random; use Neural_Kit.Random;
with Neural_Kit.Types;  use Neural_Kit.Types;
with Test_Framework;    use Test_Framework;

package body Test_Layers is

   procedure Run is
      --  Layer with 2 output neurons and 3 input features
      Weights_2x3 : constant Matrix (1 .. 2, 1 .. 3) :=
        ((1.0, 2.0, 3.0),
         (4.0, 5.0, 6.0));
      Biases_2    : constant Vector (1 .. 2) := (0.5, -0.5);

      --  Input vector of length 3: [1.0, 0.0, 1.0]
      Input_3     : constant Vector (1 .. 3) := (1.0, 0.0, 1.0);

      Gen : Generator;
   begin
      Start_Suite ("Layer and Multi-Neuron Architecture");
      Set_Seed (Gen, 42);

      --  1. Linear Layer Forward Pass
      --  Neuron 1: 1*1 + 2*0 + 3*1 + 0.5 = 4.5
      --  Neuron 2: 4*1 + 5*0 + 6*1 - 0.5 = 9.5
      declare
         L_Linear : Layer :=
           New_Layer (Weights_2x3, Biases_2, Activation => Linear);
         Out_Vec  : constant Vector := L_Linear.Predict (Input_3);
      begin
         Assert_Almost_Equal
           (Out_Vec (1), 4.5, Message => "Linear layer output neuron 1 = 4.5");
         Assert_Almost_Equal
           (Out_Vec (2), 9.5, Message => "Linear layer output neuron 2 = 9.5");
         Assert_True
           (L_Linear.Pre_Activation (1) = 4.5
            and then L_Linear.Pre_Activation (2) = 9.5,
            Message => "Pre_Activation correctly tracked in Layer");
      end;

      --  2. Sigmoid Layer Forward Pass
      declare
         L_Sig : Layer :=
           New_Layer (Weights_2x3, Biases_2, Activation => Sigmoid);
         Out_Vec : constant Vector := L_Sig.Predict (Input_3);
      begin
         Assert_Almost_Equal
           (Out_Vec (1),
            Sigmoid (4.5),
            Message => "Sigmoid layer neuron 1");
         Assert_Almost_Equal
           (Out_Vec (2),
            Sigmoid (9.5),
            Message => "Sigmoid layer neuron 2");
      end;

      --  3. ReLU Layer Forward Pass
      declare
         Negative_Biases : constant Vector (1 .. 2) := (-10.0, 0.0);
         L_ReLU : Layer :=
           New_Layer (Weights_2x3, Negative_Biases, Activation => ReLU);
         Out_Vec : constant Vector := L_ReLU.Predict (Input_3);
      begin
         --  Neuron 1: 4.0 - 10.0 = -6.0 -> ReLU(-6.0) = 0.0
         --  Neuron 2: 10.0 + 0.0 = 10.0 -> ReLU(10.0) = 10.0
         Assert_Almost_Equal
           (Out_Vec (1), 0.0, Message => "ReLU layer clamped negative to 0.0");
         Assert_Almost_Equal
           (Out_Vec (2), 10.0, Message => "ReLU layer passes positive 10.0");
      end;

      --  4. Tanh Layer Forward Pass
      declare
         L_Tanh : Layer :=
           New_Layer (Weights_2x3, Biases_2, Activation => Tanh);
         Out_Vec : constant Vector := L_Tanh.Predict (Input_3);
      begin
         Assert_Almost_Equal
           (Out_Vec (1), Tanh (4.5), Message => "Tanh layer neuron 1");
         Assert_Almost_Equal
           (Out_Vec (2), Tanh (9.5), Message => "Tanh layer neuron 2");
      end;

      --  5. Softmax Layer Forward Pass
      declare
         L_Softmax : Layer :=
           New_Layer (Weights_2x3, Biases_2, Activation => Softmax);
         Out_Vec : constant Vector := L_Softmax.Predict (Input_3);
         Sum_P   : constant Scalar := Out_Vec (1) + Out_Vec (2);
      begin
         Assert_Almost_Equal
           (Sum_P, 1.0, Tolerance => 1.0e-4, Message => "Softmax sum = 1.0");
         Assert_True
           (Out_Vec (2) > Out_Vec (1),
            Message => "Softmax neuron 2 > neuron 1");
      end;

      --  6. Xavier Initialized Layer
      declare
         L_Xavier : Layer :=
           New_Layer
             (Input_Dimension  => 4,
              Output_Dimension => 8,
              Gen              => Gen,
              Activation       => Sigmoid,
              Init_Strategy    => Xavier);
         In_4     : constant Vector (1 .. 4) := (0.5, -0.5, 1.0, 0.0);
         Out_8    : constant Vector := L_Xavier.Predict (In_4);
      begin
         Assert_True
           (Out_8'Length = 8,
            Message => "Xavier layer produces correct output dimension (8)");
         for I in Out_8'Range loop
            Assert_True
              (Out_8 (I) > 0.0 and then Out_8 (I) < 1.0,
               Message => "Sigmoid output bounded in (0, 1)");
         end loop;
      end;

      --  7. He Initialized Layer
      declare
         L_He : Layer :=
           New_Layer
             (Input_Dimension  => 10,
              Output_Dimension => 5,
              Gen              => Gen,
              Activation       => ReLU,
              Init_Strategy    => He);
         In_10 : constant Vector (1 .. 10) := (others => 0.5);
         Out_5 : constant Vector := L_He.Predict (In_10);
      begin
         Assert_True
           (Out_5'Length = 5,
            Message => "He layer produces correct output dimension (5)");
      end;

      --  8. Zeros Initialized Layer
      declare
         L_Zeros : Layer :=
           New_Layer_Zeros
             (Input_Dimension  => 3,
              Output_Dimension => 2,
              Activation       => Linear);
         Out_2 : constant Vector := L_Zeros.Predict (Input_3);
      begin
         Assert_Almost_Equal
           (Out_2 (1), 0.0, Message => "Zero-init layer output 1 = 0.0");
         Assert_Almost_Equal
           (Out_2 (2), 0.0, Message => "Zero-init layer output 2 = 0.0");
      end;

      --  9. Backward Pass and Gradient Propagation
      declare
         L_Back : Layer :=
           New_Layer (Weights_2x3, Biases_2, Activation => Linear);
         Out_Grad : constant Vector (1 .. 2) := (1.0, 2.0);
         In_Grad  : Vector (1 .. 3);
      begin
         L_Back.Forward (Input_3);
         L_Back.Backward
           (Output_Grad      => Out_Grad,
            Input_Activation => Input_3,
            Input_Grad       => In_Grad);

         --  Delta for linear: Out_Grad * 1.0 = (1.0, 2.0)
         --  Weight_Grad = Outer_Product(Delta, Input_3)
         --  Row 1: 1.0 * [1, 0, 1] = [1.0, 0.0, 1.0]
         --  Row 2: 2.0 * [1, 0, 1] = [2.0, 0.0, 2.0]
         Assert_Almost_Equal
           (L_Back.Weight_Gradients (1, 1), 1.0, Message => "Weight grad (1,1)");
         Assert_Almost_Equal
           (L_Back.Weight_Gradients (1, 2), 0.0, Message => "Weight grad (1,2)");
         Assert_Almost_Equal
           (L_Back.Weight_Gradients (2, 3), 2.0, Message => "Weight grad (2,3)");
         Assert_Almost_Equal
           (L_Back.Bias_Gradients (1), 1.0, Message => "Bias grad 1");
         Assert_Almost_Equal
           (L_Back.Bias_Gradients (2), 2.0, Message => "Bias grad 2");

         --  In_Grad = W^T * Delta
         --  Col 1: 1*1.0 + 4*2.0 = 9.0
         --  Col 2: 2*1.0 + 5*2.0 = 12.0
         --  Col 3: 3*1.0 + 6*2.0 = 15.0
         Assert_Almost_Equal
           (In_Grad (1), 9.0, Message => "Input grad component 1 = 9.0");
         Assert_Almost_Equal
           (In_Grad (2), 12.0, Message => "Input grad component 2 = 12.0");
         Assert_Almost_Equal
           (In_Grad (3), 15.0, Message => "Input grad component 3 = 15.0");

         --  Update Layer
         L_Back.Update (Learning_Rate => 0.1);
         --  W(1,1) was 1.0, grad is 1.0 -> new W(1,1) = 1.0 - 0.1*1.0 = 0.9
         Assert_Almost_Equal
           (L_Back.Weights (1, 1),
            0.9,
            Message => "Weights updated correctly");
         --  Gradients must be zeroed after update
         Assert_Almost_Equal
           (L_Back.Weight_Gradients (1, 1),
            0.0,
            Message => "Weight gradients zeroed after update");
      end;

      --  10. Dimension Mismatch Validation
      declare
         L_Dim : Layer :=
           New_Layer_Zeros
             (Input_Dimension  => 4,
              Output_Dimension => 2,
              Activation       => Linear);
         Bad_In : constant Vector (1 .. 3) := (1.0, 2.0, 3.0);
         Failed : Boolean := False;
      begin
         begin
            declare
               Dummy : constant Vector := L_Dim.Predict (Bad_In);
               pragma Unreferenced (Dummy);
            begin
               null;
            end;
         exception
            when Ada.Assertions.Assertion_Error | Dimension_Error =>
               Failed := True;
         end;
         Assert_True
           (Failed,
            Message => "Layer input dimension mismatch raises error");
      end;

      End_Suite;
   end Run;

end Test_Layers;
