with Neural_Kit.Math; use Neural_Kit.Math;

package body Neural_Kit.Neuron is

   function New_Neuron (Weights : Vector; Bias : Scalar) return Neuron is
      Norm_Weights : Vector (1 .. Weights'Length);
      W_Idx        : Positive := Weights'First;
   begin
      if Weights'Length = 0 then
         raise Dimension_Error with "Neuron weights cannot be empty";
      end if;

      for I in 1 .. Weights'Length loop
         Norm_Weights (I) := Weights (W_Idx);
         W_Idx := W_Idx + 1;
      end loop;

      return
        (Input_Dimension => Weights'Length,
         Weights         => Norm_Weights,
         Bias            => Bias,
         Pre_Activation  => 0.0,
         Output          => 0.0);
   end New_Neuron;

   procedure Forward (Self : in out Neuron; Input : Vector) is
   begin
      if Input'Length /= Self.Input_Dimension then
         raise Dimension_Error with "Input dimension does not match neuron";
      end if;

      Self.Pre_Activation := Dot (Self.Weights, Input) + Self.Bias;
   end Forward;

   procedure Activate (Self : in out Neuron) is
   begin
      Self.Output := Sigmoid (Self.Pre_Activation);
   end Activate;

   function Loss (Self : Neuron; Target : Scalar) return Scalar is
   begin
      return Binary_Cross_Entropy (Self.Output, Target);
   end Loss;

   procedure Update
     (Self          : in out Neuron;
      Input         : Vector;
      Target        : Scalar;
      Learning_Rate : Scalar)
   is
   begin
      if Input'Length /= Self.Input_Dimension then
         raise Dimension_Error with "Input dimension does not match neuron";
      end if;

      if Target < 0.0 or else Target > 1.0 then
         raise Numerical_Error with "Target must be within [0.0, 1.0]";
      end if;

      declare
         Error  : constant Scalar := Self.Output - Target;
         Grad_V : constant Vector := Scale (Input, Learning_Rate * Error);
      begin
         Self.Weights := Subtract (Self.Weights, Grad_V);
         Self.Bias    := Self.Bias - Learning_Rate * Error;
      end;
   end Update;

   function Predict
     (Self : in out Neuron; Input : Vector) return Scalar is
   begin
      Self.Forward (Input);
      Self.Activate;
      return Self.Output;
   end Predict;

   function Classify_Binary
     (Self : in out Neuron; Input : Vector) return Boolean
   is
      Prediction : constant Scalar := Self.Predict (Input);
   begin
      return Prediction >= 0.5;
   end Classify_Binary;

end Neural_Kit.Neuron;
