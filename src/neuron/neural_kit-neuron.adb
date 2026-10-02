with Ada.Numerics.Elementary_Functions;

package body Neural_Kit.Neuron is

   function New_Neuron (Weights : Weights_T; Bias : Bias_T) return Neuron_T is
   begin
      return
        (Weights        => Weights,
         Bias           => Bias,
         Pre_Activation => 0.0,
         Output         => 0.0);
   end New_Neuron;

   procedure Forward (Self : in out Neuron_T; Input : Features_T) is
      Sum : Scalar_T := Self.Bias;
   begin
      for I in Input'Range loop
         Sum := Sum + Self.Weights (I) * Input (I);
      end loop;

      Self.Pre_Activation := Sum;
   end Forward;

   procedure Activate (Self : in out Neuron_T) is
      use Ada.Numerics.Elementary_Functions;
   begin
      Self.Output := 1.0 / (1.0 + Exp (-Self.Pre_Activation));
   end Activate;

   function Loss (Self : Neuron_T; Target : Scalar_T) return Scalar_T is
      use Ada.Numerics.Elementary_Functions;

      Safe_Output : constant Scalar_T :=
        Scalar_T'Max (0.000001, Scalar_T'Min (0.999999, Self.Output));
   begin
      return
        -(Target * Log (Safe_Output)
          + (1.0 - Target) * Log (1.0 - Safe_Output));
   end Loss;

   procedure Update
     (Self          : in out Neuron_T;
      Input         : Features_T;
      Target        : Scalar_T;
      Learning_Rate : Scalar_T)
   is
      Error : constant Scalar_T := Self.Output - Target;
   begin
      for I in Input'Range loop
         Self.Weights (I) :=
           Self.Weights (I) - Learning_Rate * Error * Input (I);
      end loop;

      Self.Bias := Self.Bias - Learning_Rate * Error;
   end Update;

   function Predict
     (Self : in out Neuron_T; Input : Features_T) return Scalar_T is
   begin
      Self.Forward (Input);
      Self.Activate;
      return Self.Output;
   end Predict;

   function Classify_Binary
     (Self : in out Neuron_T; Input : Features_T) return Boolean
   is
      Prediction : constant Scalar_T := Self.Predict (Input);
   begin
      return Prediction >= 0.5;
   end Classify_Binary;

end Neural_Kit.Neuron;
