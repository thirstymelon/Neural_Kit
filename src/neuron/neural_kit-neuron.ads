with Neural_Kit.Types; use Neural_Kit.Types;

package Neural_Kit.Neuron is

   type Neuron (Input_Dimension : Positive) is tagged record
      Weights        : Vector (1 .. Input_Dimension);
      Bias           : Scalar := 0.0;
      Pre_Activation : Scalar := 0.0;
      Output         : Scalar := 0.0;
   end record;

   subtype Neuron_T is Neuron;

   function New_Neuron (Weights : Vector; Bias : Scalar) return Neuron
     with Pre => Weights'Length > 0;

   procedure Forward (Self : in out Neuron; Input : Vector)
     with Pre => Input'Length = Self.Input_Dimension;

   procedure Activate (Self : in out Neuron);

   function Loss (Self : Neuron; Target : Scalar) return Scalar
     with Pre => Target >= 0.0 and then Target <= 1.0;

   procedure Update
     (Self          : in out Neuron;
      Input         : Vector;
      Target        : Scalar;
      Learning_Rate : Scalar)
     with Pre => Input'Length = Self.Input_Dimension
                 and then Target >= 0.0
                 and then Target <= 1.0;

   function Predict
     (Self : in out Neuron; Input : Vector) return Scalar
     with Pre => Input'Length = Self.Input_Dimension;

   function Classify_Binary
     (Self : in out Neuron; Input : Vector) return Boolean
     with Pre => Input'Length = Self.Input_Dimension;

end Neural_Kit.Neuron;
