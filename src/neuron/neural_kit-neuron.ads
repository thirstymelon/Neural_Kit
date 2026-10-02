with Neural_Kit.Types; use Neural_Kit.Types;

package Neural_Kit.Neuron is

   type Neuron_T is tagged record
      Weights        : Weights_T;
      Bias           : Bias_T;
      Pre_Activation : Scalar_T := 0.0;
      Output         : Scalar_T := 0.0;
   end record;

   function New_Neuron (Weights : Weights_T; Bias : Bias_T) return Neuron_T;

   procedure Forward (Self : in out Neuron_T; Input : Features_T);

   procedure Activate (Self : in out Neuron_T);

   function Loss (Self : Neuron_T; Target : Scalar_T) return Scalar_T;

   procedure Update
     (Self          : in out Neuron_T;
      Input         : Features_T;
      Target        : Scalar_T;
      Learning_Rate : Scalar_T);

   function Predict
     (Self : in out Neuron_T; Input : Features_T) return Scalar_T;

   function Classify_Binary
     (Self : in out Neuron_T; Input : Features_T) return Boolean;

end Neural_Kit.Neuron;
