with Neural_Kit.Random;
with Neural_Kit.Types; use Neural_Kit.Types;

package Neural_Kit.Layer is

   type Activation_Kind is (Linear, Sigmoid, ReLU, Tanh, Softmax);

   type Initialization_Kind is (Zeros, Uniform, Xavier, He);

   type Layer (Input_Dimension : Positive; Output_Dimension : Positive) is
     tagged record
      Weights          : Matrix (1 .. Output_Dimension, 1 .. Input_Dimension);
      Biases           : Vector (1 .. Output_Dimension);
      Pre_Activation   : Vector (1 .. Output_Dimension) := (others => 0.0);
      Output           : Vector (1 .. Output_Dimension) := (others => 0.0);
      Activation       : Activation_Kind               := Sigmoid;
      Weight_Gradients :
        Matrix (1 .. Output_Dimension, 1 .. Input_Dimension) :=
          (others => (others => 0.0));
      Bias_Gradients   : Vector (1 .. Output_Dimension) := (others => 0.0);
   end record;

   function New_Layer
     (Weights    : Matrix;
      Biases     : Vector;
      Activation : Activation_Kind := Sigmoid) return Layer
     with Pre => Weights'Length (1) = Biases'Length
          and then Weights'Length (2) > 0;

   function New_Layer
     (Input_Dimension  : Positive;
      Output_Dimension : Positive;
      Gen              : in out Random.Generator;
      Activation       : Activation_Kind     := Sigmoid;
      Init_Strategy    : Initialization_Kind := Xavier) return Layer;

   function New_Layer_Zeros
     (Input_Dimension  : Positive;
      Output_Dimension : Positive;
      Activation       : Activation_Kind := Linear) return Layer;

   procedure Forward (Self : in out Layer; Input : Vector)
     with Pre => Input'Length = Self.Input_Dimension;

   function Predict (Self : in out Layer; Input : Vector) return Vector
     with Pre  => Input'Length = Self.Input_Dimension,
          Post => Predict'Result'Length = Self.Output_Dimension
          and then Predict'Result'First = 1;

   procedure Backward
     (Self             : in out Layer;
      Output_Grad      : Vector;
      Input_Activation : Vector;
      Input_Grad       : out Vector)
     with Pre  => Output_Grad'Length = Self.Output_Dimension
          and then Input_Activation'Length = Self.Input_Dimension,
          Post => Input_Grad'Length = Self.Input_Dimension
          and then Input_Grad'First = 1;

   procedure Update
     (Self          : in out Layer;
      Learning_Rate : Scalar);

   procedure Zero_Grad (Self : in out Layer);
   procedure Zero_Gradients (Self : in out Layer);

end Neural_Kit.Layer;
