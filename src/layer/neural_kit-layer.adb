with Neural_Kit.Math; use Neural_Kit.Math;

package body Neural_Kit.Layer is

   function New_Layer
     (Weights    : Matrix;
      Biases     : Vector;
      Activation : Activation_Kind := Sigmoid) return Layer
   is
      Out_Dim      : constant Positive := Weights'Length (1);
      In_Dim       : constant Positive := Weights'Length (2);
      Norm_Weights : Matrix (1 .. Out_Dim, 1 .. In_Dim);
      Norm_Biases  : Vector (1 .. Out_Dim);
      B_Idx        : Positive := Biases'First;
      W_Row        : Positive;
      W_Col0       : constant Positive := Weights'First (2);
   begin
      if Biases'Length /= Out_Dim then
         raise Dimension_Error
           with "Biases length must match weights row count";
      end if;

      for I in 1 .. Out_Dim loop
         Norm_Biases (I) := Biases (B_Idx);
         B_Idx := B_Idx + 1;
         W_Row := Weights'First (1) + I - 1;
         for J in 1 .. In_Dim loop
            Norm_Weights (I, J) := Weights (W_Row, W_Col0 + J - 1);
         end loop;
      end loop;

      return
        (Input_Dimension  => In_Dim,
         Output_Dimension => Out_Dim,
         Weights          => Norm_Weights,
         Biases           => Norm_Biases,
         Pre_Activation   => (others => 0.0),
         Output           => (others => 0.0),
         Activation       => Activation,
         Weight_Gradients => (others => (others => 0.0)),
         Bias_Gradients   => (others => 0.0));
   end New_Layer;

   function New_Layer
     (Input_Dimension  : Positive;
      Output_Dimension : Positive;
      Gen              : in out Random.Generator;
      Activation       : Activation_Kind     := Sigmoid;
      Init_Strategy    : Initialization_Kind := Xavier) return Layer
   is
      W_Mat : Matrix (1 .. Output_Dimension, 1 .. Input_Dimension);
      B_Vec : constant Vector := Random.Init_Zeros (Output_Dimension);
   begin
      case Init_Strategy is
         when Zeros =>
            W_Mat := Random.Init_Zeros (Output_Dimension, Input_Dimension);
         when Uniform =>
            W_Mat :=
              Random.Init_Uniform
                (Gen, Output_Dimension, Input_Dimension, -0.5, 0.5);
         when Xavier =>
            W_Mat :=
              Random.Init_Xavier
                (Gen, Output_Dimension, Input_Dimension);
         when He =>
            W_Mat :=
              Random.Init_He
                (Gen, Output_Dimension, Input_Dimension);
      end case;

      return
        (Input_Dimension  => Input_Dimension,
         Output_Dimension => Output_Dimension,
         Weights          => W_Mat,
         Biases           => B_Vec,
         Pre_Activation   => (others => 0.0),
         Output           => (others => 0.0),
         Activation       => Activation,
         Weight_Gradients => (others => (others => 0.0)),
         Bias_Gradients   => (others => 0.0));
   end New_Layer;

   function New_Layer_Zeros
     (Input_Dimension  : Positive;
      Output_Dimension : Positive;
      Activation       : Activation_Kind := Linear) return Layer
   is
   begin
      return
        (Input_Dimension  => Input_Dimension,
         Output_Dimension => Output_Dimension,
         Weights          =>
           Random.Init_Zeros (Output_Dimension, Input_Dimension),
         Biases           => Random.Init_Zeros (Output_Dimension),
         Pre_Activation   => (others => 0.0),
         Output           => (others => 0.0),
         Activation       => Activation,
         Weight_Gradients => (others => (others => 0.0)),
         Bias_Gradients   => (others => 0.0));
   end New_Layer_Zeros;

   procedure Forward (Self : in out Layer; Input : Vector) is
      Linear_Comb : Vector (1 .. Self.Output_Dimension);
   begin
      if Input'Length /= Self.Input_Dimension then
         raise Dimension_Error
           with "Input vector dimension does not match layer input dimension";
      end if;

      Linear_Comb := Multiply (Self.Weights, Input);
      Self.Pre_Activation := Add (Linear_Comb, Self.Biases);

      case Self.Activation is
         when Linear =>
            Self.Output := Self.Pre_Activation;
         when Sigmoid =>
            Self.Output := Sigmoid (Self.Pre_Activation);
         when ReLU =>
            Self.Output := ReLU (Self.Pre_Activation);
         when Tanh =>
            Self.Output := Tanh (Self.Pre_Activation);
         when Softmax =>
            Self.Output := Softmax (Self.Pre_Activation);
      end case;
   end Forward;

   function Predict (Self : in out Layer; Input : Vector) return Vector is
   begin
      Self.Forward (Input);
      return Self.Output;
   end Predict;

   procedure Backward
     (Self             : in out Layer;
      Output_Grad      : Vector;
      Input_Activation : Vector;
      Input_Grad       : out Vector)
   is
      Grad_Z     : Vector (1 .. Self.Output_Dimension);
      Deriv      : Vector (1 .. Self.Output_Dimension);
      Sum_Val    : Scalar;
      Grad_Z_Idx : Positive;
   begin
      if Output_Grad'Length /= Self.Output_Dimension
        or else Input_Activation'Length /= Self.Input_Dimension
      then
         raise Dimension_Error with "Dimension mismatch in Layer.Backward";
      end if;

      case Self.Activation is
         when Linear =>
            Deriv := (others => 1.0);
         when Sigmoid =>
            Deriv := Sigmoid_Derivative_From_Output (Self.Output);
         when ReLU =>
            Deriv := ReLU_Derivative_From_Output (Self.Output);
         when Tanh =>
            Deriv := Tanh_Derivative_From_Output (Self.Output);
         when Softmax =>
            Deriv := (others => 1.0);
      end case;

      Grad_Z := Hadamard (Output_Grad, Deriv);

      Self.Weight_Gradients :=
        Add (Self.Weight_Gradients, Outer_Product (Grad_Z, Input_Activation));
      Self.Bias_Gradients := Add (Self.Bias_Gradients, Grad_Z);

      for J in 1 .. Self.Input_Dimension loop
         Sum_Val    := 0.0;
         Grad_Z_Idx := Grad_Z'First;
         for I in 1 .. Self.Output_Dimension loop
            Sum_Val    :=
              Sum_Val + Self.Weights (I, J) * Grad_Z (Grad_Z_Idx);
            Grad_Z_Idx := Grad_Z_Idx + 1;
         end loop;
         Input_Grad (Input_Grad'First + J - 1) := Sum_Val;
      end loop;
   end Backward;

   procedure Update
     (Self          : in out Layer;
      Learning_Rate : Scalar)
   is
      Delta_W : constant Matrix :=
        Scale (Self.Weight_Gradients, Learning_Rate);
      Delta_B : constant Vector :=
        Scale (Self.Bias_Gradients, Learning_Rate);
   begin
      Self.Weights := Subtract (Self.Weights, Delta_W);
      Self.Biases  := Subtract (Self.Biases, Delta_B);
      Self.Zero_Grad;
   end Update;

   procedure Zero_Grad (Self : in out Layer) is
   begin
      Self.Weight_Gradients := (others => (others => 0.0));
      Self.Bias_Gradients   := (others => 0.0);
   end Zero_Grad;

   procedure Zero_Gradients (Self : in out Layer) is
   begin
      Self.Zero_Grad;
   end Zero_Gradients;

end Neural_Kit.Layer;
