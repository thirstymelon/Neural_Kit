with Neural_Kit.Types; use Neural_Kit.Types;

package Neural_Kit.Math is

   Epsilon : constant Scalar := 1.0e-7;

   function Dot (A, B : Vector) return Scalar
     with Pre => A'Length = B'Length;

   function Add (A, B : Vector) return Vector
     with Pre  => A'Length = B'Length,
          Post => Add'Result'Length = A'Length
          and then Add'Result'First = 1;

   function Subtract (A, B : Vector) return Vector
     with Pre  => A'Length = B'Length,
          Post => Subtract'Result'Length = A'Length
          and then Subtract'Result'First = 1;

   function Hadamard (A, B : Vector) return Vector
     with Pre  => A'Length = B'Length,
          Post => Hadamard'Result'Length = A'Length
          and then Hadamard'Result'First = 1;

   function Scale (V : Vector; Factor : Scalar) return Vector
     with Post => Scale'Result'Length = V'Length
          and then Scale'Result'First = 1;

   function Sum (V : Vector) return Scalar;

   function Mean (V : Vector) return Scalar
     with Pre => V'Length > 0;

   function Multiply (M : Matrix; V : Vector) return Vector
     with Pre  => M'Length (2) = V'Length,
          Post => Multiply'Result'Length = M'Length (1)
          and then Multiply'Result'First = 1;

   function Multiply (A, B : Matrix) return Matrix
     with Pre  => A'Length (2) = B'Length (1),
          Post => Multiply'Result'Length (1) = A'Length (1)
          and then Multiply'Result'Length (2) = B'Length (2)
          and then Multiply'Result'First (1) = 1
          and then Multiply'Result'First (2) = 1;

   function Add (A, B : Matrix) return Matrix
     with Pre  => A'Length (1) = B'Length (1)
          and then A'Length (2) = B'Length (2),
          Post => Add'Result'Length (1) = A'Length (1)
          and then Add'Result'Length (2) = A'Length (2)
          and then Add'Result'First (1) = 1
          and then Add'Result'First (2) = 1;

   function Subtract (A, B : Matrix) return Matrix
     with Pre  => A'Length (1) = B'Length (1)
          and then A'Length (2) = B'Length (2),
          Post => Subtract'Result'Length (1) = A'Length (1)
          and then Subtract'Result'Length (2) = A'Length (2)
          and then Subtract'Result'First (1) = 1
          and then Subtract'Result'First (2) = 1;

   function Hadamard (A, B : Matrix) return Matrix
     with Pre  => A'Length (1) = B'Length (1)
          and then A'Length (2) = B'Length (2),
          Post => Hadamard'Result'Length (1) = A'Length (1)
          and then Hadamard'Result'Length (2) = A'Length (2)
          and then Hadamard'Result'First (1) = 1
          and then Hadamard'Result'First (2) = 1;

   function Scale (M : Matrix; Factor : Scalar) return Matrix
     with Post => Scale'Result'Length (1) = M'Length (1)
          and then Scale'Result'Length (2) = M'Length (2)
          and then Scale'Result'First (1) = 1
          and then Scale'Result'First (2) = 1;

   function Transpose (M : Matrix) return Matrix
     with Post => Transpose'Result'Length (1) = M'Length (2)
          and then Transpose'Result'Length (2) = M'Length (1)
          and then Transpose'Result'First (1) = 1
          and then Transpose'Result'First (2) = 1;

   function Outer_Product (A, B : Vector) return Matrix
     with Post => Outer_Product'Result'Length (1) = A'Length
          and then Outer_Product'Result'Length (2) = B'Length
          and then Outer_Product'Result'First (1) = 1
          and then Outer_Product'Result'First (2) = 1;

   function Sigmoid (X : Scalar) return Scalar;
   function Sigmoid (V : Vector) return Vector
     with Post => Sigmoid'Result'Length = V'Length
          and then Sigmoid'Result'First = 1;

   function Sigmoid_Derivative_From_Output (Output : Scalar) return Scalar;
   function Sigmoid_Derivative_From_Output (Output : Vector) return Vector
     with Post => Sigmoid_Derivative_From_Output'Result'Length = Output'Length
          and then Sigmoid_Derivative_From_Output'Result'First = 1;

   function ReLU (X : Scalar) return Scalar;
   function ReLU (V : Vector) return Vector
     with Post => ReLU'Result'Length = V'Length
          and then ReLU'Result'First = 1;

   function ReLU_Derivative_From_Output (Output : Scalar) return Scalar;
   function ReLU_Derivative_From_Output (Output : Vector) return Vector
     with Post => ReLU_Derivative_From_Output'Result'Length = Output'Length
          and then ReLU_Derivative_From_Output'Result'First = 1;

   function Tanh (X : Scalar) return Scalar;
   function Tanh (V : Vector) return Vector
     with Post => Tanh'Result'Length = V'Length
          and then Tanh'Result'First = 1;

   function Tanh_Derivative_From_Output (Output : Scalar) return Scalar;
   function Tanh_Derivative_From_Output (Output : Vector) return Vector
     with Post => Tanh_Derivative_From_Output'Result'Length = Output'Length
          and then Tanh_Derivative_From_Output'Result'First = 1;

   function Softmax (V : Vector) return Vector
     with Pre  => V'Length > 0,
          Post => Softmax'Result'Length = V'Length
          and then Softmax'Result'First = 1;

   function Binary_Cross_Entropy
     (Prediction, Target : Scalar) return Scalar
     with Pre => Target >= 0.0 and then Target <= 1.0;

   function Binary_Cross_Entropy
     (Predictions, Targets : Vector) return Scalar
     with Pre => Predictions'Length = Targets'Length
          and then Predictions'Length > 0;

   function Mean_Squared_Error
     (Predictions, Targets : Vector) return Scalar
     with Pre => Predictions'Length = Targets'Length
          and then Predictions'Length > 0;

end Neural_Kit.Math;
