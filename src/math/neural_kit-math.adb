with Ada.Numerics.Elementary_Functions;

package body Neural_Kit.Math is

   use Ada.Numerics.Elementary_Functions;

   function Dot (A, B : Vector) return Scalar is
      Sum   : Scalar := 0.0;
      B_Idx : Positive := B'First;
   begin
      if A'Length /= B'Length then
         raise Dimension_Error with "Vector lengths must match for Dot";
      end if;

      for A_Idx in A'Range loop
         Sum := Sum + A (A_Idx) * B (B_Idx);
         B_Idx := B_Idx + 1;
      end loop;

      return Sum;
   end Dot;

   function Add (A, B : Vector) return Vector is
      Result : Vector (1 .. A'Length);
      B_Idx  : Positive := B'First;
      Out_I  : Positive := 1;
   begin
      if A'Length /= B'Length then
         raise Dimension_Error with "Vector lengths must match for Add";
      end if;

      for A_Idx in A'Range loop
         Result (Out_I) := A (A_Idx) + B (B_Idx);
         B_Idx := B_Idx + 1;
         Out_I := Out_I + 1;
      end loop;

      return Result;
   end Add;

   function Subtract (A, B : Vector) return Vector is
      Result : Vector (1 .. A'Length);
      B_Idx  : Positive := B'First;
      Out_I  : Positive := 1;
   begin
      if A'Length /= B'Length then
         raise Dimension_Error with "Vector lengths must match for Subtract";
      end if;

      for A_Idx in A'Range loop
         Result (Out_I) := A (A_Idx) - B (B_Idx);
         B_Idx := B_Idx + 1;
         Out_I := Out_I + 1;
      end loop;

      return Result;
   end Subtract;

   function Hadamard (A, B : Vector) return Vector is
      Result : Vector (1 .. A'Length);
      B_Idx  : Positive := B'First;
      Out_I  : Positive := 1;
   begin
      if A'Length /= B'Length then
         raise Dimension_Error with "Vector lengths must match for Hadamard";
      end if;

      for A_Idx in A'Range loop
         Result (Out_I) := A (A_Idx) * B (B_Idx);
         B_Idx := B_Idx + 1;
         Out_I := Out_I + 1;
      end loop;

      return Result;
   end Hadamard;

   function Scale (V : Vector; Factor : Scalar) return Vector is
      Result : Vector (1 .. V'Length);
      Out_I  : Positive := 1;
   begin
      for I in V'Range loop
         Result (Out_I) := V (I) * Factor;
         Out_I := Out_I + 1;
      end loop;

      return Result;
   end Scale;

   function Sum (V : Vector) return Scalar is
      Total : Scalar := 0.0;
   begin
      for I in V'Range loop
         Total := Total + V (I);
      end loop;

      return Total;
   end Sum;

   function Mean (V : Vector) return Scalar is
   begin
      if V'Length = 0 then
         raise Dimension_Error with "Cannot compute mean of empty vector";
      end if;

      return Sum (V) / Scalar (V'Length);
   end Mean;

   function Multiply (M : Matrix; V : Vector) return Vector is
      Result  : Vector (1 .. M'Length (1));
      Sum_Val : Scalar;
      V_Idx   : Positive;
      M_Row   : Positive;
      M_Col0  : constant Positive := M'First (2);
   begin
      if M'Length (2) /= V'Length then
         raise Dimension_Error
           with "Matrix columns must equal vector length for Multiply";
      end if;

      for I in 1 .. M'Length (1) loop
         Sum_Val := 0.0;
         V_Idx := V'First;
         M_Row := M'First (1) + I - 1;
         for J in 1 .. M'Length (2) loop
            Sum_Val := Sum_Val + M (M_Row, M_Col0 + J - 1) * V (V_Idx);
            V_Idx := V_Idx + 1;
         end loop;
         Result (I) := Sum_Val;
      end loop;

      return Result;
   end Multiply;

   function Multiply (A, B : Matrix) return Matrix is
      Result  : Matrix (1 .. A'Length (1), 1 .. B'Length (2));
      Sum_Val : Scalar;
      A_Row   : Positive;
      A_Col0  : constant Positive := A'First (2);
      B_Row0  : constant Positive := B'First (1);
      B_Col   : Positive;
   begin
      if A'Length (2) /= B'Length (1) then
         raise Dimension_Error
           with "Inner dimensions must match for matrix Multiply";
      end if;

      for I in 1 .. A'Length (1) loop
         A_Row := A'First (1) + I - 1;
         for J in 1 .. B'Length (2) loop
            B_Col := B'First (2) + J - 1;
            Sum_Val := 0.0;
            for K in 1 .. A'Length (2) loop
               Sum_Val :=
                 Sum_Val
                 + A (A_Row, A_Col0 + K - 1) * B (B_Row0 + K - 1, B_Col);
            end loop;
            Result (I, J) := Sum_Val;
         end loop;
      end loop;

      return Result;
   end Multiply;

   function Add (A, B : Matrix) return Matrix is
      Result : Matrix (1 .. A'Length (1), 1 .. A'Length (2));
      A_Row0 : constant Positive := A'First (1);
      A_Col0 : constant Positive := A'First (2);
      B_Row0 : constant Positive := B'First (1);
      B_Col0 : constant Positive := B'First (2);
      A_Row  : Positive;
      B_Row  : Positive;
   begin
      if A'Length (1) /= B'Length (1) or else A'Length (2) /= B'Length (2) then
         raise Dimension_Error with "Matrix dimensions must match for Add";
      end if;

      for I in 1 .. A'Length (1) loop
         A_Row := A_Row0 + I - 1;
         B_Row := B_Row0 + I - 1;
         for J in 1 .. A'Length (2) loop
            Result (I, J) :=
              A (A_Row, A_Col0 + J - 1) + B (B_Row, B_Col0 + J - 1);
         end loop;
      end loop;

      return Result;
   end Add;

   function Subtract (A, B : Matrix) return Matrix is
      Result : Matrix (1 .. A'Length (1), 1 .. A'Length (2));
      A_Row0 : constant Positive := A'First (1);
      A_Col0 : constant Positive := A'First (2);
      B_Row0 : constant Positive := B'First (1);
      B_Col0 : constant Positive := B'First (2);
      A_Row  : Positive;
      B_Row  : Positive;
   begin
      if A'Length (1) /= B'Length (1) or else A'Length (2) /= B'Length (2) then
         raise Dimension_Error
           with "Matrix dimensions must match for Subtract";
      end if;

      for I in 1 .. A'Length (1) loop
         A_Row := A_Row0 + I - 1;
         B_Row := B_Row0 + I - 1;
         for J in 1 .. A'Length (2) loop
            Result (I, J) :=
              A (A_Row, A_Col0 + J - 1) - B (B_Row, B_Col0 + J - 1);
         end loop;
      end loop;

      return Result;
   end Subtract;

   function Hadamard (A, B : Matrix) return Matrix is
      Result : Matrix (1 .. A'Length (1), 1 .. A'Length (2));
      A_Row0 : constant Positive := A'First (1);
      A_Col0 : constant Positive := A'First (2);
      B_Row0 : constant Positive := B'First (1);
      B_Col0 : constant Positive := B'First (2);
      A_Row  : Positive;
      B_Row  : Positive;
   begin
      if A'Length (1) /= B'Length (1) or else A'Length (2) /= B'Length (2) then
         raise Dimension_Error
           with "Matrix dimensions must match for Hadamard";
      end if;

      for I in 1 .. A'Length (1) loop
         A_Row := A_Row0 + I - 1;
         B_Row := B_Row0 + I - 1;
         for J in 1 .. A'Length (2) loop
            Result (I, J) :=
              A (A_Row, A_Col0 + J - 1) * B (B_Row, B_Col0 + J - 1);
         end loop;
      end loop;

      return Result;
   end Hadamard;

   function Scale (M : Matrix; Factor : Scalar) return Matrix is
      Result : Matrix (1 .. M'Length (1), 1 .. M'Length (2));
      M_Row0 : constant Positive := M'First (1);
      M_Col0 : constant Positive := M'First (2);
      M_Row  : Positive;
   begin
      for I in 1 .. M'Length (1) loop
         M_Row := M_Row0 + I - 1;
         for J in 1 .. M'Length (2) loop
            Result (I, J) := M (M_Row, M_Col0 + J - 1) * Factor;
         end loop;
      end loop;

      return Result;
   end Scale;

   function Transpose (M : Matrix) return Matrix is
      Result : Matrix (1 .. M'Length (2), 1 .. M'Length (1));
      M_Row0 : constant Positive := M'First (1);
      M_Col0 : constant Positive := M'First (2);
   begin
      for I in 1 .. M'Length (1) loop
         for J in 1 .. M'Length (2) loop
            Result (J, I) := M (M_Row0 + I - 1, M_Col0 + J - 1);
         end loop;
      end loop;

      return Result;
   end Transpose;

   function Outer_Product (A, B : Vector) return Matrix is
      Result : Matrix (1 .. A'Length, 1 .. B'Length);
      A_Idx  : Positive := A'First;
      B_Idx  : Positive;
      A_Val  : Scalar;
   begin
      for I in 1 .. A'Length loop
         A_Val := A (A_Idx);
         B_Idx := B'First;
         for J in 1 .. B'Length loop
            Result (I, J) := A_Val * B (B_Idx);
            B_Idx := B_Idx + 1;
         end loop;
         A_Idx := A_Idx + 1;
      end loop;

      return Result;
   end Outer_Product;

   function Sigmoid (X : Scalar) return Scalar is
   begin
      if X < -45.0 then
         return 0.0;
      elsif X > 45.0 then
         return 1.0;
      else
         return 1.0 / (1.0 + Exp (-X));
      end if;
   end Sigmoid;

   function Sigmoid (V : Vector) return Vector is
      Result : Vector (1 .. V'Length);
      Out_I  : Positive := 1;
   begin
      for I in V'Range loop
         Result (Out_I) := Sigmoid (V (I));
         Out_I := Out_I + 1;
      end loop;

      return Result;
   end Sigmoid;

   function Sigmoid_Derivative_From_Output (Output : Scalar) return Scalar is
   begin
      return Output * (1.0 - Output);
   end Sigmoid_Derivative_From_Output;

   function Sigmoid_Derivative_From_Output (Output : Vector) return Vector is
      Result : Vector (1 .. Output'Length);
      Out_I  : Positive := 1;
   begin
      for I in Output'Range loop
         Result (Out_I) := Sigmoid_Derivative_From_Output (Output (I));
         Out_I := Out_I + 1;
      end loop;

      return Result;
   end Sigmoid_Derivative_From_Output;

   function ReLU (X : Scalar) return Scalar is
   begin
      if X > 0.0 then
         return X;
      else
         return 0.0;
      end if;
   end ReLU;

   function ReLU (V : Vector) return Vector is
      Result : Vector (1 .. V'Length);
      Out_I  : Positive := 1;
   begin
      for I in V'Range loop
         Result (Out_I) := ReLU (V (I));
         Out_I := Out_I + 1;
      end loop;

      return Result;
   end ReLU;

   function ReLU_Derivative_From_Output (Output : Scalar) return Scalar is
   begin
      if Output > 0.0 then
         return 1.0;
      else
         return 0.0;
      end if;
   end ReLU_Derivative_From_Output;

   function ReLU_Derivative_From_Output (Output : Vector) return Vector is
      Result : Vector (1 .. Output'Length);
      Out_I  : Positive := 1;
   begin
      for I in Output'Range loop
         Result (Out_I) := ReLU_Derivative_From_Output (Output (I));
         Out_I := Out_I + 1;
      end loop;

      return Result;
   end ReLU_Derivative_From_Output;

   function Tanh (X : Scalar) return Scalar is
   begin
      return Ada.Numerics.Elementary_Functions.Tanh (X);
   end Tanh;

   function Tanh (V : Vector) return Vector is
      Result : Vector (1 .. V'Length);
      Out_I  : Positive := 1;
   begin
      for I in V'Range loop
         Result (Out_I) := Tanh (V (I));
         Out_I := Out_I + 1;
      end loop;

      return Result;
   end Tanh;

   function Tanh_Derivative_From_Output (Output : Scalar) return Scalar is
   begin
      return 1.0 - Output ** 2;
   end Tanh_Derivative_From_Output;

   function Tanh_Derivative_From_Output (Output : Vector) return Vector is
      Result : Vector (1 .. Output'Length);
      Out_I  : Positive := 1;
   begin
      for I in Output'Range loop
         Result (Out_I) := Tanh_Derivative_From_Output (Output (I));
         Out_I := Out_I + 1;
      end loop;

      return Result;
   end Tanh_Derivative_From_Output;

   function Softmax (V : Vector) return Vector is
      Result  : Vector (1 .. V'Length);
      Max_Val : Scalar;
      Sum_Exp : Scalar := 0.0;
      Out_I   : Positive := 1;
   begin
      if V'Length = 0 then
         raise Dimension_Error with "Cannot compute softmax on empty vector";
      end if;

      Max_Val := V (V'First);
      for I in V'Range loop
         if not V (I)'Valid then
            raise Numerical_Error with "Invalid float value in softmax input";
         end if;
         if V (I) > Max_Val then
            Max_Val := V (I);
         end if;
      end loop;

      for I in V'Range loop
         Result (Out_I) := Exp (V (I) - Max_Val);
         Sum_Exp := Sum_Exp + Result (Out_I);
         Out_I := Out_I + 1;
      end loop;

      if not Sum_Exp'Valid or else Sum_Exp <= 0.0 then
         raise Numerical_Error
           with "Sum of exponents is zero or invalid in softmax";
      end if;

      for I in Result'Range loop
         Result (I) := Result (I) / Sum_Exp;
      end loop;

      return Result;
   end Softmax;

   function Binary_Cross_Entropy (Prediction, Target : Scalar) return Scalar is
      Safe_P : Scalar;
   begin
      if not Target'Valid or else Target < 0.0 or else Target > 1.0 then
         raise Numerical_Error with "Target must be within [0.0, 1.0]";
      end if;

      if not Prediction'Valid then
         raise Numerical_Error with "Prediction must be a valid finite number";
      end if;

      Safe_P := Scalar'Max (Epsilon, Scalar'Min (1.0 - Epsilon, Prediction));
      return -(Target * Log (Safe_P) + (1.0 - Target) * Log (1.0 - Safe_P));
   end Binary_Cross_Entropy;

   function Binary_Cross_Entropy (Predictions, Targets : Vector) return Scalar
   is
      Total_Loss  : Scalar := 0.0;
      Targets_Idx : Positive := Targets'First;
   begin
      if Predictions'Length /= Targets'Length then
         raise Dimension_Error
           with "Predictions and Targets length must match";
      end if;

      if Predictions'Length = 0 then
         raise Dimension_Error with "Cannot compute BCE for empty vectors";
      end if;

      for I in Predictions'Range loop
         Total_Loss :=
           Total_Loss
           + Binary_Cross_Entropy (Predictions (I), Targets (Targets_Idx));
         Targets_Idx := Targets_Idx + 1;
      end loop;

      return Total_Loss / Scalar (Predictions'Length);
   end Binary_Cross_Entropy;

   function Mean_Squared_Error (Predictions, Targets : Vector) return Scalar is
      Total_Sq_Error : Scalar := 0.0;
      Targets_Idx    : Positive := Targets'First;
   begin
      if Predictions'Length /= Targets'Length then
         raise Dimension_Error
           with "Predictions and Targets length must match";
      end if;

      if Predictions'Length = 0 then
         raise Dimension_Error with "Cannot compute MSE for empty vectors";
      end if;

      for I in Predictions'Range loop
         if not (Predictions (I)'Valid and then Targets (Targets_Idx)'Valid)
         then
            raise Numerical_Error with "Invalid float value in MSE input";
         end if;
         Total_Sq_Error :=
           Total_Sq_Error + (Predictions (I) - Targets (Targets_Idx)) ** 2;
         Targets_Idx := Targets_Idx + 1;
      end loop;

      return Total_Sq_Error / Scalar (Predictions'Length);
   end Mean_Squared_Error;

end Neural_Kit.Math;
