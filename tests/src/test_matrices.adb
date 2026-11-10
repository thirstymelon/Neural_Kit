with Ada.Assertions;
with Neural_Kit.Math;  use Neural_Kit.Math;
with Neural_Kit.Types; use Neural_Kit.Types;
with Test_Framework;   use Test_Framework;

package body Test_Matrices is

   procedure Run is
      --  2 output neurons, 3 input features
      M : constant Matrix (1 .. 2, 1 .. 3) :=
        ((1.0, 2.0, 3.0),
         (4.0, 5.0, 6.0));

      V : constant Vector (1 .. 3) := (1.0, 2.0, 3.0);

      B : constant Matrix (1 .. 3, 1 .. 2) :=
        ((1.0, 2.0),
         (3.0, 4.0),
         (5.0, 6.0));

      M_Offset : constant Matrix (4 .. 5, 10 .. 12) :=
        ((1.0, 2.0, 3.0),
         (4.0, 5.0, 6.0));

      V_Offset : constant Vector (20 .. 22) := (1.0, 2.0, 3.0);
   begin
      Start_Suite ("Matrix Operations");

      --  Matrix-Vector Multiplication: Result = M * V
      --  Row 1: 1*1 + 2*2 + 3*3 = 14
      --  Row 2: 4*1 + 5*2 + 6*3 = 32
      declare
         Res : constant Vector := Multiply (M, V);
         Expected : constant Vector (1 .. 2) := (14.0, 32.0);
      begin
         Assert_Vector_Almost_Equal
           (Res, Expected, Message => "Matrix-Vector Multiply");
         Assert_True
           (Res'First = 1 and then Res'Length = 2,
            Message => "Matrix-Vector Multiply returns normalized rows");
      end;

      --  Matrix-Vector with offset bounds
      declare
         Res : constant Vector := Multiply (M_Offset, V_Offset);
         Expected : constant Vector (1 .. 2) := (14.0, 32.0);
      begin
         Assert_Vector_Almost_Equal
           (Res, Expected, Message => "Matrix-Vector Multiply offset bounds");
      end;

      --  Matrix-Matrix Multiplication: M(2x3) * B(3x2) -> Result(2x2)
      --  Row 1, Col 1: 1*1 + 2*3 + 3*5 = 22
      --  Row 1, Col 2: 1*2 + 2*4 + 3*6 = 28
      --  Row 2, Col 1: 4*1 + 5*3 + 6*5 = 49
      --  Row 2, Col 2: 4*2 + 5*4 + 6*6 = 64
      declare
         Res : constant Matrix := Multiply (M, B);
         Expected : constant Matrix (1 .. 2, 1 .. 2) :=
           ((22.0, 28.0),
            (49.0, 64.0));
      begin
         Assert_Matrix_Almost_Equal
           (Res, Expected, Message => "Matrix-Matrix Multiply");
         Assert_True
           (Res'First (1) = 1 and then Res'First (2) = 1,
            Message => "Matrix Multiply returns 1-based bounds");
      end;

      --  Matrix Add and Subtract
      declare
         M2 : constant Matrix (1 .. 2, 1 .. 3) :=
           ((2.0, 1.0, 0.0),
            (1.0, 2.0, 3.0));
         Res_Add : constant Matrix := Add (M, M2);
         Res_Sub : constant Matrix := Subtract (M, M2);
         Expected_Add : constant Matrix (1 .. 2, 1 .. 3) :=
           ((3.0, 3.0, 3.0),
            (5.0, 7.0, 9.0));
         Expected_Sub : constant Matrix (1 .. 2, 1 .. 3) :=
           ((-1.0, 1.0, 3.0),
            (3.0, 3.0, 3.0));
      begin
         Assert_Matrix_Almost_Equal
           (Res_Add, Expected_Add, Message => "Matrix Add");
         Assert_Matrix_Almost_Equal
           (Res_Sub, Expected_Sub, Message => "Matrix Subtract");
      end;

      --  Matrix Hadamard
      declare
         M_Identity_Like : constant Matrix (1 .. 2, 1 .. 3) :=
           ((2.0, 3.0, 4.0),
            (0.5, 1.0, 2.0));
         Res : constant Matrix := Hadamard (M, M_Identity_Like);
         Expected : constant Matrix (1 .. 2, 1 .. 3) :=
           ((2.0, 6.0, 12.0),
            (2.0, 5.0, 12.0));
      begin
         Assert_Matrix_Almost_Equal
           (Res, Expected, Message => "Matrix Hadamard");
      end;

      --  Matrix Scale
      declare
         Res : constant Matrix := Scale (M, 2.0);
         Expected : constant Matrix (1 .. 2, 1 .. 3) :=
           ((2.0, 4.0, 6.0),
            (8.0, 10.0, 12.0));
      begin
         Assert_Matrix_Almost_Equal
           (Res, Expected, Message => "Matrix Scale");
      end;

      --  Matrix Transpose
      declare
         Res : constant Matrix := Transpose (M);
         Expected : constant Matrix (1 .. 3, 1 .. 2) :=
           ((1.0, 4.0),
            (2.0, 5.0),
            (3.0, 6.0));
      begin
         Assert_Matrix_Almost_Equal
           (Res, Expected, Message => "Matrix Transpose");
      end;

      --  Outer Product: A(2) x B(3) -> 2x3 Matrix
      declare
         V1 : constant Vector (1 .. 2) := (1.0, 2.0);
         V2 : constant Vector (1 .. 3) := (3.0, 4.0, 5.0);
         Res : constant Matrix := Outer_Product (V1, V2);
         Expected : constant Matrix (1 .. 2, 1 .. 3) :=
           ((3.0, 4.0, 5.0),
            (6.0, 8.0, 10.0));
      begin
         Assert_Matrix_Almost_Equal
           (Res, Expected, Message => "Outer Product");
      end;

      --  Dimension Mismatch in Multiply
      declare
         Failed : Boolean := False;
         Bad_V  : constant Vector (1 .. 2) := (1.0, 2.0);
      begin
         begin
            declare
               Dummy : constant Vector := Multiply (M, Bad_V);
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
            Message => "Matrix-Vector mismatch raises Assertion/Dimension");
      end;

      End_Suite;
   end Run;

end Test_Matrices;
