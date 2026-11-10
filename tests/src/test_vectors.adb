with Ada.Assertions;
with Neural_Kit.Math;  use Neural_Kit.Math;
with Neural_Kit.Types; use Neural_Kit.Types;
with Test_Framework;   use Test_Framework;

package body Test_Vectors is

   procedure Run is
      A : constant Vector (1 .. 3) := (1.0, 2.0, 3.0);
      B : constant Vector (1 .. 3) := (4.0, 5.0, 6.0);

      --  Vectors with non-1 lower bounds
      A_Offset : constant Vector (5 .. 7) := (1.0, 2.0, 3.0);
      B_Offset : constant Vector (10 .. 12) := (4.0, 5.0, 6.0);

      Mismatch_Dim : constant Vector (1 .. 2) := (1.0, 2.0);
   begin
      Start_Suite ("Vector Operations");

      --  Dot Product
      Assert_Almost_Equal
        (Dot (A, B),
         32.0,
         Message => "Dot product standard bounds (1*4 + 2*5 + 3*6)");

      Assert_Almost_Equal
        (Dot (A_Offset, B_Offset),
         32.0,
         Message => "Dot product with arbitrary lower bounds (5..7 and 10..12)");

      --  Add
      declare
         Res : constant Vector := Add (A, B);
         Expected : constant Vector (1 .. 3) := (5.0, 7.0, 9.0);
      begin
         Assert_Vector_Almost_Equal (Res, Expected, Message => "Vector Add");
         Assert_True
           (Res'First = 1 and then Res'Last = 3,
            Message => "Add returns normalized 1-based array");
      end;

      --  Add with offset bounds
      declare
         Res : constant Vector := Add (A_Offset, B_Offset);
         Expected : constant Vector (1 .. 3) := (5.0, 7.0, 9.0);
      begin
         Assert_Vector_Almost_Equal
           (Res, Expected, Message => "Vector Add offset bounds");
         Assert_True
           (Res'First = 1,
            Message => "Add offset returns normalized 1-based array");
      end;

      --  Subtract
      declare
         Res : constant Vector := Subtract (A => B, B => A);
         Expected : constant Vector (1 .. 3) := (3.0, 3.0, 3.0);
      begin
         Assert_Vector_Almost_Equal
           (Res, Expected, Message => "Vector Subtract");
      end;

      --  Hadamard (element-wise multiplication)
      declare
         Res : constant Vector := Hadamard (A, B);
         Expected : constant Vector (1 .. 3) := (4.0, 10.0, 18.0);
      begin
         Assert_Vector_Almost_Equal
           (Res, Expected, Message => "Vector Hadamard");
      end;

      --  Scale
      declare
         Res : constant Vector := Scale (A, 2.5);
         Expected : constant Vector (1 .. 3) := (2.5, 5.0, 7.5);
      begin
         Assert_Vector_Almost_Equal (Res, Expected, Message => "Vector Scale");
      end;

      --  Sum and Mean
      Assert_Almost_Equal (Sum (A), 6.0, Message => "Vector Sum");
      Assert_Almost_Equal (Mean (A), 2.0, Message => "Vector Mean");

      --  Dimension mismatch validation
      declare
         Failed : Boolean := False;
      begin
         begin
            declare
               Dummy : constant Scalar := Dot (A, Mismatch_Dim);
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
            Message => "Dot with mismatched lengths raises Assertion/Dimension");
      end;

      End_Suite;
   end Run;

end Test_Vectors;
