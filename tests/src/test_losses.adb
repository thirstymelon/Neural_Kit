with Ada.Assertions;
with Neural_Kit.Math;  use Neural_Kit.Math;
with Neural_Kit.Types; use Neural_Kit.Types;
with Test_Framework;   use Test_Framework;

package body Test_Losses is

   procedure Run is
   begin
      Start_Suite ("Loss Functions");

      --  BCE for perfect predictions
      Assert_Almost_Equal
        (Binary_Cross_Entropy (1.0, 1.0),
         0.0,
         Tolerance => 1.0e-4,
         Message   => "BCE(1, 1) near 0");

      Assert_Almost_Equal
        (Binary_Cross_Entropy (0.0, 0.0),
         0.0,
         Tolerance => 1.0e-4,
         Message   => "BCE(0, 0) near 0");

      --  BCE at 0.5 prediction
      Assert_Almost_Equal
        (Binary_Cross_Entropy (0.5, 1.0),
         0.693147,
         Tolerance => 1.0e-4,
         Message   => "BCE(0.5, 1) = ln(2)");

      --  BCE boundary resilience (avoiding NaN or -Inf)
      declare
         Loss_Val : constant Scalar := Binary_Cross_Entropy (0.0, 1.0);
      begin
         Assert_True
           (Loss_Val > 10.0 and then Loss_Val < 25.0,
            Message => "BCE(0, 1) clamped safely without NaN or infinity");
      end;

      --  BCE Vector
      declare
         P : constant Vector (1 .. 2) := (0.8, 0.2);
         T : constant Vector (1 .. 2) := (1.0, 0.0);
         Loss_Val : constant Scalar := Binary_Cross_Entropy (P, T);
      begin
         Assert_True
           (Loss_Val > 0.0 and then Loss_Val < 0.5,
            Message => "BCE Vector computed accurately");
      end;

      --  BCE Invalid Target Domain check
      declare
         Failed : Boolean := False;
      begin
         begin
            declare
               Dummy : constant Scalar := Binary_Cross_Entropy (0.5, 1.5);
               pragma Unreferenced (Dummy);
            begin
               null;
            end;
         exception
            when Ada.Assertions.Assertion_Error | Numerical_Error =>
               Failed := True;
         end;
         Assert_True
           (Failed,
            Message => "BCE target > 1.0 raises Assertion/Numerical_Error");
      end;

      --  MSE
      declare
         P : constant Vector (1 .. 2) := (1.0, 2.0);
         T_Match : constant Vector (1 .. 2) := (1.0, 2.0);
         T_Diff  : constant Vector (1 .. 2) := (3.0, 4.0);
      begin
         Assert_Almost_Equal
           (Mean_Squared_Error (P, T_Match),
            0.0,
            Message => "MSE identical vectors = 0.0");
         Assert_Almost_Equal
           (Mean_Squared_Error (P, T_Diff),
            4.0,
            Message => "MSE ((1-3)**2 + (2-4)**2) / 2 = 4.0");
      end;

      --  MSE Mismatched Lengths
      declare
         Failed : Boolean := False;
         P : constant Vector (1 .. 2) := (1.0, 2.0);
         T : constant Vector (1 .. 3) := (1.0, 2.0, 3.0);
      begin
         begin
            declare
               Dummy : constant Scalar := Mean_Squared_Error (P, T);
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
            Message => "MSE mismatched lengths raises Assertion/Dimension_Error");
      end;

      End_Suite;
   end Run;

end Test_Losses;
