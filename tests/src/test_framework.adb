with Ada.Text_IO; use Ada.Text_IO;

package body Test_Framework is

   Total_Tests  : Natural := 0;
   Total_Passed : Natural := 0;
   Total_Failed : Natural := 0;

   Current_Suite_Tests  : Natural := 0;
   Current_Suite_Passed : Natural := 0;
   Current_Suite_Failed : Natural := 0;

   procedure Start_Suite (Suite_Name : String) is
   begin
      Current_Suite_Tests := 0;
      Current_Suite_Passed := 0;
      Current_Suite_Failed := 0;
      Put_Line ("=== Running Suite: " & Suite_Name & " ===");
   end Start_Suite;

   procedure End_Suite is
   begin
      Put_Line
        ("  Suite complete: "
         & Integer'Image (Current_Suite_Passed)
         & " / "
         & Integer'Image (Current_Suite_Tests)
         & " passed.");
      Put_Line ("");
   end End_Suite;

   procedure Assert_True
     (Condition : Boolean;
      Message   : String)
   is
   begin
      Total_Tests := Total_Tests + 1;
      Current_Suite_Tests := Current_Suite_Tests + 1;

      if Condition then
         Total_Passed := Total_Passed + 1;
         Current_Suite_Passed := Current_Suite_Passed + 1;
      else
         Total_Failed := Total_Failed + 1;
         Current_Suite_Failed := Current_Suite_Failed + 1;
         Put_Line ("  [FAIL] " & Message);
      end if;
   end Assert_True;

   procedure Assert_Almost_Equal
     (Actual    : Scalar;
      Expected  : Scalar;
      Tolerance : Scalar := 1.0e-5;
      Message   : String := "")
   is
      Diff : constant Scalar := abs (Actual - Expected);
      Cond : constant Boolean := Diff <= Tolerance;
      Msg  : constant String :=
        (if Message'Length > 0 then Message & ": " else "")
        & "Expected ~"
        & Scalar'Image (Expected)
        & ", got "
        & Scalar'Image (Actual);
   begin
      Assert_True (Cond, Msg);
   end Assert_Almost_Equal;

   procedure Assert_Vector_Almost_Equal
     (Actual    : Vector;
      Expected  : Vector;
      Tolerance : Scalar := 1.0e-5;
      Message   : String := "")
   is
      Match : Boolean;
      Act_I : Positive;
      Exp_I : Positive;
   begin
      if Actual'Length /= Expected'Length then
         Assert_True
           (False,
            (if Message'Length > 0 then Message & ": " else "")
            & "Vector length mismatch");
         return;
      end if;

      Match := True;
      Act_I := Actual'First;
      Exp_I := Expected'First;

      for I in 1 .. Actual'Length loop
         if abs (Actual (Act_I) - Expected (Exp_I)) > Tolerance then
            Match := False;
            exit;
         end if;
         Act_I := Act_I + 1;
         Exp_I := Exp_I + 1;
      end loop;

      Assert_True
        (Match,
         (if Message'Length > 0 then Message & ": " else "")
         & "Vector values mismatch");
   end Assert_Vector_Almost_Equal;

   procedure Assert_Matrix_Almost_Equal
     (Actual    : Matrix;
      Expected  : Matrix;
      Tolerance : Scalar := 1.0e-5;
      Message   : String := "")
   is
      Match : Boolean;
   begin
      if Actual'Length (1) /= Expected'Length (1)
        or else Actual'Length (2) /= Expected'Length (2)
      then
         Assert_True
           (False,
            (if Message'Length > 0 then Message & ": " else "")
            & "Matrix dimension mismatch");
         return;
      end if;

      Match := True;
      for I in 1 .. Actual'Length (1) loop
         for J in 1 .. Actual'Length (2) loop
            if abs
              (Actual (Actual'First (1) + I - 1, Actual'First (2) + J - 1)
               - Expected
                 (Expected'First (1) + I - 1, Expected'First (2) + J - 1))
              > Tolerance
            then
               Match := False;
               exit;
            end if;
         end loop;
         exit when not Match;
      end loop;

      Assert_True
        (Match,
         (if Message'Length > 0 then Message & ": " else "")
         & "Matrix values mismatch");
   end Assert_Matrix_Almost_Equal;

   function All_Passed return Boolean is
   begin
      return Total_Failed = 0;
   end All_Passed;

   procedure Print_Summary is
   begin
      Put_Line ("========================================");
      Put_Line
        ("Test Summary: "
         & Integer'Image (Total_Passed)
         & " passed,"
         & Integer'Image (Total_Failed)
         & " failed out of "
         & Integer'Image (Total_Tests)
         & " total assertions.");
      if Total_Failed = 0 then
         Put_Line ("ALL TESTS PASSED.");
      else
         Put_Line ("SOME TESTS FAILED.");
      end if;
      Put_Line ("========================================");
   end Print_Summary;

end Test_Framework;

