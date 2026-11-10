with Neural_Kit.Types; use Neural_Kit.Types;

package Test_Framework is

   procedure Assert_True
     (Condition : Boolean;
      Message   : String);

   procedure Assert_Almost_Equal
     (Actual    : Scalar;
      Expected  : Scalar;
      Tolerance : Scalar := 1.0e-5;
      Message   : String := "");

   procedure Assert_Vector_Almost_Equal
     (Actual    : Vector;
      Expected  : Vector;
      Tolerance : Scalar := 1.0e-5;
      Message   : String := "");

   procedure Assert_Matrix_Almost_Equal
     (Actual    : Matrix;
      Expected  : Matrix;
      Tolerance : Scalar := 1.0e-5;
      Message   : String := "");

   procedure Start_Suite (Suite_Name : String);

   procedure End_Suite;

   function All_Passed return Boolean;

   procedure Print_Summary;

end Test_Framework;

