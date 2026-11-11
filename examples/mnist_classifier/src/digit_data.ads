with Neural_Kit.Types; use Neural_Kit.Types;

package Digit_Data is

   Pixel_Count : constant Positive := 64;
   Class_Count : constant Positive := 10;

   type Digit_Sample is record
      Pixels     : Vector (1 .. Pixel_Count);
      One_Hot    : Vector (1 .. Class_Count);
      True_Digit : Natural range 0 .. 9;
   end record;

   Train_Count : constant Positive := 40;
   Test_Count  : constant Positive := 20;

   type Train_Array is array (1 .. Train_Count) of Digit_Sample;
   type Test_Array is array (1 .. Test_Count) of Digit_Sample;

   function To_One_Hot (D : Natural) return Vector;

   function Training_Sample (Index : Positive) return Digit_Sample
     with Pre => Index in 1 .. Train_Count;

   function Testing_Sample (Index : Positive) return Digit_Sample
     with Pre => Index in 1 .. Test_Count;

end Digit_Data;
